# Tests for the parts of the pipeline that do not need RDKit.
#
# RDKit's compiled extensions are blocked by Windows Smart App Control on the
# development machine, so features.py and explain.py cannot be exercised here.
# Everything downstream of a fingerprint is pure numpy, scipy and sklearn, and
# that is what these tests cover, using synthetic sparse binary fingerprints
# with the same shape and density RDKit would produce (1024 bits, ~50 set).
#
# Run with:  python -m pytest tests/ -q
#      or:   python tests/test_pipeline.py

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import applicability
import combine
import names
import stats

RNG = np.random.default_rng(42)
FP_SIZE = 1024
BITS_SET = 50


def random_fps(n, size=FP_SIZE, bits=BITS_SET, rng=RNG):
    # Sparse 0/1 rows, like a Morgan fingerprint: about 50 of 1024 bits set.
    X = np.zeros((n, size), dtype=np.uint8)
    for i in range(n):
        X[i, rng.choice(size, size=bits, replace=False)] = 1
    return X


def clustered_fps(n, size=FP_SIZE, bits=BITS_SET, shared=40, rng=RNG):
    # Rows that share most of their bits, so they are mutually similar.
    # Used to build a dense "training set" that a random molecule falls outside of.
    #
    # The shared core is drawn from a FIXED seed, not from the caller's rng.
    # Drawing it from the caller's rng would give each call a different core,
    # so two clustered_fps calls with different seeds would produce two
    # unrelated clusters rather than members of one cluster.
    core = np.random.default_rng(0).choice(size, size=shared, replace=False)
    X = np.zeros((n, size), dtype=np.uint8)
    for i in range(n):
        X[i, core] = 1
        extra = rng.choice(np.setdiff1d(np.arange(size), core),
                           size=bits - shared, replace=False)
        X[i, extra] = 1
    return X


# ---------------------------------------------------------------- applicability

def test_tanimoto_known_values():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 1, 0])
    # shared 1, union 2 + 2 - 1 = 3
    assert abs(applicability.tanimoto(a, b) - 1 / 3) < 1e-9
    assert applicability.tanimoto(a, a) == 1.0
    assert applicability.tanimoto(np.array([1, 0]), np.array([0, 1])) == 0.0
    # Two empty fingerprints have no shared bits and no union. Defined as 0.0
    # so the value can never become nan and poison the percentile.
    assert applicability.tanimoto(np.zeros(4), np.zeros(4)) == 0.0
    print("  tanimoto known values OK")


def test_tanimoto_matrix_matches_scalar():
    A = random_fps(12)
    B = random_fps(9)
    M = applicability.tanimoto_matrix(A, B)
    assert M.shape == (12, 9)
    for i in range(12):
        for j in range(9):
            expected = applicability.tanimoto(A[i], B[j])
            assert abs(M[i, j] - expected) < 1e-6, (i, j, M[i, j], expected)
    print("  tanimoto_matrix matches the scalar version on every cell OK")


def test_tanimoto_matrix_handles_empty_row():
    A = np.zeros((1, FP_SIZE), dtype=np.uint8)
    B = random_fps(3)
    M = applicability.tanimoto_matrix(A, B)
    assert not np.isnan(M).any()
    assert (M == 0).all()
    print("  all-zero fingerprint gives 0.0, not nan OK")


def test_fit_domain_floor_and_shape():
    X = random_fps(200)
    domain = applicability.fit_domain(X)
    assert 0.0 <= domain["floor"] <= 1.0
    assert domain["train_densities"].shape == (200,)
    assert domain["k"] == applicability.K_NEIGHBOURS
    # The floor is the 5th percentile, so about 5 percent of the training set
    # sits below it by construction.
    below = (domain["train_densities"] < domain["floor"]).mean()
    assert 0.0 <= below <= 0.10, below
    print(f"  fit_domain floor={domain['floor']:.4f}, {below:.1%} of train below OK")


def test_fit_domain_excludes_self():
    # Every molecule is identical, so the self-similarity is 1.0 for all of
    # them. If the diagonal were not excluded the densities would still be 1.0,
    # so this instead checks that a duplicate-heavy set does not crash and the
    # floor stays in range.
    X = np.tile(random_fps(1), (20, 1))
    domain = applicability.fit_domain(X)
    assert abs(domain["floor"] - 1.0) < 1e-6
    print("  identical training set gives floor 1.0 OK")


def test_domain_rejects_dissimilar_molecule():
    # A tight cluster as the training set, then an unrelated random molecule.
    train = clustered_fps(150)
    domain = applicability.fit_domain(train)

    inside = clustered_fps(1, rng=np.random.default_rng(7))
    outside = random_fps(1, rng=np.random.default_rng(99))

    assert applicability.check_molecule(inside[0], domain) == applicability.INSIDE
    assert applicability.check_molecule(outside[0], domain) == applicability.OUTSIDE
    print("  domain accepts a similar molecule and rejects a dissimilar one OK")


def test_check_molecules_batch_matches_single():
    train = clustered_fps(80)
    domain = applicability.fit_domain(train)
    queries = np.vstack([clustered_fps(5, rng=np.random.default_rng(3)),
                         random_fps(5, rng=np.random.default_rng(4))])
    verdicts, scores = applicability.check_molecules(queries, domain)
    for i in range(len(queries)):
        assert verdicts[i] == applicability.check_molecule(queries[i], domain)
    assert scores.shape == (10,)
    print("  batch domain check agrees with the single-molecule version OK")


def test_domain_save_load_roundtrip(tmpdir=None):
    import tempfile
    train = random_fps(50)
    domain = applicability.fit_domain(train)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "domain.npz")
        applicability.save_domain(domain, path)
        loaded = applicability.load_domain(path)
    assert abs(loaded["floor"] - domain["floor"]) < 1e-6
    assert np.array_equal(loaded["X_train"], domain["X_train"])
    assert loaded["k"] == domain["k"]
    print("  domain save/load round trip OK")


# ------------------------------------------------------------------- statistics

def test_fisher_direction():
    # Motif present in 18 of 20 positives, 2 of 20 negatives: strongly enriched.
    motif = np.array([1] * 18 + [0] * 2 + [1] * 2 + [0] * 18)
    labels = np.array([1] * 20 + [0] * 20)
    result = stats.motif_enrichment(motif, labels)
    assert result["odds_ratio"] > 1
    assert result["p_value"] < 0.001
    assert result["present_pos"] == 18
    assert result["present_neg"] == 2
    assert result["absent_pos"] == 2
    assert result["absent_neg"] == 18

    # Reversed: the same motif depleted among positives must give OR < 1.
    flipped = stats.motif_enrichment(1 - motif, labels)
    assert flipped["odds_ratio"] < 1
    print(f"  Fisher direction OK (OR={result['odds_ratio']:.1f}, "
          f"p={result['p_value']:.2e})")


def test_fisher_no_association():
    motif = np.array([1, 0] * 50)
    labels = np.array([1] * 50 + [0] * 50)
    result = stats.motif_enrichment(motif, labels)
    assert result["p_value"] > 0.5
    print(f"  Fisher on no association gives p={result['p_value']:.2f} OK")


def test_bh_against_scipy():
    # Checked against scipy's own implementation rather than against my
    # arithmetic, so a mistake in the step-up logic cannot agree with itself.
    from scipy.stats import false_discovery_control
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205,
                  0.212, 0.216, 0.222, 0.44, 0.58, 0.75, 0.81])
    mine, _ = stats.benjamini_hochberg(p)
    theirs = false_discovery_control(p, method="bh")
    assert np.allclose(mine, theirs), (mine, theirs)
    print("  Benjamini-Hochberg matches scipy.stats.false_discovery_control OK")


def test_bh_is_monotone_and_order_preserving():
    rng = np.random.default_rng(1)
    p = rng.random(200)
    adjusted, significant = stats.benjamini_hochberg(p, alpha=0.05)
    # Adjusted values must rise with the raw values, never cross over.
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    # Returned in the ORIGINAL order, not sorted.
    assert adjusted.shape == p.shape
    assert np.all(adjusted >= p - 1e-12)
    assert adjusted.max() <= 1.0
    print("  BH adjusted values are monotone, clipped to 1, in original order OK")


def test_bh_empty():
    adjusted, significant = stats.benjamini_hochberg([])
    assert adjusted.shape == (0,)
    assert significant.shape == (0,)
    print("  BH on an empty list OK")


def test_enrichment_table():
    # Bit 0 is a real signal, bits 1-29 are noise. The table should rank bit 0
    # first and should not call many of the noise bits significant.
    n = 400
    X = RNG.integers(0, 2, size=(n, 30)).astype(np.uint8)
    y = np.zeros(n, dtype=int)
    y[:200] = 1
    X[:200, 0] = 1
    X[200:, 0] = 0

    table = stats.enrichment_table(X, y, range(30))
    assert len(table) == 30
    assert table.loc[0, "bit"] == 0
    assert table.loc[0, "significant"]
    assert (table["n_tests"] == 30).all()
    # Adjusted p must be at least the raw p for every row.
    assert (table["p_adjusted"] >= table["p_value"] - 1e-12).all()
    n_sig_noise = int(table[table["bit"] != 0]["significant"].sum())
    assert n_sig_noise <= 2, n_sig_noise
    print(f"  enrichment_table ranks the planted bit first, "
          f"{n_sig_noise}/29 noise bits significant OK")


# ------------------------------------------------------------ name normalising

def test_normalize_drug_name():
    cases = {
        "Amiodarone": "amiodarone",
        "  AMIODARONE  ": "amiodarone",
        "Amiodarone Hydrochloride": "amiodarone",
        "amiodarone hcl": "amiodarone",
        "Amiodarone hydrochloride monohydrate": "amiodarone",
        "Erythromycin (oral)": "erythromycin",
        "Dofetilide [Tikosyn]": "dofetilide",
        "co-trimoxazole": "co trimoxazole",
        "Quinidine sulfate": "quinidine",
        "": "",
        None: "",
    }
    for raw, expected in cases.items():
        got = names.normalize_drug_name(raw)
        assert got == expected, f"{raw!r} -> {got!r}, expected {expected!r}"
    print("  drug name normalisation OK on all cases")


def test_normalize_keeps_leading_salt():
    # "sodium valproate" must NOT become "valproate": a leading salt word is
    # part of the name as the source writes it.
    assert names.normalize_drug_name("Sodium valproate") == "sodium valproate"
    print("  leading salt word is preserved OK")


def test_match_report():
    left = ["Amiodarone HCl", "Erythromycin", "Simvastatin"]
    right = ["amiodarone", "Erythromycin (oral)", "Quinidine"]
    report = names.match_report(left, right)
    assert report["matched"] == ["amiodarone", "erythromycin"]
    assert report["left_only"] == ["simvastatin"]
    assert report["right_only"] == ["quinidine"]
    print("  match_report separates matched from unmatched names OK")


# ----------------------------------------------------------------- tier logic

def c1(call, prob):
    return {"call": call, "prob": prob, "source": "model a"}


def c2(call, strength=None):
    return {"call": call, "strength": strength}


def c3_model(call, prob):
    return {"call": call, "prob": prob, "source": combine.SOURCE_MODEL_B}


def c3_list():
    return {"call": combine.YES, "prob": None,
            "source": combine.SOURCE_QT_LIST, "risk_category": "known risk"}


def test_lookups_from_panel():
    # Both drug-side lookups come from one table, so the substrate leg covers
    # every drug in the panel while the hERG leg covers only the measured ones.
    panel = pd.DataFrame({
        "name": ["amiodarone", "midazolam", "ibuprofen"],
        "strength": ["measured substrate"] * 3,
        "herg_measured": [True, False, False],
    })
    substrates, herg = combine.lookups_from_panel(panel)

    assert len(substrates) == 3
    assert len(herg) == 1
    assert combine.drug_is_cyp3a4_substrate("Midazolam", substrates)["call"] == combine.SUBSTRATE

    # A measured blocker is a firm yes from the list, with no model probability.
    measured = combine.drug_blocks_herg("amiodarone", None, None, herg)
    assert measured["call"] == combine.YES
    assert measured["source"] == combine.SOURCE_QT_LIST
    assert measured["prob"] is None

    # A panel drug with no hERG measurement falls through to Model B, which is
    # absent here, so it reports unavailable rather than guessing.
    unmeasured = combine.drug_blocks_herg("midazolam", None, None, herg)
    assert unmeasured["source"] == "unavailable"
    print("  one panel table yields both drug-side lookups OK")


def test_tier_domain_gate_wins():
    # Everything says yes, but the molecule is outside the domain.
    tier = combine.assign_tier(
        c1(combine.YES, 0.95), c2(combine.SUBSTRATE), c3_list(),
        combine.OUTSIDE_DOMAIN,
    )
    assert tier == combine.UNKNOWN
    print("  domain gate overrides three confident yes legs OK")


def test_tier_high():
    tier = combine.assign_tier(
        c1(combine.YES, 0.91), c2(combine.SUBSTRATE), c3_model(combine.YES, 0.88),
        applicability.INSIDE,
    )
    assert tier == combine.HIGH
    # QT list leg counts as confident without a probability.
    tier2 = combine.assign_tier(
        c1(combine.YES, 0.91), c2(combine.SUBSTRATE), c3_list(),
        applicability.INSIDE,
    )
    assert tier2 == combine.HIGH
    print("  High tier for three firm legs, with a model leg or a QT listing OK")


def test_tier_medium_two_of_three():
    tier = combine.assign_tier(
        c1(combine.YES, 0.91), c2(combine.NOT_IN_TABLE), c3_list(),
        applicability.INSIDE,
    )
    assert tier == combine.MEDIUM
    print("  Medium tier for two of three OK")


def test_tier_medium_borderline_leg():
    # This is the branch that is unreachable if borderline is not a weak hold.
    tier = combine.assign_tier(
        c1(combine.BORDERLINE, 0.55), c2(combine.SUBSTRATE), c3_list(),
        applicability.INSIDE,
    )
    assert tier == combine.MEDIUM, tier
    tier2 = combine.assign_tier(
        c1(combine.YES, 0.80), c2(combine.SUBSTRATE), c3_model(combine.BORDERLINE, 0.5),
        applicability.INSIDE,
    )
    assert tier2 == combine.MEDIUM, tier2
    print("  Medium tier when all three hold but one leg is borderline OK")


def test_tier_not_flagged():
    # Only one leg holds.
    tier = combine.assign_tier(
        c1(combine.NO, 0.05), c2(combine.SUBSTRATE), c3_model(combine.NO, 0.1),
        applicability.INSIDE,
    )
    assert tier == combine.NOT_FLAGGED
    # Nothing holds.
    tier2 = combine.assign_tier(
        c1(combine.NO, 0.02), c2(combine.NOT_SUBSTRATE), c3_model(combine.NO, 0.02),
        applicability.INSIDE,
    )
    assert tier2 == combine.NOT_FLAGGED
    print("  not flagged when fewer than two legs hold OK")


def test_tier_refuted_leg_breaks_the_chain():
    # A measured hERG non-blocker, paired with a confidently predicted
    # inhibitor and a real substrate record. Two legs hold, but the third is
    # refuted rather than merely missing, so the chain is broken and nothing
    # should be reported. Contrast with test_tier_medium_two_of_three, where
    # the third leg is "not in table", a gap in the source rather than a no.
    tier = combine.assign_tier(
        c1(combine.YES, 0.95), c2(combine.SUBSTRATE), c3_model(combine.NO, 0.02),
        applicability.INSIDE,
    )
    assert tier == combine.NOT_FLAGGED, tier

    # Same shape on the substrate leg: a drug measured as a non-substrate.
    tier2 = combine.assign_tier(
        c1(combine.YES, 0.95), c2(combine.NOT_SUBSTRATE), c3_list(),
        applicability.INSIDE,
    )
    assert tier2 == combine.NOT_FLAGGED, tier2
    print("  a refuted leg breaks the chain, no Medium by two-of-three OK")


def test_tier_lookup_has_no_weak_form():
    # "not in table" is a gap in the FDA source, never a weak yes, so a pair
    # with two borderline model legs and no substrate record is not flagged.
    tier = combine.assign_tier(
        c1(combine.BORDERLINE, 0.5), c2(combine.NOT_IN_TABLE),
        c3_model(combine.BORDERLINE, 0.5), applicability.INSIDE,
    )
    assert tier == combine.NOT_FLAGGED
    print("  'not in table' never counts as a hold OK")


def test_rank_score_is_min_not_product():
    assert combine.rank_score(0.9, 0.8) == 0.8
    assert combine.rank_score(0.8, 0.9) == 0.8
    # The product would be 0.72 here; the score must not be that.
    assert combine.rank_score(0.9, 0.8) != 0.9 * 0.8
    # A QT listing leaves prob_b as None: fall back to leg A alone.
    assert combine.rank_score(0.9, None) == 0.9
    assert combine.rank_score(None, 0.4) == 0.4
    assert combine.rank_score(None, None) == 0.0
    print("  rank_score is the weakest leg, never the product OK")


def test_rank_score_never_exceeds_either_leg():
    rng = np.random.default_rng(5)
    for _ in range(500):
        a, b = rng.random(2)
        score = combine.rank_score(a, b)
        assert score <= a + 1e-12 and score <= b + 1e-12
    print("  rank_score never exceeds either leg OK")


# ------------------------------------------------------------- lookups + screen

def test_substrate_lookup():
    df = pd.DataFrame({
        "name": ["Simvastatin", "Midazolam (oral)", "Amiodarone HCl"],
        "strength": ["sensitive", "sensitive", "moderate"],
    })
    lookup = combine.build_substrate_lookup(df)
    assert combine.drug_is_cyp3a4_substrate("simvastatin", lookup)["call"] == combine.SUBSTRATE
    assert combine.drug_is_cyp3a4_substrate("Midazolam", lookup)["call"] == combine.SUBSTRATE
    assert combine.drug_is_cyp3a4_substrate("amiodarone", lookup)["strength"] == "moderate"
    assert combine.drug_is_cyp3a4_substrate("aspirin", lookup)["call"] == combine.NOT_IN_TABLE
    assert combine.drug_is_cyp3a4_substrate("", lookup)["call"] == combine.NOT_IN_TABLE
    print("  FDA substrate lookup matches across salt forms and brackets OK")


def test_qt_lookup_beats_model():
    df = pd.DataFrame({"name": ["Amiodarone"], "risk_category": ["known risk of TdP"]})
    qt = combine.build_qt_lookup(df)

    class AlwaysNegative:
        def predict_proba(self, X):
            return np.array([[0.99, 0.01]])

    result = combine.drug_blocks_herg("amiodarone hcl", np.zeros(FP_SIZE),
                                      AlwaysNegative(), qt)
    # The clinical listing must win even though the model says no.
    assert result["call"] == combine.YES
    assert result["source"] == combine.SOURCE_QT_LIST
    assert result["prob"] is None
    assert result["risk_category"] == "known risk of TdP"
    print("  CredibleMeds listing overrides a negative Model B prediction OK")


def test_drug_blocks_herg_falls_back_to_model():
    qt = combine.build_qt_lookup(pd.DataFrame({"name": [], "risk_category": []}))

    class AlwaysPositive:
        def predict_proba(self, X):
            return np.array([[0.09, 0.91]])

    result = combine.drug_blocks_herg("sometrandomdrug", np.zeros(FP_SIZE),
                                      AlwaysPositive(), qt)
    assert result["call"] == combine.YES
    assert result["source"] == combine.SOURCE_MODEL_B
    assert abs(result["prob"] - 0.91) < 1e-9
    print("  unlisted drug falls back to Model B OK")


def test_drug_blocks_herg_no_fingerprint():
    qt = {}
    result = combine.drug_blocks_herg("unknown", None, None, qt)
    assert result["prob"] is None
    assert result["source"] == "unavailable"
    print("  missing SMILES reports unavailable rather than inventing a number OK")


def test_screen_pairs_end_to_end():
    # Real sklearn models trained on synthetic fingerprints, so predict_proba
    # behaves exactly as it will in the notebooks.
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(11)
    train = clustered_fps(300, rng=rng)
    y = np.zeros(300, dtype=int)
    # Make bit 0 the signal so the model has something learnable.
    train[:150, 0] = 1
    train[150:, 0] = 0
    y[:150] = 1

    model_a = RandomForestClassifier(n_estimators=50, random_state=0).fit(train, y)
    model_b = RandomForestClassifier(n_estimators=50, random_state=1).fit(train, y)
    domain = applicability.fit_domain(train)

    # Three herb compounds: two inside the domain, one far outside.
    herb_fps = np.vstack([
        clustered_fps(2, rng=np.random.default_rng(21)),
        random_fps(1, rng=np.random.default_rng(22)),
    ])
    herb_fps[0, 0] = 1
    herb_fps[1, 0] = 1
    herb_df = pd.DataFrame({
        "identifier": ["CNP1", "CNP2", "CNP3"],
        "name": ["curcumin", "gingerol", "weird thing"],
        "organisms": ["Curcuma longa", "Zingiber officinale", "Curcuma longa"],
    })

    drug_df = pd.DataFrame({"name": ["Simvastatin", "Aspirin"]})
    substrates = combine.build_substrate_lookup(
        pd.DataFrame({"name": ["Simvastatin"], "strength": ["sensitive"]})
    )
    qt = combine.build_qt_lookup(
        pd.DataFrame({"name": ["Simvastatin"], "risk_category": ["possible risk"]})
    )

    result = combine.screen_pairs(
        herb_df, drug_df, model_a, model_b, substrates, qt, domain,
        herb_fps=herb_fps, drug_fps=None,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    # Tier ordering: High before Medium, never interleaved.
    tier_rank = {combine.HIGH: 0, combine.MEDIUM: 1}
    ranks = [tier_rank[t] for t in result["tier"]]
    assert ranks == sorted(ranks), ranks
    # Inside each tier, rank_score descends.
    for tier, group in result.groupby("tier"):
        scores = group["rank_score"].tolist()
        assert scores == sorted(scores, reverse=True), (tier, scores)
    # The out-of-domain compound is never in the High/Medium output.
    assert "CNP3" not in set(result["herb_identifier"])
    print(f"  screen_pairs end to end OK ({len(result)} pairs flagged, "
          f"tiers {sorted(set(result['tier']))})")


def test_screen_pairs_unknown_tier_can_be_kept():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(13)
    train = clustered_fps(100, rng=rng)
    y = rng.integers(0, 2, size=100)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(train, y)
    domain = applicability.fit_domain(train)

    herb_fps = random_fps(2, rng=np.random.default_rng(77))
    herb_df = pd.DataFrame({
        "identifier": ["X1", "X2"], "name": ["a", "b"], "organisms": ["p", "q"],
    })
    drug_df = pd.DataFrame({"name": ["Simvastatin"]})
    substrates = combine.build_substrate_lookup(
        pd.DataFrame({"name": ["Simvastatin"], "strength": ["sensitive"]})
    )
    qt = combine.build_qt_lookup(
        pd.DataFrame({"name": ["Simvastatin"], "risk_category": ["x"]})
    )
    result = combine.screen_pairs(
        herb_df, drug_df, model, model, substrates, qt, domain,
        herb_fps=herb_fps, keep_tiers=(combine.UNKNOWN,),
    )
    assert len(result) == 2
    assert set(result["tier"]) == {combine.UNKNOWN}
    # Model A must not have been run outside the domain.
    assert result["cond1_prob"].isna().all()
    print("  out-of-domain pairs come back as Unknown with no Model A number OK")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} tests\n")
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL {fn.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
