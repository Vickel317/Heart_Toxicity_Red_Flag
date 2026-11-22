# The three conditions, and the tier logic that turns them into one verdict.
# This is the core claim of the project, so the rules live in one place.

import numpy as np
import pandas as pd

from names import normalize_drug_name

# Confidence bands for the model legs. A probability in the middle is treated as
# low confidence rather than as a yes or a no.
HIGH_CONF = 0.70
LOW_CONF = 0.30

# Condition 1 / 3 model calls
YES = "yes"
NO = "no"
BORDERLINE = "borderline"

# Condition 2 lookup results
SUBSTRATE = "substrate"
NOT_SUBSTRATE = "not substrate"
NOT_IN_TABLE = "not in table"

# Tiers
HIGH = "High"
MEDIUM = "Medium"
UNKNOWN = "Unknown"
NOT_FLAGGED = None

# Where condition 3 came from. A clinical listing outranks a model output.
SOURCE_QT_LIST = "qt list"
SOURCE_MODEL_B = "model b"

OUTSIDE_DOMAIN = "outside domain"


def _band(prob):
    # Turn a probability into yes / no / borderline using the two cut-offs.
    if prob >= HIGH_CONF:
        return YES
    if prob <= LOW_CONF:
        return NO
    return BORDERLINE


def _positive_prob(model, fp):
    # Positive class probability for a single fingerprint row.
    row = np.asarray(fp).reshape(1, -1)
    return float(model.predict_proba(row)[0, 1])


def herb_inhibits_cyp3a4(herb_fp, model_a):
    # Condition 1. Does the herb compound inhibit CYP3A4? Predicted by Model A.
    prob = _positive_prob(model_a, herb_fp)
    return {"call": _band(prob), "prob": prob, "source": "model a"}


def drug_is_cyp3a4_substrate(drug_name, substrate_table):
    # Condition 2. Is the drug cleared by CYP3A4?
    # This is a LOOKUP in the FDA substrate table, never a prediction. If the
    # drug is not in the table we say so, we do not guess.
    #
    # substrate_table is a dict of normalised name -> record, or a DataFrame
    # with a "name" column. Build it with build_substrate_lookup below.
    key = normalize_drug_name(drug_name)
    if not key:
        return {"call": NOT_IN_TABLE, "strength": None}

    record = substrate_table.get(key)
    if record is None:
        return {"call": NOT_IN_TABLE, "strength": None}

    # The FDA splits sensitive substrates from moderate ones. That distinction
    # is kept, because it decides how strong a claim the pair can support.
    return {
        "call": SUBSTRATE,
        "strength": record.get("strength"),
        "matched_name": record.get("name"),
    }


def drug_blocks_herg(drug_name, drug_fp, model_b, qt_list):
    # Condition 3. Does the drug block hERG or prolong QT?
    #
    # The CredibleMeds QT list is checked FIRST. A clinical listing beats a
    # prediction, so a listed drug never gets a model probability attached,
    # and the risk category from the list is carried through rather than
    # flattened into a yes.
    key = normalize_drug_name(drug_name)
    if key and qt_list is not None:
        record = qt_list.get(key)
        if record is not None:
            return {
                "call": YES,
                "prob": None,
                "source": SOURCE_QT_LIST,
                "risk_category": record.get("risk_category"),
                "matched_name": record.get("name"),
            }

    # Not listed, so fall back to Model B. No fingerprint means no answer;
    # reporting borderline here would claim a measurement that was never made.
    if drug_fp is None or model_b is None:
        return {
            "call": BORDERLINE,
            "prob": None,
            "source": "unavailable",
            "risk_category": None,
        }

    prob = _positive_prob(model_b, drug_fp)
    return {
        "call": _band(prob),
        "prob": prob,
        "source": SOURCE_MODEL_B,
        "risk_category": None,
    }


def assign_tier(cond1, cond2, cond3, domain_status):
    # Combine the three conditions into High, Medium, Unknown, or not flagged.
    #
    # Rules, checked in this order:
    #   1. outside the applicability domain -> Unknown. The gate wins over
    #      everything, because out there Model A has no basis for its answer
    #      and the other two legs cannot rescue it.
    #   2. all three hold firmly AND both model legs confident -> High
    #   3. two of three hold firmly, or all three hold with one of them only
    #      weakly (a borderline probability) -> Medium
    #   4. anything else -> not flagged
    #
    # On "firmly" versus "weakly": a model leg is a firm yes at p >= HIGH_CONF,
    # and borderline between LOW_CONF and HIGH_CONF. A borderline leg is
    # treated as a WEAK HOLD, not as a failure, which is what makes the second
    # half of rule 3 reachable at all. Without that distinction the yes band
    # and the confidence band use the same cut-off, so "all three yes but a leg
    # is borderline" could never happen and every such pair would silently drop
    # to the two-of-three case.
    #
    # The lookup leg has no weak form. The FDA table either lists a drug as a
    # CYP3A4 substrate or it does not, and "not in table" is a gap in the
    # source rather than a weak yes, so it never counts as a hold.
    if domain_status == OUTSIDE_DOMAIN:
        return UNKNOWN

    c1 = cond1.get("call")
    c2 = cond2.get("call")
    c3 = cond3.get("call")

    firm = [c1 == YES, c2 == SUBSTRATE, c3 == YES]
    weak = [
        c1 in (YES, BORDERLINE),
        c2 == SUBSTRATE,
        c3 in (YES, BORDERLINE),
    ]
    firm_count = sum(firm)

    if firm_count == 3:
        # Leg 1 is always a model. Leg 3 is only a model when it did not come
        # from the QT list; a clinical listing counts as confident, because it
        # is a stronger statement than any probability this project produces.
        prob1 = cond1.get("prob")
        leg1_confident = prob1 is not None and prob1 >= HIGH_CONF
        if cond3.get("source") == SOURCE_QT_LIST:
            leg3_confident = True
        else:
            prob3 = cond3.get("prob")
            leg3_confident = prob3 is not None and prob3 >= HIGH_CONF

        if leg1_confident and leg3_confident:
            return HIGH
        return MEDIUM

    if firm_count == 2:
        return MEDIUM

    # All three hold, but at least one only weakly.
    if all(weak):
        return MEDIUM

    return NOT_FLAGGED


def rank_score(prob_a, prob_b):
    # A single number used ONLY to order pairs that already share a tier.
    #
    # The minimum of the two legs, so a pair is only ranked as high as its
    # weakest leg. prob_b is None when condition 3 came from the QT list; a
    # clinical listing is treated as the strongest possible leg, so the score
    # falls back to the Model A probability alone.
    #
    # NEVER multiply prob_a by prob_b. Random Forest probabilities are not
    # calibrated, they are just the fraction of trees that voted yes, so the
    # product is not a joint probability of anything. It would look like a real
    # risk number while meaning nothing, which is worse than no number.
    if prob_a is None and prob_b is None:
        return 0.0
    if prob_b is None:
        return float(prob_a)
    if prob_a is None:
        return float(prob_b)
    return float(min(prob_a, prob_b))


def build_substrate_lookup(df, name_col="name", strength_col="strength"):
    # DataFrame -> dict of normalised name -> record, for drug_is_cyp3a4_substrate.
    lookup = {}
    for _, row in df.iterrows():
        key = normalize_drug_name(row[name_col])
        if not key:
            continue
        lookup[key] = {
            "name": row[name_col],
            "strength": row[strength_col] if strength_col in df.columns else None,
        }
    return lookup


def build_qt_lookup(df, name_col="name", category_col="risk_category"):
    # DataFrame -> dict of normalised name -> record, for drug_blocks_herg.
    lookup = {}
    for _, row in df.iterrows():
        key = normalize_drug_name(row[name_col])
        if not key:
            continue
        lookup[key] = {
            "name": row[name_col],
            "risk_category": row[category_col] if category_col in df.columns else None,
        }
    return lookup


def screen_pairs(
    herb_df,
    drug_df,
    model_a,
    model_b,
    substrate_table,
    qt_list,
    domain,
    herb_fps=None,
    drug_fps=None,
    keep_tiers=(HIGH, MEDIUM),
):
    # Score every (herb compound, drug) pair and return one row per flagged pair.
    #
    # herb_df needs an "identifier" and a "name" column; herb_fps is the
    # matching fingerprint matrix, row i going with herb_df row i.
    # drug_df needs a "name" column; drug_fps may be None for drugs with no SMILES.
    #
    # Condition 1 and the domain check depend only on the herb compound, so
    # they are computed once per compound rather than once per pair. With a few
    # thousand compounds and a few dozen drugs that is the difference between
    # minutes and hours.
    import applicability

    herb_df = herb_df.reset_index(drop=True)
    drug_df = drug_df.reset_index(drop=True)

    if herb_fps is None:
        raise ValueError("screen_pairs needs herb_fps, the herb fingerprint matrix")

    domain_verdicts, domain_scores = applicability.check_molecules(herb_fps, domain)

    # Condition 1 once per herb compound.
    cond1_cache = []
    for i in range(len(herb_df)):
        if domain_verdicts[i] == OUTSIDE_DOMAIN:
            # Outside the domain the tier is Unknown regardless, so Model A is
            # not run at all. Running it would produce a number that the tier
            # logic then throws away, and that number tends to get quoted.
            cond1_cache.append({"call": None, "prob": None, "source": "not run"})
        else:
            cond1_cache.append(herb_inhibits_cyp3a4(herb_fps[i], model_a))

    # Condition 2 and 3 once per drug.
    cond2_cache = []
    cond3_cache = []
    for j in range(len(drug_df)):
        drug_name = drug_df.loc[j, "name"]
        cond2_cache.append(drug_is_cyp3a4_substrate(drug_name, substrate_table))
        fp = None if drug_fps is None else drug_fps[j]
        cond3_cache.append(drug_blocks_herg(drug_name, fp, model_b, qt_list))

    rows = []
    for i in range(len(herb_df)):
        for j in range(len(drug_df)):
            cond1 = cond1_cache[i]
            cond2 = cond2_cache[j]
            cond3 = cond3_cache[j]
            tier = assign_tier(cond1, cond2, cond3, domain_verdicts[i])
            if tier not in keep_tiers:
                continue
            rows.append(
                {
                    "herb_identifier": herb_df.loc[i, "identifier"],
                    "herb_name": herb_df.loc[i, "name"],
                    "herb_organisms": herb_df.loc[i, "organisms"]
                    if "organisms" in herb_df.columns
                    else None,
                    "drug": drug_df.loc[j, "name"],
                    "tier": tier,
                    "cond1_call": cond1.get("call"),
                    "cond1_prob": cond1.get("prob"),
                    "cond2_call": cond2.get("call"),
                    "cond2_strength": cond2.get("strength"),
                    "cond3_call": cond3.get("call"),
                    "cond3_prob": cond3.get("prob"),
                    "cond3_source": cond3.get("source"),
                    "cond3_risk_category": cond3.get("risk_category"),
                    "domain_status": domain_verdicts[i],
                    "domain_score": float(domain_scores[i]),
                    "rank_score": rank_score(cond1.get("prob"), cond3.get("prob")),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "herb_identifier", "herb_name", "drug", "tier",
                "cond1_call", "cond1_prob", "cond2_call", "cond3_call",
                "cond3_source", "domain_status", "rank_score",
            ]
        )

    out = pd.DataFrame(rows)
    # Sort by tier first, then by score INSIDE the tier. The score never moves
    # a pair between tiers, so tier is the primary key and always wins.
    tier_order = {HIGH: 0, MEDIUM: 1, UNKNOWN: 2}
    out["_tier_rank"] = out["tier"].map(tier_order)
    out = out.sort_values(
        ["_tier_rank", "rank_score"], ascending=[True, False]
    ).drop(columns="_tier_rank")
    return out.reset_index(drop=True)
