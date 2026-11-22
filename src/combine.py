# The whole idea of the project lives in this file.
#
# THE STORY. Someone takes jamu and a prescription drug on the same day. For
# that to actually be dangerous, three separate things all have to be true:
#
#   1. a compound in the jamu jams CYP3A4
#      (CYP3A4 = the liver enzyme that clears about half of all drugs)
#   2. the drug they are taking is one that CYP3A4 clears
#      (if the drug leaves the body another way, a jammed enzyme changes nothing)
#   3. that drug causes heart trouble when it builds up
#      (it blocks hERG, a channel the heart uses to reset between beats, which
#       stretches the gap called the QT interval and can trigger an arrhythmia)
#
# Break any one link and there is no risk. That is why this is a CHAIN, and why
# a pair only gets flagged when the links hold together.
#
# Only condition 1 is a prediction. Conditions 2 and 3 are looked up in lab
# data wherever that data exists, because a real measurement beats a model.

import numpy as np
import pandas as pd

from names import normalize_drug_name

# Where to draw the line on a model's probability. Anything in the middle is
# treated as "not sure", which is different from a no.
HIGH_CONF = 0.70
LOW_CONF = 0.30

# What a model leg can answer (conditions 1 and 3).
YES = "yes"
NO = "no"
BORDERLINE = "borderline"

# What a lookup leg can answer (condition 2). Note that "not in table" and
# "not substrate" are NOT the same thing: one means we have no information,
# the other means we checked and the answer is no.
SUBSTRATE = "substrate"
NOT_SUBSTRATE = "not substrate"
NOT_IN_TABLE = "not in table"

# The final verdict a pair can get.
HIGH = "High"
MEDIUM = "Medium"
UNKNOWN = "Unknown"
NOT_FLAGGED = None

# Where condition 3's answer came from. A lab measurement outranks a model.
SOURCE_QT_LIST = "qt list"
SOURCE_MODEL_B = "model b"

# The label we attach to a measured condition 3. Originally this was meant to
# be the CredibleMeds clinical risk category, but that list cannot be
# redistributed, so we use TDC's lab hERG results instead. Those are two
# different kinds of evidence (a lab assay vs what happened to patients), so
# the output records which one it is rather than hiding the swap.
HERG_MEASURED = "measured hERG block (TDC)"

OUTSIDE_DOMAIN = "outside domain"


def _band(prob):
    # Probability in, yes / no / not-sure out.
    if prob >= HIGH_CONF:
        return YES
    if prob <= LOW_CONF:
        return NO
    return BORDERLINE


def _positive_prob(model, fp):
    # Ask a model for one molecule's probability of being a yes.
    row = np.asarray(fp).reshape(1, -1)
    return float(model.predict_proba(row)[0, 1])


def herb_inhibits_cyp3a4(herb_fp, model_a):
    # CONDITION 1: does this jamu compound jam CYP3A4?
    # This is the one leg that is genuinely a prediction, from Model A.
    prob = _positive_prob(model_a, herb_fp)
    return {"call": _band(prob), "prob": prob, "source": "model a"}


def drug_is_cyp3a4_substrate(drug_name, substrate_table):
    # CONDITION 2: is this drug cleared by CYP3A4?
    #
    # Looked up in lab data, never predicted. If the drug is not in the table,
    # we say "not in table" and stop. We do not guess, because guessing here
    # would invent a link in the chain.
    #
    # substrate_table maps a cleaned drug name to its record. Build it with
    # build_substrate_lookup, or get both lookups from lookups_from_panel.
    key = normalize_drug_name(drug_name)
    if not key:
        return {"call": NOT_IN_TABLE, "strength": None}

    record = substrate_table.get(key)
    if record is None:
        return {"call": NOT_IN_TABLE, "strength": None}

    # "strength" says how strongly the source backs this up. It is carried
    # through rather than thrown away, because it limits how firmly a flagged
    # pair can be stated later.
    return {
        "call": SUBSTRATE,
        "strength": record.get("strength"),
        "matched_name": record.get("name"),
    }


def drug_blocks_herg(drug_name, drug_fp, model_b, qt_list):
    # CONDITION 3: does this drug cause heart trouble when it builds up?
    #
    # The measured list is checked FIRST, and only if the drug is not on it do
    # we fall back to Model B. A lab result beats a prediction, so a listed
    # drug never gets a model probability attached to it at all.
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

    # Not on the measured list, so ask Model B. If we have no structure for the
    # drug there is nothing to ask about, so the answer is "unavailable" rather
    # than a guess.
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
    # Take the three answers and produce one verdict.
    #
    # The rules, checked in this order:
    #   1. the compound is outside what Model A knows  -> Unknown
    #   2. any link is actively broken                 -> not flagged
    #   3. all three links hold, and solidly           -> High
    #   4. two links hold, or all three but one shaky  -> Medium
    #   5. anything else                               -> not flagged
    #
    # Rule 1 wins over everything. If Model A has never seen chemistry like
    # this, condition 1 is not weak evidence, it is no evidence, and the other
    # two links cannot make up for a missing first link.
    #
    # Rule 2 is the chain idea taken seriously. "Two of three hold" in rule 4
    # is meant for a link we have no data on, NOT for one we checked and found
    # false. A drug measured as not blocking hERG has no route to the heart
    # problem, so the pair is not a risk no matter how strong the other two
    # links look.
    #
    # "Solidly" vs "shaky": a model leg is solid at 0.70 or above, shaky
    # between 0.30 and 0.70. A shaky leg still counts as holding, just weakly.
    # That distinction is what makes rule 4's second half possible at all; if
    # the yes line and the confidence line were the same number, "all three yes
    # but one is shaky" could never happen.
    #
    # The lookup leg has no shaky version. The table either lists the drug or
    # it does not, and "not in table" is missing information, not a weak yes,
    # so it never counts as a link holding.
    if domain_status == OUTSIDE_DOMAIN:
        return UNKNOWN

    c1 = cond1.get("call")
    c2 = cond2.get("call")
    c3 = cond3.get("call")

    # A refuted leg breaks the chain, so the pair is not flagged however strong
    # the other two are. This is NOT the same as a leg with no data: "not in
    # table" is a gap in the source, while "not a substrate" and a confident
    # model NO are statements that this link does not hold.
    #
    # Without this check the two-of-three rule below gave Medium to a drug that
    # is measured as a hERG non-blocker, which is the opposite of what the
    # chain means: no hERG block, no arrhythmia route, no risk to report.
    if c1 == NO or c2 == NOT_SUBSTRATE or c3 == NO:
        return NOT_FLAGGED

    firm = [c1 == YES, c2 == SUBSTRATE, c3 == YES]
    weak = [
        c1 in (YES, BORDERLINE),
        c2 == SUBSTRATE,
        c3 in (YES, BORDERLINE),
    ]
    firm_count = sum(firm)

    if firm_count == 3:
        # All three hold. High or Medium now depends on how sure we are.
        #
        # Link 1 is always Model A, so it needs a confident probability.
        # Link 3 is only a model when the drug was not on the measured list.
        # If it WAS on the list, that counts as confident automatically: a lab
        # result is a stronger statement than any probability we can produce.
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
    # One number, used ONLY to sort pairs that are already in the same tier.
    # It never moves a pair from Medium to High.
    #
    # It takes the SMALLER of the two probabilities, so a pair is only as
    # strong as its weakest link. prob_b is None when condition 3 came from the
    # measured list, which is the strongest possible link, so in that case the
    # score is just Model A's number.
    #
    # NEVER multiply the two together. It is tempting: "70% x 80% = 56% risk"
    # looks like a real answer. It is not. These probabilities are not
    # calibrated. A random forest saying 0.70 means "70% of my trees voted
    # yes", not "this happens 70% of the time". Multiplying two numbers that do
    # not mean what they look like produces something that looks like a risk
    # percentage and means nothing at all, which is worse than giving no number.
    if prob_a is None and prob_b is None:
        return 0.0
    if prob_b is None:
        return float(prob_a)
    if prob_a is None:
        return float(prob_b)
    return float(min(prob_a, prob_b))


def build_substrate_lookup(df, name_col="name", strength_col="strength"):
    # Turn a table of drugs into a fast name -> record lookup for condition 2.
    # Names are cleaned first (see names.py) so spelling differences do not
    # cause a drug to quietly go missing.
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
    # Same thing for condition 3: name -> record, for the drugs with a measured
    # hERG result.
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


def lookups_from_panel(panel):
    # Build BOTH drug-side lookups from the single drug panel table.
    #
    # These used to be two separate CSV files that had to be kept in step by
    # hand. They are really just two columns of one table:
    #   condition 2 -> "strength",      filled in for every drug in the panel
    #   condition 3 -> "herg_measured", true only where a lab actually measured it
    #
    # Building both from one source means they cannot disagree about which
    # drugs are in the screen. Returns (substrate lookup, hERG lookup).
    measured = panel[panel["herg_measured"].astype(bool)].copy()
    measured["risk_category"] = HERG_MEASURED
    return build_substrate_lookup(panel), build_qt_lookup(measured)


def screen_pairs(
    herb_df,
    drug_df,
    model_a,
    model_b,
    substrate_table,
    qt_list,
    domain,
    herb_fps=None,
    herb_bits=None,
    drug_fps=None,
    keep_tiers=(HIGH, MEDIUM),
):
    # Check every (jamu compound, drug) combination and return the flagged ones.
    #
    # What goes in:
    #   herb_df    one row per jamu compound, needs "identifier" and "name"
    #   herb_fps   the number rows for those compounds, same order as herb_df
    #   herb_bits  the SAME compounds as yes/no boxes (see the warning below)
    #   drug_df    one row per drug, needs "name"
    #   drug_fps   number rows for the drugs; an entry may be None
    #
    # WHY herb_fps AND herb_bits BOTH. They describe the same molecules but the
    # two users need different formats. The models want fragment counts plus
    # the whole-molecule numbers. The applicability domain wants plain yes/no
    # boxes, because Tanimoto similarity only works on those. Pass the wrong
    # one and it either crashes on the column count or, worse, silently returns
    # nonsense. Build it with features.bits_from_matrix(herb_fps).
    #
    # SPEED. Condition 1 and the domain check only depend on the jamu compound,
    # and conditions 2 and 3 only on the drug. So each is worked out ONCE per
    # compound or per drug and reused, instead of being redone for every
    # combination. With ~1,000 compounds and ~340 drugs that is the difference
    # between minutes and hours.
    import applicability

    herb_df = herb_df.reset_index(drop=True)
    drug_df = drug_df.reset_index(drop=True)

    if herb_fps is None:
        raise ValueError("screen_pairs needs herb_fps, the herb fingerprint matrix")

    if herb_bits is None:
        herb_bits = herb_fps

    domain_verdicts, domain_scores = applicability.check_molecules(herb_bits, domain)

    # Condition 1, once per jamu compound.
    cond1_cache = []
    for i in range(len(herb_df)):
        if domain_verdicts[i] == OUTSIDE_DOMAIN:
            # Outside the domain the answer is Unknown whatever Model A says,
            # so we do not even ask it. If we did, we would produce a number
            # that gets thrown away two lines later, and numbers like that have
            # a habit of being quoted anyway.
            cond1_cache.append({"call": None, "prob": None, "source": "not run"})
        else:
            cond1_cache.append(herb_inhibits_cyp3a4(herb_fps[i], model_a))

    # Conditions 2 and 3, once per drug.
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
    # Sort by tier first, then by score within the tier. The score decides the
    # order inside a tier and nothing else; it can never promote a Medium pair
    # into High.
    tier_order = {HIGH: 0, MEDIUM: 1, UNKNOWN: 2}
    out["_tier_rank"] = out["tier"].map(tier_order)
    out = out.sort_values(
        ["_tier_rank", "rank_score"], ascending=[True, False]
    ).drop(columns="_tier_rank")
    return out.reset_index(drop=True)
