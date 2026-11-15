# The three conditions, and the tier logic that turns them into one verdict.
# This is the core claim of the project, so the rules live in one place.

# STUB. Nothing here is implemented yet.

# Confidence bands for the model legs. A probability in the middle is treated as
# low confidence rather than as a yes or a no.
HIGH_CONF = 0.70
LOW_CONF = 0.30


def herb_inhibits_cyp3a4(herb_fp, model_a):
    # Condition 1. Does the herb compound inhibit CYP3A4?
    # TODO:
    #   1. model_a.predict_proba on the single fingerprint row.
    #   2. Take the positive class probability.
    #   3. Return the probability and a "yes" / "no" / "borderline" call using
    #      HIGH_CONF and LOW_CONF.
    pass


def drug_is_cyp3a4_substrate(drug_name, substrate_table):
    # Condition 2. Is the drug cleared by CYP3A4?
    # This is a LOOKUP in the FDA substrate table, never a prediction. If the
    # drug is not in the table we say so, we do not guess.
    # TODO:
    #   1. Normalise drug_name, lowercase and strip, because the source tables
    #      disagree about spelling and salt forms.
    #   2. Look it up in substrate_table.
    #   3. Return "substrate", "not substrate", or "not in table".
    pass


def drug_blocks_herg(drug_name, drug_fp, model_b, qt_list):
    # Condition 3. Does the drug block hERG or prolong QT?
    # TODO:
    #   1. Check the CredibleMeds QT list first. A clinical listing beats a
    #      prediction, so if the drug is on it, return that and mark the source
    #      as "qt list".
    #   2. Otherwise fall back to model_b.predict_proba on drug_fp and mark the
    #      source as "model b".
    #   3. Return the call, the probability if a model was used, and the source.
    pass


def assign_tier(cond1, cond2, cond3, domain_status):
    # Combine the three conditions into High, Medium, or Unknown.
    # TODO the rules, checked in this order:
    #   1. domain_status == "outside domain" -> "Unknown". The applicability gate
    #      wins over everything, because outside the domain Model A has no basis
    #      for its answer and the other two legs cannot rescue it.
    #   2. All three conditions yes AND both model legs above HIGH_CONF -> "High".
    #   3. Two of three yes, OR all three yes but some leg is borderline -> "Medium".
    #   4. Anything else -> not flagged.
    pass


def rank_score(prob_a, prob_b):
    # A single number used ONLY to order pairs that already share a tier.
    # TODO: something simple and monotone, for example the minimum of the two
    # probabilities, so a pair is only ranked as high as its weakest leg.
    #
    # NEVER multiply prob_a by prob_b. Random Forest probabilities are not
    # calibrated, they are just the fraction of trees that voted yes, so the
    # product is not a joint probability of anything. It would look like a real
    # risk number while meaning nothing, which is worse than no number.
    # The score also must never move a pair between tiers.
    pass


def screen_pairs(herb_df, drug_df, model_a, model_b, substrate_table, qt_list, domain):
    # Score every (herb compound, drug) pair and return one row per flagged pair.
    # TODO: plain nested for loop over herb compounds and drugs, call the three
    # condition functions, call assign_tier, keep the High and Medium rows, then
    # sort by tier and by rank_score inside each tier.
    pass
