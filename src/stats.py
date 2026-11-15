# Is a substructure genuinely enriched among inhibitors, or did it turn up by chance?
# SHAP says which bits the model leaned on. It does not say whether that
# association is statistically real, which is what these tests are for.

# STUB. Nothing here is implemented yet.


def motif_enrichment(motif_present, labels):
    # One motif against one label column.
    # TODO:
    #   1. Build the 2x2 count table: motif present or absent, against label 1 or 0.
    #   2. odds_ratio, p_value = scipy.stats.fisher_exact(table)
    #      Fisher rather than chi-squared because rare motifs give small cell counts.
    #   3. Return the odds ratio and the raw p value.
    pass


def benjamini_hochberg(pvalues, alpha=0.05):
    # Correct for testing hundreds of motifs at once.
    # TODO:
    #   1. Sort the p values ascending, remembering the original positions.
    #   2. For rank i out of n, compare p to (i / n) * alpha.
    #   3. The largest rank that passes sets the cutoff. Everything at or below
    #      that rank counts as significant.
    #   4. Return adjusted p values back in the original order.
    # Needed because one test per fingerprint bit over 1024 bits would hand us
    # roughly 50 false hits at alpha 0.05 even if no motif mattered at all.
    pass


def enrichment_table(X, labels, bit_ids):
    # Run the test for a list of bits and collect the results in one dataframe.
    # TODO: loop over bit_ids, call motif_enrichment on the column, collect
    # odds ratios and p values, then apply benjamini_hochberg to the p value
    # column and sort by adjusted p value.
    pass
