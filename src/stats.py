# "Is this fragment really linked to inhibition, or did it just come up by luck?"
#
# SHAP tells us which fragments the model leaned on. That is a fact about the
# model, not about chemistry. The model could be leaning on something that
# happens to correlate in our particular dataset by accident.
#
# So we check separately, straight from the labels and ignoring the model: among
# molecules that HAVE this fragment, are inhibitors more common than among
# molecules that do not? That is what the tests in here do.

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def motif_enrichment(motif_present, labels):
    # Test one fragment against the true labels.
    #
    # Builds a 2x2 count table and runs Fisher's exact test, which asks: if the
    # fragment had nothing to do with inhibition, how unlikely is a split this
    # lopsided? A small p-value means "unlikely to be luck".
    #
    # Fisher rather than chi-squared because rare fragments give tiny counts,
    # and chi-squared stops being trustworthy when the counts get small.
    #
    # The table is laid out this way on purpose, because it decides which
    # direction the odds ratio points:
    #     [[has fragment & inhibitor, has fragment & not inhibitor],
    #      [no  fragment & inhibitor, no  fragment & not inhibitor]]
    # With that layout, odds ratio above 1 means the fragment is commoner among
    # inhibitors, and below 1 means commoner among non-inhibitors.
    present = np.asarray(motif_present).astype(bool)
    y = np.asarray(labels).astype(bool)
    if present.shape[0] != y.shape[0]:
        raise ValueError("motif column and labels have different lengths")

    present_pos = int(np.count_nonzero(present & y))
    present_neg = int(np.count_nonzero(present & ~y))
    absent_pos = int(np.count_nonzero(~present & y))
    absent_neg = int(np.count_nonzero(~present & ~y))

    table = [[present_pos, present_neg], [absent_pos, absent_neg]]
    odds_ratio, p_value = fisher_exact(table)
    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "n_present": present_pos + present_neg,
        "present_pos": present_pos,
        "present_neg": present_neg,
        "absent_pos": absent_pos,
        "absent_neg": absent_neg,
    }


def benjamini_hochberg(pvalues, alpha=0.05):
    # Fix the problem of running thousands of tests at once.
    #
    # A p-value of 0.05 means "this would happen by luck 1 time in 20". Test
    # 2,000 fragments and about 100 will clear that bar even if not one of them
    # matters. This raises the bar depending on how many tests were run.
    #
    # Returns (adjusted p-values in the original order, True/False significant).
    p = np.asarray(pvalues, dtype=float)
    n = p.shape[0]
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, n + 1, dtype=float)

    # Scale each p-value by how many tests ran and where it ranks, then sweep
    # backwards keeping the running minimum. Without that sweep the adjusted
    # values can go out of order, so a fragment could end up with a smaller
    # adjusted p than another fragment that had a smaller raw p. That would be
    # nonsense, and it is the step people most often leave out.
    scaled = p_sorted * n / ranks
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= alpha


def enrichment_table(X, labels, bit_ids, alpha=0.05):
    # Test a whole list of fragments and put the results in one table, sorted
    # with the most convincing first.
    X = np.asarray(X)
    y = np.asarray(labels)
    bit_ids = list(bit_ids)

    rows = []
    for bit in bit_ids:
        result = motif_enrichment(X[:, bit], y)
        result["bit"] = int(bit)
        rows.append(result)

    if len(rows) == 0:
        return pd.DataFrame(
            columns=["bit", "odds_ratio", "p_value", "p_adjusted", "significant"]
        )

    df = pd.DataFrame(rows)
    adjusted, significant = benjamini_hochberg(df["p_value"].values, alpha=alpha)
    df["p_adjusted"] = adjusted
    df["significant"] = significant

    # Record how many tests the correction covered. Correcting across 20
    # fragments and across 2,000 gives different answers for the same raw
    # p-value, so a reader has to know which one this was. Quoting an adjusted
    # p without saying how many tests it covers means nothing.
    df["n_tests"] = len(rows)

    columns = [
        "bit",
        "odds_ratio",
        "p_value",
        "p_adjusted",
        "significant",
        "n_tests",
        "n_present",
        "present_pos",
        "present_neg",
        "absent_pos",
        "absent_neg",
    ]
    df = df[columns].sort_values(["p_adjusted", "p_value"]).reset_index(drop=True)
    return df
