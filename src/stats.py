# Is a substructure genuinely enriched among inhibitors, or did it turn up by chance?
# SHAP says which bits the model leaned on. It does not say whether that
# association is statistically real, which is what these tests are for.

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def motif_enrichment(motif_present, labels):
    # One motif against one label column.
    # Builds the 2x2 table and runs Fisher's exact test. Fisher rather than
    # chi-squared because rare motifs give small cell counts, where the
    # chi-squared approximation is not trustworthy.
    #
    # Table orientation, which fixes what the odds ratio means:
    #     [[motif present & label 1, motif present & label 0],
    #      [motif absent  & label 1, motif absent  & label 0]]
    # So an odds ratio above 1 means the motif is enriched among the positives.
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
    # Correct for testing many motifs at once.
    # One test per fingerprint bit over 1024 bits would hand us roughly 50
    # false hits at alpha 0.05 even if no motif mattered at all.
    #
    # Returns (adjusted p values in the original order, boolean significant mask).
    p = np.asarray(pvalues, dtype=float)
    n = p.shape[0]
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)

    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, n + 1, dtype=float)

    # Raw BH scaling, then the step-up running minimum from the largest p
    # downwards. Without that running minimum the "adjusted" values are not
    # monotone, and a bit could come out with a smaller adjusted p than a bit
    # with a smaller raw p.
    scaled = p_sorted * n / ranks
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= alpha


def enrichment_table(X, labels, bit_ids, alpha=0.05):
    # Run the test for a list of bits and collect the results in one dataframe,
    # sorted by adjusted p value.
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

    # The correction was applied across exactly the bits tested here, so record
    # that count. Testing 20 bits and testing 1024 give different adjusted
    # values for the same raw p, and a reader needs to know which this was.
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
