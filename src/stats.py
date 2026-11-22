# Is a substructure genuinely enriched among inhibitors, or did it turn up by chance?
# SHAP says which bits the model leaned on. It does not say whether that
# association is statistically real, which is what these tests are for.

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def motif_enrichment(motif_present, labels):
    # One motif against one label column.
    #
    # Builds the 2x2 table and runs Fisher's exact test rather than chi-squared,
    # because rare motifs produce small cell counts where chi-squared is unreliable.
    #
    #                      active (y=1)   inactive (y=0)
    #   motif present            a              b
    #   motif absent             c              d
    motif_present = np.asarray(motif_present).astype(bool)
    labels = np.asarray(labels).astype(int)

    a = int(np.sum(motif_present & (labels == 1)))
    b = int(np.sum(motif_present & (labels == 0)))
    c = int(np.sum(~motif_present & (labels == 1)))
    d = int(np.sum(~motif_present & (labels == 0)))

    odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
    return odds_ratio, p_value


def benjamini_hochberg(pvalues, alpha=0.05):
    # Correct for testing hundreds of motifs at once.
    #
    # Testing one bit at a time at p < 0.05 over 2048 bits would hand us roughly
    # 100 "significant" hits even if no motif mattered at all. This returns
    # adjusted p values (q values), in the original order, so a reader can apply
    # whatever cutoff they like.
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])

    order = np.argsort(p)
    ranked = p[order]

    # p * n / rank, for rank 1 .. n
    adjusted = ranked * n / np.arange(1, n + 1)

    # Walking back from the largest keeps the adjusted values non-decreasing,
    # so a motif can never be reported as more significant than a stronger one.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    out = np.empty(n, dtype=float)
    out[order] = adjusted
    return out


def enrichment_table(X, labels, bit_ids, alpha=0.05):
    # Run the test for a list of bits and collect the results in one dataframe.
    # X may hold counts, so "present" means the fragment appears at least once.
    X = np.asarray(X)
    labels = np.asarray(labels).astype(int)

    rows = []
    for bit in bit_ids:
        present = X[:, bit] > 0
        odds_ratio, p_value = motif_enrichment(present, labels)
        n_present = int(present.sum())
        n_active = int(np.sum(present & (labels == 1)))
        rows.append({
            "bit": int(bit),
            "molecules_with_motif": n_present,
            "of_which_active": n_active,
            "active_rate": (n_active / n_present) if n_present else np.nan,
            "odds_ratio": odds_ratio,
            "p_value": p_value,
        })

    table = pd.DataFrame(rows)
    if len(table) == 0:
        return table

    table["p_adjusted"] = benjamini_hochberg(table["p_value"].values, alpha)
    table["significant"] = table["p_adjusted"] < alpha
    return table.sort_values("p_adjusted").reset_index(drop=True)
