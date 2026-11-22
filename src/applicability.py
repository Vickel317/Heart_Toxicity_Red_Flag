# Applicability domain check.
# A model prediction only means something if the new molecule resembles the
# training set. Plant natural products often do not look like the synthetic
# drug-like compounds in CYP3A4_Veith, so this gate produces the Unknown tier.

import numpy as np

# How many nearest training neighbours define "is this molecule in familiar territory".
N_NEIGHBOURS = 5

# The sparsest 5 percent of the training set sets the bar. Anything less typical
# than that is called outside the domain.
DOMAIN_PERCENTILE = 5


def tanimoto(fp_a, fp_b):
    # Tanimoto similarity between two 0/1 fingerprint arrays.
    # shared bits / total distinct bits. 1.0 means identical fingerprints.
    fp_a = np.asarray(fp_a)
    fp_b = np.asarray(fp_b)
    shared = float(np.logical_and(fp_a, fp_b).sum())
    total = float(fp_a.sum()) + float(fp_b.sum()) - shared
    if total == 0:
        return 0.0
    return shared / total


def tanimoto_matrix(A, B):
    # Same formula as tanimoto(), but for every row of A against every row of B
    # at once. Comparing 12k molecules to each other one pair at a time would be
    # 152 million Python loops; as a matrix multiply it takes a few seconds.
    #
    # The trick: for 0/1 vectors, the dot product IS the count of shared bits.
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    shared = A @ B.T
    bits_a = A.sum(axis=1)[:, None]
    bits_b = B.sum(axis=1)[None, :]
    total = bits_a + bits_b - shared
    # where=total > 0 avoids dividing by zero for empty fingerprints.
    return np.divide(shared, total, out=np.zeros_like(shared), where=total > 0)


def _mean_top_neighbours(sim_rows, n_neighbours):
    # Average of the n highest similarities in each row.
    # np.partition is used instead of sorting because we only need the top n,
    # not the full ordering.
    n_neighbours = min(n_neighbours, sim_rows.shape[1])
    top = np.partition(sim_rows, -n_neighbours, axis=1)[:, -n_neighbours:]
    return top.mean(axis=1)


def fit_domain(X_train, chunk_size=512):
    # Work out the similarity floor from the training fingerprints.
    # For each training molecule: how similar is it to its 5 nearest neighbours?
    # A molecule sitting in a crowded, well covered region scores high.
    # The floor is the 5th percentile of those scores.
    X_train = np.asarray(X_train, dtype=np.uint8)
    n = len(X_train)
    scores = np.zeros(n, dtype=np.float32)

    # Done in chunks of rows so we never hold a 12k x 12k matrix in memory.
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sim = tanimoto_matrix(X_train[start:stop], X_train)
        # A molecule is always identical to itself, so blank out the diagonal
        # before taking neighbours, otherwise every score is inflated by a 1.0.
        for row in range(stop - start):
            sim[row, start + row] = -1.0
        scores[start:stop] = _mean_top_neighbours(sim, N_NEIGHBOURS)

    floor = float(np.percentile(scores, DOMAIN_PERCENTILE))
    return {"X_train": X_train, "floor": floor, "train_scores": scores}


def domain_scores(X_new, domain, chunk_size=512):
    # Similarity-to-training score for many new molecules at once.
    # Notebook 06 screens thousands of herb compounds, so this is the batch path.
    X_new = np.asarray(X_new, dtype=np.uint8)
    out = np.zeros(len(X_new), dtype=np.float32)
    for start in range(0, len(X_new), chunk_size):
        stop = min(start + chunk_size, len(X_new))
        sim = tanimoto_matrix(X_new[start:stop], domain["X_train"])
        out[start:stop] = _mean_top_neighbours(sim, N_NEIGHBOURS)
    return out


def check_molecule(fp, domain):
    # Decide whether one new molecule is inside the domain.
    score = float(domain_scores(np.asarray(fp)[None, :], domain)[0])
    if score < domain["floor"]:
        return "outside domain"
    return "inside domain"
