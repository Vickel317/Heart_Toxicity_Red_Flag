# Applicability domain check.
# A model prediction only means something if the new molecule resembles the
# training set. Plant natural products often do not look like the synthetic
# drug-like compounds in CYP3A4_Veith, so this gate produces the Unknown tier.
#
# The rule (fixed, do not tune it after seeing the herb results):
#   density(molecule) = mean Tanimoto to its K nearest training neighbours
#   floor             = 5th percentile of density over the training set
#   a new molecule is inside the domain if its density >= floor
#
# So the sparsest 5 percent of the training set sets the bar. The floor is
# computed from training data ONLY, before any herb compound is scored.

import numpy as np

# Number of nearest neighbours averaged into a density score.
K_NEIGHBOURS = 5

# The percentile of training densities used as the floor.
FLOOR_PERCENTILE = 5

INSIDE = "inside domain"
OUTSIDE = "outside domain"

# Rows per chunk when building a similarity block. 2048 x 12000 float32 is
# about 100 MB, which keeps the full 12k x 12k comparison off the heap.
_CHUNK = 2048


def tanimoto(fp_a, fp_b):
    # Tanimoto similarity between two 0/1 numpy fingerprint arrays.
    # shared / (set in a + set in b - shared), and 0.0 when neither has any bit set.
    a = np.asarray(fp_a, dtype=bool)
    b = np.asarray(fp_b, dtype=bool)
    shared = int(np.count_nonzero(a & b))
    total = int(np.count_nonzero(a)) + int(np.count_nonzero(b)) - shared
    if total == 0:
        return 0.0
    return shared / total


def tanimoto_matrix(A, B):
    # All pairwise Tanimoto similarities between the rows of A and the rows of B.
    # Returns an (len(A), len(B)) float32 array.
    #
    # The loop version of this is O(n*m) Python calls, which is hopeless at
    # 12k x 12k. Written as matrix algebra instead: the dot product of two 0/1
    # rows IS the count of shared bits, so one matmul gives every intersection
    # at once and BLAS does the work.
    Af = np.asarray(A, dtype=np.float32)
    Bf = np.asarray(B, dtype=np.float32)
    if Af.ndim == 1:
        Af = Af.reshape(1, -1)
    if Bf.ndim == 1:
        Bf = Bf.reshape(1, -1)

    shared = Af @ Bf.T
    bits_a = Af.sum(axis=1, dtype=np.float32).reshape(-1, 1)
    bits_b = Bf.sum(axis=1, dtype=np.float32).reshape(1, -1)
    total = bits_a + bits_b - shared

    # total is 0 only when both molecules are all-zero fingerprints. Divide
    # safely and call that pair 0.0 similarity rather than letting it be nan,
    # because a nan would silently poison the percentile later.
    out = np.zeros_like(shared)
    np.divide(shared, total, out=out, where=total > 0)
    return out


def _knn_density(sim_block, k, drop_self):
    # Mean of the k highest similarities in each row of sim_block.
    # drop_self removes one extra column per row, used when the block is the
    # training set against itself and the diagonal is a molecule's own 1.0.
    take = k + 1 if drop_self else k
    take = min(take, sim_block.shape[1])

    # argpartition puts the `take` largest in the tail without a full sort.
    idx = np.argpartition(sim_block, -take, axis=1)[:, -take:]
    top = np.take_along_axis(sim_block, idx, axis=1)
    top = np.sort(top, axis=1)[:, ::-1]

    if drop_self:
        # The single highest value in each row is the molecule matched to
        # itself. Dropping column 0 excludes it. Done positionally rather than
        # by index so exact duplicate molecules still lose exactly one row.
        top = top[:, 1:]
    return top[:, :k].mean(axis=1)


def fit_domain(X_train, k=K_NEIGHBOURS, percentile=FLOOR_PERCENTILE):
    # Work out the similarity floor from the training fingerprints.
    # Returns a dict, because scoring a new molecule needs both the floor and
    # the training fingerprints it gets compared against.
    X = np.asarray(X_train, dtype=np.uint8)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("fit_domain needs a non-empty 2D fingerprint matrix")

    n = X.shape[0]
    k_eff = min(k, max(1, n - 1))

    densities = np.empty(n, dtype=np.float32)
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        block = tanimoto_matrix(X[start:stop], X)
        densities[start:stop] = _knn_density(block, k_eff, drop_self=True)

    floor = float(np.percentile(densities, percentile))
    return {
        "floor": floor,
        "X_train": X,
        "k": k_eff,
        "percentile": percentile,
        "train_densities": densities,
    }


def density_scores(X, domain):
    # Density score for each row of X against the training set in domain.
    # Batched, because notebook 06 scores thousands of herb compounds at once.
    Xq = np.asarray(X, dtype=np.uint8)
    if Xq.ndim == 1:
        Xq = Xq.reshape(1, -1)

    X_train = domain["X_train"]
    k = domain["k"]
    out = np.empty(Xq.shape[0], dtype=np.float32)
    for start in range(0, Xq.shape[0], _CHUNK):
        stop = min(start + _CHUNK, Xq.shape[0])
        block = tanimoto_matrix(Xq[start:stop], X_train)
        # drop_self is False: a query molecule is not part of the training set,
        # so its top similarity is a genuine neighbour, not itself.
        out[start:stop] = _knn_density(block, k, drop_self=False)
    return out


def check_molecule(fp, domain):
    # Decide whether one new molecule is inside the domain.
    score = float(density_scores(fp, domain)[0])
    return INSIDE if score >= domain["floor"] else OUTSIDE


def check_molecules(X, domain):
    # Batch version. Returns (list of verdicts, array of density scores).
    scores = density_scores(X, domain)
    verdicts = [INSIDE if s >= domain["floor"] else OUTSIDE for s in scores]
    return verdicts, scores


def save_domain(domain, path):
    # The domain carries the whole training matrix, so it is saved compressed.
    np.savez_compressed(
        path,
        floor=np.float32(domain["floor"]),
        X_train=domain["X_train"],
        k=np.int32(domain["k"]),
        percentile=np.int32(domain["percentile"]),
        train_densities=domain["train_densities"],
    )


def load_domain(path):
    z = np.load(path)
    return {
        "floor": float(z["floor"]),
        "X_train": z["X_train"],
        "k": int(z["k"]),
        "percentile": int(z["percentile"]),
        "train_densities": z["train_densities"],
    }
