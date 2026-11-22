# "Do I even know enough to answer this?"
#
# A model will happily give a confident number for ANY molecule you show it,
# including one completely unlike anything it was trained on. That number is
# made up. This file is the gate that catches those cases and says Unknown
# instead, which is the honest answer.
#
# It matters here because Model A learned from synthetic, drug-like molecules,
# and jamu compounds are plant chemistry. They are often built differently. In
# our run 852 of 1,861 jamu compounds (46%) landed outside, against 4.7% of
# held-out drugs.
#
# The rule, in plain terms:
#   1. for a molecule, find its 5 most similar molecules in the training set
#   2. average those 5 similarities -> call it the molecule's "density"
#   3. do that for every training molecule, and take the 5th percentile of
#      those as the "floor" (so the loneliest 5% of training data sets the bar)
#   4. a new molecule is inside the domain if its density clears the floor
#
# Similarity here is Tanimoto: shared fragments divided by total fragments
# across both molecules. 1.0 means identical fragment sets, 0.0 means nothing
# in common.
#
# The floor is worked out from training data ONLY, before any jamu compound is
# scored. Do not re-tune it after seeing the results, or the gate stops meaning
# anything.

import numpy as np

# How many neighbours to average.
K_NEIGHBOURS = 5

# Which percentile of training densities becomes the floor.
FLOOR_PERCENTILE = 5

INSIDE = "inside domain"
OUTSIDE = "outside domain"

# How many rows to compare at a time. The full 12k x 12k similarity table would
# be far too big to hold at once, so it is built in slices.
_CHUNK = 2048


def tanimoto(fp_a, fp_b):
    # Similarity between two molecules, each given as yes/no fragment boxes.
    # shared boxes / (boxes in a + boxes in b - shared). 0.0 if both are empty.
    a = np.asarray(fp_a, dtype=bool)
    b = np.asarray(fp_b, dtype=bool)
    shared = int(np.count_nonzero(a & b))
    total = int(np.count_nonzero(a)) + int(np.count_nonzero(b)) - shared
    if total == 0:
        return 0.0
    return shared / total


def tanimoto_matrix(A, B):
    # Every molecule in A compared against every molecule in B at once.
    #
    # Doing this with a Python loop means one call per pair, which is hopeless
    # at 12,000 x 12,000. The trick: multiplying two rows of 0s and 1s together
    # and summing IS the count of shared boxes. So one matrix multiply gives
    # every "shared" count in the whole table, and the fast linear algebra
    # library does the work instead of Python.
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

    # total is 0 only when both molecules have no boxes ticked at all. Divide
    # carefully and call that pair 0.0 rather than letting it become nan, since
    # a single nan would quietly wreck the percentile calculation later.
    out = np.zeros_like(shared)
    np.divide(shared, total, out=out, where=total > 0)
    return out


def _knn_density(sim_block, k, drop_self):
    # Average the k biggest similarities in each row.
    #
    # drop_self is for when we compare the training set against itself: every
    # molecule is its own best match at 1.0, and counting that would make
    # everything look comfortably similar to everything.
    take = k + 1 if drop_self else k
    take = min(take, sim_block.shape[1])

    # argpartition shoves the `take` biggest values to the end without sorting
    # the whole row, which is much faster than a full sort we do not need.
    idx = np.argpartition(sim_block, -take, axis=1)[:, -take:]
    top = np.take_along_axis(sim_block, idx, axis=1)
    top = np.sort(top, axis=1)[:, ::-1]

    if drop_self:
        # Drop the single highest value in each row, which is the molecule
        # matched against itself. Done by position, not by looking up an index,
        # so that genuine duplicate molecules still lose exactly one entry.
        top = top[:, 1:]
    return top[:, :k].mean(axis=1)


def fit_domain(X_train, k=K_NEIGHBOURS, percentile=FLOOR_PERCENTILE):
    # Work out the floor from the training molecules.
    #
    # Returns a dict rather than one number, because scoring a new molecule
    # later needs the training fingerprints too, to compare against.
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
    # How close is each molecule in X to the training set?
    # Done in slices because notebook 06 scores thousands of jamu compounds.
    Xq = np.asarray(X, dtype=np.uint8)
    if Xq.ndim == 1:
        Xq = Xq.reshape(1, -1)

    X_train = domain["X_train"]
    k = domain["k"]
    out = np.empty(Xq.shape[0], dtype=np.float32)
    for start in range(0, Xq.shape[0], _CHUNK):
        stop = min(start + _CHUNK, Xq.shape[0])
        block = tanimoto_matrix(Xq[start:stop], X_train)
        # drop_self is False here: these molecules are not in the training set,
        # so their best match is a real neighbour, not themselves.
        out[start:stop] = _knn_density(block, k, drop_self=False)
    return out


def check_molecule(fp, domain):
    # Inside or outside, for one molecule.
    score = float(density_scores(fp, domain)[0])
    return INSIDE if score >= domain["floor"] else OUTSIDE


def check_molecules(X, domain):
    # Same thing for many. Returns (list of verdicts, array of scores).
    scores = density_scores(X, domain)
    verdicts = [INSIDE if s >= domain["floor"] else OUTSIDE for s in scores]
    return verdicts, scores


def save_domain(domain, path):
    # The domain carries the entire training fingerprint matrix with it, so it
    # is written compressed.
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
