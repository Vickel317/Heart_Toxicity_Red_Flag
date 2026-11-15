# Applicability domain check.
# A model prediction only means something if the new molecule resembles the
# training set. Plant natural products often do not look like the synthetic
# drug-like compounds in CYP3A4_Veith, so this gate produces the Unknown tier.

# STUB. Nothing here is implemented yet.


def tanimoto(fp_a, fp_b):
    # Tanimoto similarity between two 0/1 numpy fingerprint arrays.
    # TODO:
    #   1. shared = number of bits set in both.
    #   2. total = bits set in a, plus bits set in b, minus shared.
    #   3. Return shared / total, and 0.0 when total is 0.
    pass


def fit_domain(X_train):
    # Work out the similarity floor from the training fingerprints.
    # TODO the rule:
    #   1. For each training molecule, compute its Tanimoto similarity to every
    #      other training molecule.
    #   2. Take its 5 nearest neighbours, excluding itself.
    #   3. Average those 5 similarities. That is the molecule's density score.
    #   4. The floor is the 5th percentile of all the density scores, so the
    #      sparsest 5 percent of the training set sets the bar.
    #   5. Return the floor together with X_train, because scoring a new
    #      molecule needs both.
    pass


def check_molecule(fp, domain):
    # Decide whether one new molecule is inside the domain.
    # TODO:
    #   1. Tanimoto of fp against every training fingerprint in domain.
    #   2. Mean of the 5 highest of those similarities.
    #   3. Below the floor returns "outside domain", otherwise "inside domain".
    # Returning a placeholder so callers can be written before this works.
    return "unknown"
