# Turn SMILES strings into the numbers the models learn from.
# These features are the only thing the models ever see, so everything downstream
# (SHAP, applicability domain, bit to substructure mapping) uses this same setup.
#
# Each molecule becomes N_FEATURES numbers, in two blocks:
#
#   columns 0 .. FP_SIZE-1        Morgan fingerprint COUNTS
#                                 how many times each chemical fragment occurs
#   columns FP_SIZE .. N_FEATURES-1   ten whole-molecule descriptors
#                                 size, greasiness, polarity, flexibility
#
# Why counts instead of plain 0/1 bits:
#   a molecule with six aromatic rings and a molecule with one ring set exactly
#   the same bit, but they do not behave the same way in the liver. Counting
#   keeps that difference instead of throwing it away.
#
# Why the ten descriptors:
#   CYP3A4 has an unusually large, greasy binding pocket, so whether a molecule
#   fits is partly a question about the WHOLE molecule (how big, how greasy).
#   No single fragment bit can express that.
#
# Both choices were measured on a validation split before being adopted. On a
# scaffold split of CYP3A4_Veith they moved Model A from 0.760 to 0.802 ROC-AUC,
# which was the difference between missing and meeting the 0.80 target.
#
# IMPORTANT, and the reason smiles_to_bits() exists further down:
#   Tanimoto similarity is only defined for SETS of bits, so the applicability
#   domain must never be fed this matrix. Counts are not 0/1, and molecular
#   weight is on a scale of hundreds, so it would swamp everything else and the
#   similarity floor would come out as 0.0 (meaning nothing is ever rejected).

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# radius 2 is the usual ECFP4 style setting for this kind of task.
# 2048 rather than 1024 bits: twice the room means fewer unrelated fragments
# colliding onto the same column, which SHAP would then be unable to tell apart.
FP_SIZE = 2048
FP_RADIUS = 2

# One generator, built once at import time, reused for every molecule.
# explain.py needs this exact object to read the bit info map, so it is shared.
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)

# Ten cheap, interpretable whole-molecule properties. Each is a plain number,
# so SHAP can report "MolLogP pushed this prediction up" in notebook 04.
DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),                          # how heavy
    ("MolLogP", Descriptors.MolLogP),                      # how greasy
    ("TPSA", Descriptors.TPSA),                            # polar surface area
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),  # how floppy
    ("RingCount", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3),            # how 3D rather than flat
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
]

N_DESCRIPTORS = len(DESCRIPTORS)
N_FEATURES = FP_SIZE + N_DESCRIPTORS

# Readable column names, so a SHAP table says "MolLogP" rather than "feature 2049".
FEATURE_NAMES = [f"bit_{i}" for i in range(FP_SIZE)] + [name for name, _ in DESCRIPTORS]


def mol_to_features(mol):
    # Build the feature row for an already parsed molecule.
    # Kept separate from smiles_to_fp so callers that already hold a mol object
    # (explain.py, the app) do not pay to parse the same SMILES twice.
    row = np.zeros(N_FEATURES, dtype=np.float32)

    # Fragment counts. Only about 50 of the 2048 columns are ever non-zero, so
    # we fill them straight from the sparse dictionary RDKit hands back rather
    # than walking all 2048.
    for bit, count in MORGAN_GEN.GetCountFingerprint(mol).GetNonzeroElements().items():
        row[bit] = count

    for position, (_, function) in enumerate(DESCRIPTORS):
        try:
            row[FP_SIZE + position] = function(mol)
        except Exception:
            # A handful of odd structures make one descriptor fail. Writing 0.0
            # is safer than letting one molecule abort a 12k featurisation run.
            row[FP_SIZE + position] = 0.0

    # A descriptor can return inf or nan on a pathological molecule, and either
    # would poison training silently rather than raising.
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


def smiles_to_fp(smiles):
    # Return the N_FEATURES long feature row, or None if RDKit cannot parse it.
    # This is what the models take as input.
    #
    # The name is kept from the earlier fingerprint-only version so that
    # combine.py, the Streamlit app and the tests keep working unchanged.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_features(mol)


def smiles_to_bits(smiles):
    # Plain 0/1 fingerprint, no counts and no descriptors.
    # Use this, and only this, for Tanimoto similarity and the applicability
    # domain. See the warning in the module header above.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bits = np.zeros(FP_SIZE, dtype=np.uint8)
    for bit in MORGAN_GEN.GetFingerprint(mol).GetOnBits():
        bits[bit] = 1
    return bits


def bits_from_matrix(X):
    # Recover the 0/1 fingerprints from a feature matrix built by this module.
    # "the fragment appeared at least once" is exactly the plain fingerprint,
    # so the applicability domain never needs a second matrix saved to disk.
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] < FP_SIZE:
        raise ValueError(
            f"expected at least {FP_SIZE} columns, got {X.shape[1]}. "
            "This matrix was probably built by an older version of features.py, "
            "so rerun notebook 01."
        )
    return (X[:, :FP_SIZE] > 0).astype(np.uint8)


def smiles_list_to_matrix(smiles_list):
    # Featurise many molecules at once.
    # Bad SMILES are skipped, so we also return the indices that survived.
    # Callers need those indices to line the labels back up with the rows.
    rows = []
    kept = []
    for i in range(len(smiles_list)):
        row = smiles_to_fp(smiles_list[i])
        if row is None:
            continue
        rows.append(row)
        kept.append(i)
    if len(rows) == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32), kept
    return np.vstack(rows), kept
