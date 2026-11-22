# Turn SMILES strings into the numbers the models learn from.
# These features are the only thing the models ever see, so everything downstream
# (SHAP, applicability domain, bit to substructure mapping) uses this same setup.
#
# Each molecule becomes 2058 numbers:
#   columns 0 .. 2047   Morgan fingerprint COUNTS  (how many times each fragment appears)
#   columns 2048 .. 2057  ten whole-molecule descriptors (size, greasiness, polarity ...)
#
# Why counts rather than plain 0/1: a molecule with six aromatic rings and one with
# a single ring set the same bit, but they do not behave the same way in the liver.
# Why the descriptors: CYP3A4 has a famously large binding pocket and prefers big,
# greasy molecules, which is a whole-molecule property no single fragment bit captures.
# Both choices were tested on a validation split before being adopted; see
# data/processed/model_a_feature_search.csv.

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

FP_SIZE = 2048
FP_RADIUS = 2

# One generator, built once at import time, reused for every molecule.
# explain.py needs this exact object to read the bit info map, so it is shared.
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)

# Ten cheap, interpretable whole-molecule properties.
DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),                       # how heavy
    ("MolLogP", Descriptors.MolLogP),                   # how greasy
    ("TPSA", Descriptors.TPSA),                         # polar surface area
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),  # how floppy
    ("RingCount", Descriptors.RingCount),
    ("FractionCSP3", Descriptors.FractionCSP3),         # how 3D rather than flat
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
]

N_DESCRIPTORS = len(DESCRIPTORS)
N_FEATURES = FP_SIZE + N_DESCRIPTORS

# Readable column names, so a SHAP plot says "MolLogP" instead of "feature 2049".
FEATURE_NAMES = ["bit_%d" % i for i in range(FP_SIZE)] + [name for name, _ in DESCRIPTORS]


def smiles_to_features(smiles):
    # Return the 2058 long feature vector, or None if RDKit cannot parse the SMILES.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_features(mol)


def mol_to_features(mol):
    # Same thing when the molecule has already been parsed.
    row = np.zeros(N_FEATURES, dtype=np.float32)

    # Fragment counts. Only about 50 of the 2048 entries are ever non-zero,
    # so we fill them straight from the sparse dictionary RDKit hands back.
    counts = MORGAN_GEN.GetCountFingerprint(mol).GetNonzeroElements()
    for bit, value in counts.items():
        row[bit] = value

    for j in range(N_DESCRIPTORS):
        try:
            row[FP_SIZE + j] = DESCRIPTORS[j][1](mol)
        except Exception:
            # A handful of odd structures make a descriptor fail. 0.0 is a safer
            # answer than crashing the whole featurisation run.
            row[FP_SIZE + j] = 0.0

    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


def smiles_to_bits(smiles):
    # Plain 0/1 fingerprint, no counts and no descriptors.
    # Used for Tanimoto similarity, which is only defined for sets of bits.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    arr = np.zeros(FP_SIZE, dtype=np.uint8)
    for bit in MORGAN_GEN.GetFingerprint(mol).GetOnBits():
        arr[bit] = 1
    return arr


def bits_from_matrix(X):
    # Recover the 0/1 fingerprints from a feature matrix built above.
    # "appeared at least once" is exactly the plain fingerprint, so the
    # applicability domain never needs a second matrix stored on disk.
    return (np.asarray(X)[:, :FP_SIZE] > 0).astype(np.uint8)


def smiles_list_to_matrix(smiles_list):
    # Featurise many molecules at once.
    # Bad SMILES are skipped, so we also return the indices that survived.
    # Callers need those indices to line the labels back up with the rows.
    rows = []
    kept = []
    for i in range(len(smiles_list)):
        row = smiles_to_features(smiles_list[i])
        if row is None:
            continue
        rows.append(row)
        kept.append(i)
    if len(rows) == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32), kept
    return np.vstack(rows), kept
