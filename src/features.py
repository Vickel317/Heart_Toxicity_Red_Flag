# Turn a molecule into numbers, because models cannot read chemistry.
#
# A molecule arrives as a SMILES string, which is just text: "CCO" is ethanol.
# A model cannot learn from text, so every molecule has to become a fixed-length
# list of numbers first. This file does that, and nothing else in the project
# does it, so every part of the pipeline sees molecules the same way.
#
# Each molecule becomes N_FEATURES numbers, in two blocks:
#
#   columns 0 .. FP_SIZE-1    how many times each small fragment appears
#   the last ten columns      ten facts about the whole molecule
#
# The first block is a "Morgan fingerprint". The idea: walk over every atom,
# look at the little neighbourhood around it, and tick a box for each distinct
# neighbourhood you find. Molecules that share fragments tick the same boxes.
#
# Why count the fragments instead of just ticking yes/no:
#   one aromatic ring and six aromatic rings tick the same box, but they do not
#   behave the same way in the liver. Counting keeps that difference.
#
# Why the extra ten columns:
#   CYP3A4 (the liver enzyme) has a big greasy pocket, so whether a molecule
#   fits is partly about the whole molecule, not its pieces. "How heavy is it"
#   is not something any single fragment box can say.
#
# Both choices were tested, not guessed. They moved Model A from 0.760 to 0.802
# ROC-AUC, which was the difference between missing and hitting the 0.80 target.
#
# ONE TRAP, and the reason smiles_to_bits() exists below:
#   the applicability domain compares molecules with Tanimoto similarity, which
#   only works on yes/no boxes. Feed it this matrix and it breaks twice over:
#   the counts are not 0/1, and molecular weight is in the hundreds so it drowns
#   out everything else. The symptom is a similarity floor of 0.0, meaning no
#   molecule is ever rejected. Use smiles_to_bits for that, always.

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# radius 2 = look two bonds out from each atom. This is the standard setting.
# 2048 boxes rather than 1024: with more boxes, fewer unrelated fragments get
# squashed into the same one, which would leave SHAP unable to tell them apart.
FP_SIZE = 2048
FP_RADIUS = 2

# Built once when this file is imported, then reused for every molecule.
# explain.py needs this exact object to ask "which atoms set box 1536?", so it
# is shared rather than rebuilt.
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)

# Ten plain facts about the whole molecule. Each is one number, so SHAP can say
# "MolLogP is what pushed this prediction up" in notebook 04.
DESCRIPTORS = [
    ("MolWt", Descriptors.MolWt),                          # how heavy
    ("MolLogP", Descriptors.MolLogP),                      # how greasy
    ("TPSA", Descriptors.TPSA),                            # how much polar surface
    ("NumHDonors", Descriptors.NumHDonors),                # can donate H bonds
    ("NumHAcceptors", Descriptors.NumHAcceptors),          # can accept H bonds
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),  # how floppy
    ("RingCount", Descriptors.RingCount),                  # how many rings
    ("FractionCSP3", Descriptors.FractionCSP3),            # how 3D rather than flat
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),        # atoms, ignoring hydrogens
    ("NumAromaticRings", Descriptors.NumAromaticRings),    # how many flat rings
]

N_DESCRIPTORS = len(DESCRIPTORS)
N_FEATURES = FP_SIZE + N_DESCRIPTORS

# Column names, so a SHAP table says "MolLogP" and not "feature 2049".
FEATURE_NAMES = [f"bit_{i}" for i in range(FP_SIZE)] + [name for name, _ in DESCRIPTORS]


def mol_to_features(mol):
    # Build the number row for a molecule that is already parsed.
    # Separate from smiles_to_fp so callers holding a parsed molecule already
    # (explain.py, the app) do not pay to parse the same text twice.
    row = np.zeros(N_FEATURES, dtype=np.float32)

    # Fill in the fragment counts. Only ~50 of the 2048 boxes are ever non-zero,
    # so read straight from the short list RDKit gives back instead of looping
    # over all 2048 and writing mostly zeros.
    for bit, count in MORGAN_GEN.GetCountFingerprint(mol).GetNonzeroElements().items():
        row[bit] = count

    for position, (_, function) in enumerate(DESCRIPTORS):
        try:
            row[FP_SIZE + position] = function(mol)
        except Exception:
            # A few strange structures make one descriptor fail. Writing 0.0 is
            # better than letting one bad molecule kill a 12,000 molecule run.
            row[FP_SIZE + position] = 0.0

    # A descriptor can also return infinity or nan on a broken molecule. Either
    # one would quietly ruin training instead of raising an error, so clean them.
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


def smiles_to_fp(smiles):
    # Text in, number row out. None if RDKit cannot make sense of the text.
    # This is what the models take as input.
    #
    # The name is left over from when this returned only a fingerprint. It is
    # kept so combine.py, the app and the tests do not all have to change.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_features(mol)


def smiles_to_bits(smiles):
    # Yes/no boxes only: no counts, no whole-molecule facts.
    # This is the one to use for Tanimoto similarity and the applicability
    # domain. See the trap described at the top of this file.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bits = np.zeros(FP_SIZE, dtype=np.uint8)
    for bit in MORGAN_GEN.GetFingerprint(mol).GetOnBits():
        bits[bit] = 1
    return bits


def bits_from_matrix(X):
    # Get the yes/no boxes back out of a matrix this file already built.
    # "this fragment appeared at least once" is exactly the yes/no version, so
    # the domain never needs a second copy of the fingerprints saved to disk.
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
    # Do many molecules at once.
    #
    # Unparseable ones get skipped, so we also return which positions survived.
    # The caller needs those to line its labels and names back up with the rows,
    # otherwise row 5 gets the name of molecule 5 when it is really molecule 7.
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
