# Turn SMILES strings into Morgan fingerprints.
# These bits are the only thing the models ever see, so everything downstream
# (SHAP, applicability domain, bit to substructure mapping) uses this same setup.

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# radius 2 over 1024 bits is the usual ECFP4 style setting for this kind of task.
FP_SIZE = 1024
FP_RADIUS = 2

# One generator, built once at import time, reused for every molecule.
# explain.py needs this exact object to read the bit info map, so it is shared.
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)


def smiles_to_fp(smiles):
    # Return a 1024 long numpy array of 0s and 1s, or None if RDKit cannot parse the SMILES.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bitvect = MORGAN_GEN.GetFingerprint(mol)
    # Only about 50 of the 1024 bits are ever set, so filling them one by one
    # is cheap and avoids relying on the RDKit to numpy conversion helper.
    arr = np.zeros(FP_SIZE, dtype=np.uint8)
    for bit in bitvect.GetOnBits():
        arr[bit] = 1
    return arr


def smiles_list_to_matrix(smiles_list):
    # Featurise many molecules at once.
    # Bad SMILES are skipped, so we also return the indices that survived.
    # Callers need those indices to line the labels back up with the rows.
    rows = []
    kept = []
    for i in range(len(smiles_list)):
        fp = smiles_to_fp(smiles_list[i])
        if fp is None:
            continue
        rows.append(fp)
        kept.append(i)
    if len(rows) == 0:
        return np.zeros((0, FP_SIZE), dtype=np.uint8), kept
    return np.vstack(rows), kept
