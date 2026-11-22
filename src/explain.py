# SHAP explanations, and mapping fingerprint bits back to real chemistry.
# A bit index on its own tells a reader nothing. The point of this module is to
# turn "bit 807 mattered" into "this aromatic ring with a basic nitrogen mattered".

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

import features


def _positive_class_shap(values):
    # SHAP's output layout for a binary classifier changed between versions.
    # Older releases return a list [negative_class_array, positive_class_array].
    # Newer ones return a single array shaped (n_rows, n_features, n_classes).
    # Guessing wrong silently explains the WRONG class, which looks like a
    # working result, so the shape is checked rather than assumed.
    if isinstance(values, list):
        if len(values) == 2:
            return np.asarray(values[1])
        if len(values) == 1:
            return np.asarray(values[0])
        raise ValueError(f"unexpected SHAP list length: {len(values)}")

    arr = np.asarray(values)
    if arr.ndim == 3:
        # (rows, features, classes) -> take the positive class slice.
        return arr[:, :, -1]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"unexpected SHAP array shape: {arr.shape}")


def explain_model(model, X, top_n=25):
    # Rank which fingerprint bits drive the model's positive predictions.
    # Returns (top bit indices, their mean absolute SHAP values, the full
    # positive class SHAP matrix) so the caller can also draw a summary plot.
    #
    # TreeExplainer is exact for tree models, not an approximation, which is
    # why a Random Forest was worth keeping over something less interpretable.
    import shap

    X = np.asarray(X)
    explainer = shap.TreeExplainer(model)
    shap_values = _positive_class_shap(explainer.shap_values(X))

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_n = min(top_n, mean_abs.shape[0])
    top_bits = np.argsort(mean_abs)[::-1][:top_n]
    return top_bits, mean_abs[top_bits], shap_values


def bit_to_atoms(smiles, bit_ids):
    # For one molecule, find which atoms each requested bit came from.
    # Returns {bit id: [atom indices]}, with bits absent from this molecule
    # simply left out.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    # The shared generator from features.py is reused so the bit numbering
    # matches the training data exactly. A second generator built with the same
    # arguments would be fine today, but it is one edit away from silently
    # renumbering every bit, which would make every explanation wrong.
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    features.MORGAN_GEN.GetFingerprint(mol, additionalOutput=ao)
    bit_info = ao.GetBitInfoMap()

    out = {}
    for bit in bit_ids:
        bit = int(bit)
        if bit not in bit_info:
            continue
        atoms = set()
        for centre, radius in bit_info[bit]:
            if radius == 0:
                atoms.add(int(centre))
                continue
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, int(centre))
            for bond_idx in env:
                bond = mol.GetBondWithIdx(bond_idx)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            # A radius with no bonds in its environment still centres on an atom.
            atoms.add(int(centre))
        out[bit] = sorted(atoms)
    return out


def describe_fragment(mol, atom_ids):
    # Plain language description of one fragment, for the report and the demo app.
    #
    # The chemistry flags are read straight off the PARENT molecule atoms that
    # atom_ids points at. RDKit already computed aromaticity, ring membership
    # and valence when it parsed the whole molecule.
    #
    # The fragment is NOT written out as a SMILES string and re-parsed with
    # Chem.MolFromSmiles. A fragment such as "ccc" is not valid standalone
    # SMILES, so MolFromSmiles returns None, and the chemistry check then
    # silently reports nothing instead of raising an error.
    if mol is None or not atom_ids:
        return {"summary": "no fragment", "features": []}

    atoms = [mol.GetAtomWithIdx(int(i)) for i in atom_ids]
    ring_info = mol.GetRingInfo()

    elements = sorted({a.GetSymbol() for a in atoms})
    n_aromatic = sum(1 for a in atoms if a.GetIsAromatic())
    halogens = sorted({a.GetSymbol() for a in atoms if a.GetSymbol() in
                       ("F", "Cl", "Br", "I")})

    # "Basic nitrogen" here means an sp3 nitrogen that is not an amide and not
    # aromatic: the kind that carries a positive charge at blood pH, which is
    # the recurring feature of hERG blockers.
    basic_n = []
    for a in atoms:
        if a.GetSymbol() != "N" or a.GetIsAromatic():
            continue
        if a.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        # Exclude amide nitrogens: N attached to a C that is double bonded to O.
        is_amide = False
        for nbr in a.GetNeighbors():
            if nbr.GetSymbol() != "C":
                continue
            for bond in nbr.GetBonds():
                other = bond.GetOtherAtom(nbr)
                if (bond.GetBondType() == Chem.BondType.DOUBLE
                        and other.GetSymbol() == "O"):
                    is_amide = True
        if not is_amide:
            basic_n.append(a.GetIdx())

    ring_sizes = set()
    for a in atoms:
        for size in (3, 4, 5, 6, 7):
            if ring_info.IsAtomInRingOfSize(a.GetIdx(), size):
                ring_sizes.add(size)

    flags = []
    if n_aromatic:
        flags.append(f"{n_aromatic} aromatic atom(s)")
    if ring_sizes:
        flags.append("ring size " + ", ".join(str(s) for s in sorted(ring_sizes)))
    if basic_n:
        flags.append(f"{len(basic_n)} basic nitrogen(s)")
    if halogens:
        flags.append("halogen: " + ", ".join(halogens))
    if any(a.GetSymbol() == "O" for a in atoms):
        flags.append("oxygen present")

    summary = f"{len(atom_ids)} atoms ({'/'.join(elements)})"
    if flags:
        summary += " - " + "; ".join(flags)

    return {
        "summary": summary,
        "n_atoms": len(atom_ids),
        "elements": elements,
        "n_aromatic": n_aromatic,
        "basic_nitrogens": basic_n,
        "halogens": halogens,
        "ring_sizes": sorted(ring_sizes),
        "features": flags,
    }


def draw_fragment(mol, atom_ids, size=(400, 300), as_svg=True):
    # Highlight the fragment on a picture of the molecule.
    # SVG by default, because MolDraw2DCairo needs a Cairo build that is not
    # always present and fails at import time rather than when drawing.
    if mol is None:
        return None
    atom_ids = [int(i) for i in atom_ids]

    # Highlight the bonds whose BOTH ends are in the fragment, otherwise the
    # picture shows loose highlighted atoms with no connection between them.
    bonds = []
    for bond in mol.GetBonds():
        if (bond.GetBeginAtomIdx() in atom_ids
                and bond.GetEndAtomIdx() in atom_ids):
            bonds.append(bond.GetIdx())

    if as_svg:
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=atom_ids, highlightBonds=bonds
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def top_bit_examples(smiles_list, labels, bit_ids, X, n_examples=3):
    # For each top bit, find a few positive molecules that actually contain it,
    # so the report can show the motif on real inhibitors rather than in the
    # abstract. Returns {bit: [(row index, smiles), ...]}.
    X = np.asarray(X)
    y = np.asarray(labels).astype(bool)
    out = {}
    for bit in bit_ids:
        bit = int(bit)
        has_bit = X[:, bit].astype(bool) & y
        rows = np.flatnonzero(has_bit)[:n_examples]
        out[bit] = [(int(r), smiles_list[int(r)]) for r in rows]
    return out
