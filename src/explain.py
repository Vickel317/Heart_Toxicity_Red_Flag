# SHAP explanations, and mapping fingerprint bits back to real chemistry.
# A bit index on its own tells a reader nothing. The point of this module is to
# turn "bit 807 mattered" into "this aromatic ring with a basic nitrogen mattered".

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

import features


def shap_values_for_positive_class(model, X):
    # Run SHAP and return one (n_molecules, n_features) array of contributions
    # towards the POSITIVE class.
    #
    # The output layout of shap_values changed between SHAP versions and differs
    # between model types, so the shape is checked rather than assumed:
    #   (n, f)        already the positive class  -> use as is
    #   (n, f, 2)     one slice per class         -> take slice 1
    #   (2, n, f)     classes first               -> take slice 1
    import shap

    explainer = shap.TreeExplainer(model)
    try:
        values = np.asarray(explainer.shap_values(X))
    except Exception as error:
        # A Random Forest over a wide float32 feature matrix can fail SHAP's own
        # additivity check through accumulated rounding, even when the attributions
        # are perfectly usable. We retry without the check rather than give up, and
        # audit_additivity() below measures how far off it actually was, so nobody
        # has to take that on trust.
        if "additivity" not in str(error).lower():
            raise
        print("note: SHAP additivity check failed, retrying without it.")
        print("      run explain.audit_additivity(model, X) to see the real error size.")
        values = np.asarray(explainer.shap_values(X, check_additivity=False))

    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[0] == 2:
            values = values[1]
        else:
            raise ValueError("unexpected SHAP output shape %s" % (values.shape,))

    if values.shape != (X.shape[0], X.shape[1]):
        raise ValueError("SHAP shape %s does not match input %s"
                         % (values.shape, X.shape))
    return values


def audit_additivity(model, X):
    # How well do the SHAP values actually add up?
    #
    # SHAP's promise is: base value + sum of all feature contributions = the model's
    # own output for that molecule. If that holds to a small number, the explanation
    # is faithful whatever the library's internal check decided. This returns the
    # worst gap across the given molecules, in probability units.
    import shap

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X, check_additivity=False))
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] == 2 else values[1]

    base = explainer.expected_value
    base = np.asarray(base).ravel()
    base = float(base[1]) if base.size == 2 else float(base[0])

    reconstructed = base + values.sum(axis=1)
    actual = model.predict_proba(X)[:, 1]

    gap = np.abs(reconstructed - actual)
    return {"max_error": float(gap.max()),
            "mean_error": float(gap.mean()),
            "base_value": base}


def explain_model(model, X, top_n=20, fingerprint_only=True):
    # Rank which features drive the model's predictions, averaged over molecules.
    #
    # mean(|SHAP|) measures how much a feature moves predictions in either
    # direction. mean(SHAP) keeps the sign, so a reader can tell a risk driver
    # from a protective group.
    values = shap_values_for_positive_class(model, X)

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)

    table = pd.DataFrame({
        "feature": np.arange(X.shape[1]),
        "name": features.FEATURE_NAMES[:X.shape[1]],
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,
        "is_fingerprint_bit": np.arange(X.shape[1]) < features.FP_SIZE,
    })
    table["direction"] = np.where(table["mean_shap"] > 0,
                                  "raises risk", "lowers risk")

    if fingerprint_only:
        table = table[table["is_fingerprint_bit"]]

    table = table.sort_values("mean_abs_shap", ascending=False)
    return table.head(top_n).reset_index(drop=True), values


def bit_to_atoms(smiles, bit_ids):
    # For one molecule, find which atoms each requested bit came from.
    # Returns a dict of bit id -> list of atom indices. Bits that do not occur
    # in this molecule are simply absent from the result.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    extra = rdFingerprintGenerator.AdditionalOutput()
    extra.AllocateBitInfoMap()
    # Reuse the shared generator so the bit numbering matches the training data.
    features.MORGAN_GEN.GetFingerprint(mol, additionalOutput=extra)
    info = extra.GetBitInfoMap()

    found = {}
    for bit in bit_ids:
        bit = int(bit)
        if bit not in info:
            continue
        atoms = set()
        for centre, radius in info[bit]:
            if radius == 0:
                atoms.add(centre)
                continue
            # A radius-n environment is returned as bonds; collect their atoms.
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, centre)
            for bond_idx in env:
                bond = mol.GetBondWithIdx(bond_idx)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
        found[bit] = sorted(atoms)
    return found


def describe_fragment(mol, atom_ids):
    # Plain language description of one fragment, for the report and the demo app.
    #
    # The chemistry flags are read straight off the PARENT molecule's atoms.
    # Writing the fragment out as SMILES and re-parsing it would not work: a
    # fragment such as "ccc" is not valid standalone SMILES, so MolFromSmiles
    # returns None and the check silently reports nothing.
    if mol is None or len(atom_ids) == 0:
        return "empty fragment"

    atoms = [mol.GetAtomWithIdx(int(i)) for i in atom_ids]
    notes = []

    if any(a.GetIsAromatic() for a in atoms):
        notes.append("aromatic ring")

    # A basic nitrogen is the classic hERG pharmacophore: protonated at blood pH,
    # it pairs with an acidic residue in the channel pore.
    for a in atoms:
        if a.GetSymbol() != "N":
            continue
        if a.GetIsAromatic():
            continue
        if str(a.GetHybridization()) == "SP3" and a.GetFormalCharge() >= 0:
            notes.append("basic nitrogen")
            break

    halogens = sorted({a.GetSymbol() for a in atoms
                       if a.GetSymbol() in ("F", "Cl", "Br", "I")})
    if halogens:
        notes.append("halogen (" + ", ".join(halogens) + ")")

    if any(a.GetSymbol() == "O" for a in atoms):
        notes.append("oxygen")
    if any(a.GetSymbol() == "S" for a in atoms):
        notes.append("sulfur")

    ring_info = mol.GetRingInfo()
    sizes = set()
    for a in atoms:
        for ring in ring_info.AtomRings():
            if a.GetIdx() in ring:
                sizes.add(len(ring))
    if sizes:
        notes.append("in a " + "/".join(str(s) for s in sorted(sizes)) + " membered ring")

    heavy = len(atoms)
    formula = "".join(sorted(a.GetSymbol() for a in atoms))
    if not notes:
        notes.append("aliphatic fragment")

    return "%d atoms (%s): %s" % (heavy, formula, ", ".join(notes))


def draw_fragment(mol, atom_ids, size=(340, 280)):
    # Highlight the fragment on a picture of the molecule.
    # Returns PNG bytes, which both Jupyter and Streamlit display directly.
    if mol is None:
        return None
    highlight = [int(i) for i in atom_ids]

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def draw_fragment_svg(mol, atom_ids, size=(340, 280)):
    # SVG version, for environments without the Cairo backend.
    if mol is None:
        return None
    highlight = [int(i) for i in atom_ids]

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def example_molecules_for_bit(smiles_list, X, bit, labels=None, n=3):
    # Find molecules that actually contain a given bit, so a reader can see the
    # fragment in context rather than as a number.
    present = np.where(np.asarray(X)[:, int(bit)] > 0)[0]
    if labels is not None:
        labels = np.asarray(labels)
        active = [i for i in present if labels[i] == 1]
        present = np.array(active) if len(active) else present
    chosen = present[:n]
    return [(int(i), smiles_list[int(i)]) for i in chosen]
