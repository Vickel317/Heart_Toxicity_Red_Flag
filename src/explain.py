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


def shap_matrix(model, X):
    # Run SHAP once and hand back the positive class contributions.
    #
    # TreeExplainer is exact for tree models, not an approximation, which is why
    # a tree model was worth keeping over something less interpretable.
    #
    # The try/except is not defensive padding, it handles a real failure:
    # on a random forest grown on 12k molecules, SHAP's own additivity check
    # raises outright. The check asks whether
    #
    #     base value + sum of contributions == the model's own prediction
    #
    # and on a deep forest over a wide float32 matrix the two sides drift apart
    # by rounding alone, even though the contributions are perfectly usable.
    # Rather than crash or silently ignore it, we retry without the check and
    # point the reader at audit_additivity(), which measures the real error.
    import shap

    X = np.asarray(X)
    explainer = shap.TreeExplainer(model)
    try:
        return _positive_class_shap(explainer.shap_values(X))
    except Exception as error:
        if "additivity" not in str(error).lower():
            raise
        print("note: SHAP additivity check failed, retrying without it.")
        print("      call explain.audit_additivity(model, X) to see the error size.")
        return _positive_class_shap(explainer.shap_values(X, check_additivity=False))


def audit_additivity(model, X):
    # How well do the SHAP values actually add up?
    #
    # SHAP's promise is that base value + sum of contributions reproduces the
    # model's own output for each molecule. If that holds to a small number, the
    # explanation is faithful whatever the library's internal check decided.
    # Returns the worst and average gap, in probability units: anything near
    # 1e-6 is exact to floating point precision.
    import shap

    X = np.asarray(X)
    explainer = shap.TreeExplainer(model)
    values = _positive_class_shap(explainer.shap_values(X, check_additivity=False))

    base = np.asarray(explainer.expected_value).ravel()
    base = float(base[-1]) if base.size >= 2 else float(base[0])

    # Which scale are the contributions on? This is easy to get wrong and the
    # mistake is loud: gradient boosting hands back LOG-ODDS (the raw margin),
    # while a random forest hands back probabilities. Comparing log-odds against
    # predict_proba gives errors of several whole units and makes a perfectly
    # faithful explanation look broken.
    if hasattr(model, "decision_function"):
        actual = model.decision_function(X)
        scale = "log-odds"
    else:
        actual = model.predict_proba(X)[:, 1]
        scale = "probability"

    gap = np.abs(base + values.sum(axis=1) - actual)
    return {"max_error": float(gap.max()),
            "mean_error": float(gap.mean()),
            "base_value": base,
            "scale": scale}


def explain_model(model, X, top_n=25, fingerprint_only=True):
    # Rank which features drive the model's positive predictions.
    # Returns (top indices, their mean absolute SHAP values, the full positive
    # class SHAP matrix) so the caller can also draw a summary plot.
    #
    # fingerprint_only keeps the ranking to the Morgan columns, because those
    # are the ones bit_to_atoms() can turn back into pictures of chemistry. The
    # ten descriptors live in the columns after them and are ranked separately
    # by top_descriptors() below; mixing them in would hand bit_to_atoms an
    # index such as 2051 ("MolLogP") that corresponds to no fragment at all.
    shap_values = shap_matrix(model, X)

    mean_abs = np.abs(shap_values).mean(axis=0)
    searchable = mean_abs[:features.FP_SIZE] if fingerprint_only else mean_abs

    top_n = min(top_n, searchable.shape[0])
    top_bits = np.argsort(searchable)[::-1][:top_n]
    return top_bits, searchable[top_bits], shap_values


def top_descriptors(shap_values, X=None):
    # The ten whole-molecule descriptors, ranked, with their direction.
    #
    # Worth reading separately from the bits: the bits say which PIECES of a
    # molecule matter, the descriptors say what KIND of molecule matters. If
    # size and greasiness rank high for Model A, that matches what is known
    # about CYP3A4's large hydrophobic pocket, which is a sign the model learned
    # chemistry rather than memorising the training set.
    #
    # Direction is read as the CORRELATION between a descriptor's value and its
    # SHAP contribution, not from the sign of the mean contribution. The mean
    # sign answers the wrong question: it describes the particular molecules in
    # this sample, so a sample of mostly small molecules makes MolWt look like a
    # "protective" feature. The correlation answers the question actually being
    # asked, which is "as this descriptor goes UP, does predicted risk go up?".
    shap_values = np.asarray(shap_values)
    if shap_values.shape[1] <= features.FP_SIZE:
        return []

    rows = []
    for position, (name, _) in enumerate(features.DESCRIPTORS):
        column = shap_values[:, features.FP_SIZE + position]

        direction = "unknown"
        correlation = float("nan")
        if X is not None:
            values = np.asarray(X)[:, features.FP_SIZE + position]
            # A constant descriptor has no direction to report, and np.corrcoef
            # would return nan with a runtime warning.
            if values.std() > 0 and column.std() > 0:
                correlation = float(np.corrcoef(values, column)[0, 1])
                direction = "higher raises risk" if correlation > 0 else "higher lowers risk"

        rows.append({
            "name": name,
            "mean_abs_shap": float(np.abs(column).mean()),
            "mean_shap": float(column.mean()),
            "correlation": correlation,
            "direction": direction,
        })
    return sorted(rows, key=lambda r: r["mean_abs_shap"], reverse=True)


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


# =====================================================================
# Atom level attribution: from "bit 1873 mattered" to "THIS ring mattered"
#
# This is the part the proposal describes as mapping SHAP weights back to 2D
# atomic coordinates. Everything above works at the level of a fingerprint bit,
# which is a hashed fragment, not a place on a molecule. A chemist reading the
# report wants to see the atoms.
#
# How the attribution works. Each Morgan bit was generated by one or more
# (centre atom, radius) environments, so we know exactly which atoms produced
# it. A bit's SHAP value is therefore shared out equally among the atoms that
# produced it, and an atom's score is the sum of every share it receives.
#
# What the sign means, following the proposal:
#   score > 0   this part of the molecule pushes the prediction towards
#               inhibition (CYP3A4) or block (hERG)   -> risk driver
#   score < 0   this part pushes the prediction away                -> protective
#
# One honest limitation: Morgan environments overlap heavily, so neighbouring
# atoms share credit and a single atom's score is not a measurement of that one
# atom's chemistry. Read regions, not individual atoms.
# =====================================================================


def atom_contributions(smiles, shap_row):
    # Spread the fingerprint part of one molecule's SHAP row over its atoms.
    # Returns (mol, per-atom score array), or (None, None) if parsing fails.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    shap_row = np.asarray(shap_row).ravel()

    extra = rdFingerprintGenerator.AdditionalOutput()
    extra.AllocateBitInfoMap()
    features.MORGAN_GEN.GetFingerprint(mol, additionalOutput=extra)
    bit_info = extra.GetBitInfoMap()

    scores = np.zeros(mol.GetNumAtoms(), dtype=float)

    for bit, environments in bit_info.items():
        if bit >= shap_row.shape[0]:
            continue
        value = float(shap_row[bit])
        if value == 0.0:
            continue

        # Collect every atom this bit was computed from, across all the
        # environments that hashed to it.
        atoms = set()
        for centre, radius in environments:
            if radius == 0:
                atoms.add(int(centre))
                continue
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, centre)
            for bond_idx in env:
                bond = mol.GetBondWithIdx(bond_idx)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            atoms.add(int(centre))

        if not atoms:
            continue
        share = value / len(atoms)
        for atom_idx in atoms:
            scores[atom_idx] += share

    return mol, scores


def draw_atom_heatmap(mol, atom_scores, size=(420, 380), as_svg=True):
    # Green/red heatmap over the molecule: red regions drive the prediction up,
    # green regions pull it down. This is the figure for the report.
    from rdkit.Chem.Draw import SimilarityMaps

    if mol is None or atom_scores is None:
        return None

    drawer = (rdMolDraw2D.MolDraw2DSVG(size[0], size[1]) if as_svg
              else rdMolDraw2D.MolDraw2DCairo(size[0], size[1]))
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol, [float(s) for s in atom_scores], draw2d=drawer
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def explain_molecule(model, smiles, top_n=5):
    # The reusable entry point: one SMILES in, a full explanation out.
    #
    # This is what the demo app and the jamu comparison both call. It answers,
    # for a single molecule: how risky does the model think this is, which parts
    # drive that upwards, and which parts pull it back down.
    row = features.smiles_to_fp(smiles)
    if row is None:
        return None

    X = row.reshape(1, -1)
    probability = float(model.predict_proba(X)[0, 1])
    shap_row = shap_matrix(model, X)[0]

    mol, atom_scores = atom_contributions(smiles, shap_row)

    # Rank only the bits this molecule actually contains. A bit it does not
    # contain can still carry a SHAP value (absence is informative to the
    # model), but it cannot be drawn on the structure.
    present = np.flatnonzero(np.asarray(row)[:features.FP_SIZE] > 0)
    ranked = sorted(present, key=lambda b: abs(shap_row[b]), reverse=True)

    bit_atoms = bit_to_atoms(smiles, ranked[:top_n * 3])
    fragments = []
    for bit in ranked:
        if len(fragments) >= top_n:
            break
        atoms = bit_atoms.get(int(bit), [])
        if not atoms:
            continue
        fragments.append({
            "bit": int(bit),
            "shap": float(shap_row[bit]),
            "role": "risk driver" if shap_row[bit] > 0 else "protective",
            "atoms": atoms,
            "description": describe_fragment(mol, atoms)["summary"],
        })

    return {
        "smiles": smiles,
        "probability": probability,
        "mol": mol,
        "atom_scores": atom_scores,
        "shap_row": shap_row,
        "fragments": fragments,
        "descriptors": top_descriptors(shap_row.reshape(1, -1), X),
    }
