# WHY did the model give this molecule that score?
#
# A prediction on its own is not science. "0.97" is not a finding. "This ring
# is why it scored 0.97" is. This file turns the first into the second.
#
# It uses SHAP, which breaks one prediction into one number per feature: how
# much did this feature push the score up or down, for THIS molecule. Those
# numbers are called phi. Positive phi pushed towards "jams the enzyme",
# negative phi pushed away from it.
#
# But SHAP answers in fingerprint box numbers, and "box 1536 mattered" means
# nothing to a human. So the other job of this file is working backwards from a
# box number to the actual atoms that ticked it, and drawing them.

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

import features


def _positive_class_shap(values):
    # Pull out the numbers for the class we care about ("yes, it jams it").
    #
    # Different SHAP versions return this in different shapes. Old ones give a
    # list [no-numbers, yes-numbers]; new ones give one block with the class on
    # the end. Guessing wrong explains the OPPOSITE class and still looks like
    # a working result, so the shape is checked instead of assumed.
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
    # Run SHAP and get one number per feature, per molecule.
    #
    # For tree models SHAP is exact, not an estimate. That is a big reason the
    # project stayed with trees instead of something harder to explain.
    #
    # The try/except handles a real failure, not an imagined one. SHAP checks
    # its own work by asking:
    #
    #     starting point + all the pushes == the model's actual answer?
    #
    # On a big random forest the two sides end up slightly apart from rounding
    # alone, and SHAP refuses rather than returning anything at all. The numbers
    # are still fine, so we retry with the check off and point at
    # audit_additivity below, which measures how far off it really was.
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
    # Check the explanations ourselves: do the pushes add back up to the answer?
    #
    # If starting point + all the pushes returns the model's own output, the
    # explanation is faithful, whatever SHAP's built-in check decided. Returns
    # the worst and the average gap. Anything near 1e-6 or below is as exact as
    # floating point arithmetic gets. Ours comes out around 1e-13.
    import shap

    X = np.asarray(X)
    explainer = shap.TreeExplainer(model)
    values = _positive_class_shap(explainer.shap_values(X, check_additivity=False))

    base = np.asarray(explainer.expected_value).ravel()
    base = float(base[-1]) if base.size >= 2 else float(base[0])

    # What units are the pushes in? Easy to get wrong, and the mistake is loud.
    #
    # Gradient boosting explains in log-odds (its raw internal score), while a
    # random forest explains in probabilities. Compare log-odds against a
    # probability and the gap comes out as whole units, making a perfectly good
    # explanation look broken. That exact mistake happened here: the error read
    # 8.64 until this check was added, then dropped to 3.55e-13.
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
    # Which features matter most to the model overall?
    #
    # Ranked by average size of push, ignoring direction, so a feature that
    # pushes some molecules up and others down still ranks high.
    #
    # fingerprint_only sticks to the fragment boxes, because those are the ones
    # we can turn back into pictures. The ten whole-molecule numbers sit in the
    # columns after them and are ranked separately by top_descriptors below.
    # Mixing them in would hand the fragment-drawing code an index like 2051
    # (which is "MolLogP"), and that is not a fragment, so it cannot be drawn.
    shap_values = shap_matrix(model, X)

    mean_abs = np.abs(shap_values).mean(axis=0)
    searchable = mean_abs[:features.FP_SIZE] if fingerprint_only else mean_abs

    top_n = min(top_n, searchable.shape[0])
    top_bits = np.argsort(searchable)[::-1][:top_n]
    return top_bits, searchable[top_bits], shap_values


def top_descriptors(shap_values, X=None):
    # Rank the ten whole-molecule numbers, and say which way each one points.
    #
    # Read these separately from the fragments. Fragments say which PIECES of a
    # molecule matter; these say what KIND of molecule matters. When size and
    # greasiness come out on top for Model A, that matches the known fact that
    # CYP3A4 has a big greasy pocket, which is evidence the model learned real
    # chemistry instead of memorising its training set.
    #
    # Direction comes from CORRELATION, not from the average push. That is a
    # real bug we hit. The average push describes whichever molecules happen to
    # be in this sample, so a sample of mostly small molecules made MolWt look
    # protective. Correlation answers what we actually mean: "when this number
    # goes UP, does predicted risk go up?" After the fix MolLogP reads r =
    # +0.85, higher raises risk, which is the correct direction.
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
    # Work backwards: which atoms in this molecule ticked box 1536?
    #
    # This is the step that makes the whole SHAP analysis readable. Without it
    # the result is "box 1536 matters" and nobody can act on that. With it the
    # result is "this specific ring matters", which can be drawn and checked
    # against known chemistry.
    #
    # Returns {box number: [atom positions]}. Boxes this molecule does not have
    # are simply left out.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    # Ask RDKit to record which atoms set which box while it builds the
    # fingerprint. Note it reuses the SAME generator object as features.py.
    # Building a second one with identical settings would work today, but one
    # edit to either copy would renumber every box, and then every explanation
    # would quietly be about the wrong fragment.
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
            # Even when the neighbourhood contains no bonds, it still sits on
            # a centre atom, so add that.
            atoms.add(int(centre))
        out[bit] = sorted(atoms)
    return out


def describe_fragment(mol, atom_ids):
    # Describe a fragment in words, for the tables and the demo app.
    #
    # The facts are read off the atoms in the WHOLE molecule, not off a copy of
    # the fragment cut out on its own. That matters: cutting a ring fragment
    # out leaves dangling bonds, and RDKit would then report it as not aromatic
    # when in the real molecule it is. Reading from the parent avoids inventing
    # a chemistry mistake.
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
    # Turn per-FRAGMENT pushes into per-ATOM pushes, so they can be coloured
    # onto a picture of the molecule.
    #
    # The problem: SHAP gives one number per fragment box, but fragments
    # overlap. One atom can belong to five different boxes at once. So each
    # box's push is split evenly over the atoms that produced it, and the atoms
    # add up whatever lands on them. An atom sitting inside several
    # risk-raising fragments ends up hot.
    #
    # This is a reasonable way to share the credit, not the only one. It is
    # also what makes the red/green molecule pictures possible.
    #
    # Returns (molecule, one score per atom), or (None, None) if parsing fails.
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

        # Gather every atom that contributed to this box. One box can be
        # ticked by several places in the same molecule, so collect them all.
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
    # The main entry point: one molecule in, a full explanation out.
    #
    # This is what the demo app and the jamu comparison both call. For a single
    # molecule it answers three things at once: how risky does the model think
    # this is, which parts push that up, and which parts pull it back down.
    row = features.smiles_to_fp(smiles)
    if row is None:
        return None

    X = row.reshape(1, -1)
    probability = float(model.predict_proba(X)[0, 1])
    shap_row = shap_matrix(model, X)[0]

    mol, atom_scores = atom_contributions(smiles, shap_row)

    # Only rank boxes this molecule actually has.
    #
    # A box the molecule does NOT have can still carry a SHAP number, because
    # the model finds a missing fragment informative too. That is real, but it
    # cannot be drawn on a picture of the structure, and reporting "this absent
    # fragment is a risk driver" reads as nonsense. So absences are left out
    # here.
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
