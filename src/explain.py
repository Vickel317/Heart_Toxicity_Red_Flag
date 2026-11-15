# SHAP explanations, and mapping fingerprint bits back to real chemistry.
# A bit index on its own tells a reader nothing. The point of this module is to
# turn "bit 807 mattered" into "this aromatic ring with a basic nitrogen mattered".

# STUB. Nothing here is implemented yet.


def explain_model(model, X):
    # Rank which fingerprint bits drive the model's positive predictions.
    # TODO:
    #   1. explainer = shap.TreeExplainer(model)
    #   2. values = explainer.shap_values(X)
    #   3. For a binary Random Forest, take the positive class slice of the output.
    #      Check the array shape first, the layout changed between SHAP versions.
    #   4. Rank bits by mean absolute SHAP value across the rows.
    #   5. Return the top bit indices and their mean absolute values.
    pass


def bit_to_atoms(smiles, bit_ids):
    # For one molecule, find which atoms each requested bit came from.
    # TODO:
    #   1. mol = Chem.MolFromSmiles(smiles), stop if it is None.
    #   2. ao = rdFingerprintGenerator.AdditionalOutput()
    #   3. ao.AllocateBitInfoMap()
    #   4. features.MORGAN_GEN.GetFingerprint(mol, additionalOutput=ao)
    #      Reuse the shared generator so the bit numbering matches the training data.
    #   5. ao.GetBitInfoMap() maps bit -> ((centre atom index, radius), ...)
    #   6. Expand each (centre, radius) into actual atom indices with
    #      Chem.FindAtomEnvironmentOfRadiusN, then collect the atoms touched by
    #      the bonds it returns.
    #   7. Return a dict of bit id -> list of atom indices.
    pass


def describe_fragment(mol, atom_ids):
    # Plain language description of one fragment, for the report and the demo app.
    # TODO: read the chemistry flags straight off the PARENT molecule atoms that
    # atom_ids points at: basic nitrogen, aromatic ring membership, halogen,
    # ring size, and so on. RDKit already computed all of that when it parsed
    # the whole molecule.
    #
    # Do NOT write the fragment out as a SMILES string and re-parse it with
    # Chem.MolFromSmiles. A fragment such as "ccc" is not valid standalone
    # SMILES, so MolFromSmiles returns None, and the chemistry check then
    # silently reports nothing instead of raising an error.
    pass


def draw_fragment(mol, atom_ids):
    # Highlight the fragment on a picture of the molecule, for the notebook and app.
    # TODO: rdMolDraw2D.MolDraw2DCairo with highlightAtoms set to atom_ids.
    pass
