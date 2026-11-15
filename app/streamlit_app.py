# Minimal demo page. Paste a herb compound SMILES, pick a drug, see the tier.
# Everything below the input is placeholder text for now.

# STUB. The models are not wired in yet.

import streamlit as st

st.set_page_config(page_title="Herb-Drug Cardiac Risk Screen")

st.title("Herb-Drug Cardiac Risk Screen")
st.caption(
    "Course project demo. Every result is a hypothesis for laboratory testing, "
    "not a safety claim about any herb, product, or combination."
)

smiles = st.text_input("Herb compound SMILES", value="")
drug_name = st.text_input("Drug name", value="")

if st.button("Screen this pair"):
    # TODO wire this up to the real pipeline:
    #   1. features.smiles_to_fp(smiles), and show an error if it returns None.
    #   2. applicability.check_molecule to catch the Unknown tier before predicting.
    #   3. combine.assign_tier with Model A, the FDA substrate table, Model B,
    #      and the CredibleMeds QT list.
    #   4. explain.bit_to_atoms plus explain.draw_fragment to show which parts
    #      of the molecule drove Model A.
    #   5. Load the trained models once and cache them with st.cache_resource,
    #      otherwise every click retrains or reloads from disk.

    st.subheader("Verdict")
    st.write("placeholder: tier not computed yet")

    st.subheader("Which three conditions held")
    st.write("placeholder: condition 1 (herb inhibits CYP3A4) not computed yet")
    st.write("placeholder: condition 2 (drug cleared by CYP3A4) not computed yet")
    st.write("placeholder: condition 3 (drug blocks hERG or prolongs QT) not computed yet")

    st.subheader("Fragments behind the call")
    st.write("placeholder: fragment highlighting not computed yet")

    st.subheader("Confidence warning")
    st.warning(
        "placeholder: this is where the applicability domain warning goes, "
        "for herb compounds that do not resemble the training set."
    )
