# Demo page. Paste a herb compound SMILES, pick a drug, see the tier.
#
# Everything here calls the same functions the notebooks do, so the app cannot
# drift away from the pipeline: the tier shown is the tier combine.py assigns.
#
# The app needs the artefacts notebooks 02, 03 and 05 produce. If they are
# missing it says exactly which notebook to run rather than failing with a
# traceback or, worse, falling back to a made up number.

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

PROCESSED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed")
)

st.set_page_config(page_title="Herb-Drug Cardiac Risk Screen", layout="centered")

st.title("Herb-Drug Cardiac Risk Screen")
st.caption(
    "Course project demo. Every result is a hypothesis for laboratory testing, "
    "not a safety claim about any herb, product, or combination."
)


# --------------------------------------------------------------------- loading

@st.cache_resource(show_spinner="Loading models...")
def load_pipeline():
    # Loaded once per session and cached. Without the cache every click would
    # reload a 500 tree forest and the full training fingerprint matrix.
    import joblib

    import applicability
    import combine

    missing = []
    paths = {
        "model_a": os.path.join(PROCESSED_DIR, "model_a.joblib"),
        "model_b": os.path.join(PROCESSED_DIR, "model_b.joblib"),
        "domain": os.path.join(PROCESSED_DIR, "domain_a.npz"),
        "substrates": os.path.join(PROCESSED_DIR, "cyp3a4_substrates.csv"),
        "qt": os.path.join(PROCESSED_DIR, "qt_drugs.csv"),
    }
    needed_by = {
        "model_a": "notebook 02", "domain": "notebook 02",
        "model_b": "notebook 03",
        "substrates": "notebook 05", "qt": "notebook 05",
    }
    for key, path in paths.items():
        if not os.path.exists(path):
            missing.append((os.path.basename(path), needed_by[key]))
    if missing:
        return None, missing

    bundle = {
        "model_a": joblib.load(paths["model_a"]),
        "model_b": joblib.load(paths["model_b"]),
        "domain": applicability.load_domain(paths["domain"]),
    }
    substrates = pd.read_csv(paths["substrates"])
    qt = pd.read_csv(paths["qt"])
    bundle["substrate_lookup"] = combine.build_substrate_lookup(substrates)
    bundle["qt_lookup"] = combine.build_qt_lookup(qt)
    bundle["drug_names"] = sorted(
        set(substrates["name"].astype(str)) | set(qt["name"].astype(str))
    )

    smiles_path = os.path.join(
        os.path.dirname(PROCESSED_DIR), "raw", "drug_smiles.csv"
    )
    if os.path.exists(smiles_path):
        ds = pd.read_csv(smiles_path)
        bundle["drug_smiles"] = {
            str(n): str(s)
            for n, s in zip(ds["name"], ds["smiles"])
            if isinstance(s, str) and s and s != "nan"
        }
    else:
        bundle["drug_smiles"] = {}
    return bundle, []


@st.cache_resource
def load_rdkit():
    # RDKit is imported lazily so the page can still explain itself on a
    # machine where the install is broken, instead of dying at import time.
    try:
        from rdkit import Chem
        import features
        import explain
        return {"Chem": Chem, "features": features, "explain": explain}, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


bundle, missing = load_pipeline()
rdkit_tools, rdkit_error = load_rdkit()

if rdkit_error:
    st.error(
        "RDKit could not be imported, so no SMILES can be parsed.\n\n"
        f"`{rdkit_error}`\n\n"
        "On Windows this is usually Smart App Control blocking RDKit's compiled "
        "libraries. Running in Google Colab avoids it."
    )
    st.stop()

if bundle is None:
    st.warning("The pipeline is not built yet. These files are missing:")
    st.table(pd.DataFrame(missing, columns=["missing file", "produced by"]))
    st.info(
        "Run the notebooks in order (01 -> 02 -> 03 -> 05), then reload this page. "
        "Notebook 05 also needs the two lookup tables downloaded into `data/raw/`."
    )
    st.stop()

import applicability  # noqa: E402
import combine  # noqa: E402

Chem = rdkit_tools["Chem"]
features = rdkit_tools["features"]
explain = rdkit_tools["explain"]


# ----------------------------------------------------------------------- input

st.subheader("Inputs")

col_left, col_right = st.columns([3, 2])
with col_left:
    smiles = st.text_input(
        "Herb compound SMILES",
        value="",
        placeholder="COc1cc(C=CC(=O)CC(=O)C=Cc2ccc(O)c(OC)c2)ccc1O",
        help="A single compound from the plant, not a whole extract.",
    )
with col_right:
    drug_name = st.selectbox(
        "Drug",
        options=[""] + bundle["drug_names"],
        help="Drugs from the FDA substrate table and the CredibleMeds QT list.",
    )

run = st.button("Screen this pair", type="primary")

if not run:
    st.stop()

if not smiles.strip():
    st.error("Enter a SMILES string for the herb compound.")
    st.stop()
if not drug_name:
    st.error("Pick a drug.")
    st.stop()


# --------------------------------------------------------------------- screening

fp = features.smiles_to_fp(smiles.strip())
if fp is None:
    st.error(
        "RDKit could not parse that SMILES. Check for unbalanced brackets or "
        "ring closure digits."
    )
    st.stop()

mol = Chem.MolFromSmiles(smiles.strip())

# The domain gate is checked BEFORE Model A runs, not after. Outside the domain
# the tier is Unknown regardless, and a probability that the tier logic then
# discards is exactly the number that ends up getting quoted anyway.
domain_status = applicability.check_molecule(fp, bundle["domain"])
domain_score = float(applicability.density_scores(fp, bundle["domain"])[0])

if domain_status == applicability.OUTSIDE:
    cond1 = {"call": None, "prob": None, "source": "not run"}
else:
    cond1 = combine.herb_inhibits_cyp3a4(fp, bundle["model_a"])

cond2 = combine.drug_is_cyp3a4_substrate(drug_name, bundle["substrate_lookup"])

drug_smiles = bundle["drug_smiles"].get(str(drug_name))
drug_fp = features.smiles_to_fp(drug_smiles) if drug_smiles else None
cond3 = combine.drug_blocks_herg(drug_name, drug_fp, bundle["model_b"], bundle["qt_lookup"])

tier = combine.assign_tier(cond1, cond2, cond3, domain_status)


# ----------------------------------------------------------------------- output

st.divider()
st.subheader("Verdict")

if tier == combine.HIGH:
    st.error(f"**{tier} tier** - all three conditions hold and both legs are confident.")
elif tier == combine.MEDIUM:
    st.warning(f"**{tier} tier** - two of three conditions hold, or a leg is borderline.")
elif tier == combine.UNKNOWN:
    st.info(
        f"**{tier} tier** - this compound falls outside Model A's applicability "
        "domain, so the model has no basis for an answer."
    )
else:
    st.success("**Not flagged** - fewer than two conditions hold for this pair.")

score = combine.rank_score(cond1.get("prob"), cond3.get("prob"))
st.caption(
    f"Ranking score {score:.3f}. This orders pairs **inside** a tier only. It is "
    "the weaker of the two model legs, never their product: Random Forest "
    "probabilities are not calibrated, so a product would look like a joint risk "
    "while meaning nothing."
)

st.subheader("Which three conditions held")

def render(label, call, detail):
    icon = {True: "yes", False: "no"}.get(call, "unclear")
    return {"condition": label, "result": icon, "detail": detail}

rows = []
if cond1["call"] is None:
    rows.append(render("1. Herb compound inhibits CYP3A4", None,
                       "Model A not run: outside the applicability domain"))
else:
    rows.append(render(
        "1. Herb compound inhibits CYP3A4",
        cond1["call"] == combine.YES,
        f"Model A p = {cond1['prob']:.3f} ({cond1['call']})",
    ))

rows.append(render(
    "2. Drug is cleared by CYP3A4",
    cond2["call"] == combine.SUBSTRATE,
    f"FDA table lookup: {cond2['call']}"
    + (f", {cond2['strength']} substrate" if cond2.get("strength") else ""),
))

if cond3.get("source") == combine.SOURCE_QT_LIST:
    detail3 = f"CredibleMeds QT list: {cond3.get('risk_category')}"
elif cond3.get("prob") is not None:
    detail3 = f"Model B p = {cond3['prob']:.3f} ({cond3['call']})"
else:
    detail3 = "no SMILES for this drug, so Model B could not run"
rows.append(render("3. Drug blocks hERG or prolongs QT",
                   cond3["call"] == combine.YES, detail3))

st.table(pd.DataFrame(rows))

if cond3.get("source") == combine.SOURCE_QT_LIST:
    st.caption(
        "Condition 3 came from the clinical QT list, not from Model B. A listing "
        "outranks a prediction, so Model B was not consulted for this drug."
    )

st.subheader("Fragments behind the call")

if cond1["call"] is None:
    st.write(
        "Model A was not run on this compound, so there is nothing to attribute."
    )
else:
    try:
        top_bits, _values, _shap = explain.explain_model(
            bundle["model_a"], fp.reshape(1, -1), top_n=6
        )
        atom_map = explain.bit_to_atoms(smiles.strip(), top_bits)
        shown = 0
        for bit in top_bits:
            atoms = atom_map.get(int(bit))
            if not atoms:
                continue
            description = explain.describe_fragment(mol, atoms)
            st.markdown(f"**bit {int(bit)}** - {description['summary']}")
            st.image(explain.draw_fragment(mol, atoms), width=380)
            shown += 1
            if shown >= 3:
                break
        if shown == 0:
            st.write(
                "None of the most influential bits are present in this molecule, "
                "so the call rests on the absence of those motifs rather than on "
                "any fragment that can be highlighted."
            )
    except Exception as exc:
        st.write(f"Fragment attribution unavailable: {type(exc).__name__}: {exc}")

st.subheader("Confidence")

floor = bundle["domain"]["floor"]
if domain_status == applicability.OUTSIDE:
    st.warning(
        f"**Outside the applicability domain.** Mean Tanimoto to the "
        f"{bundle['domain']['k']} nearest training compounds is {domain_score:.3f}, "
        f"below the floor of {floor:.3f}.\n\n"
        "Model A was trained on synthetic, drug-like compounds. Plant natural "
        "products often look nothing like them, and out here the model is "
        "extrapolating rather than predicting. The honest answer is Unknown."
    )
else:
    st.success(
        f"Inside the applicability domain: similarity {domain_score:.3f} against "
        f"a floor of {floor:.3f}."
    )

st.divider()
st.caption(
    "**Limitations that decide what this output means.** Chaining two imperfect "
    "models compounds their error rates. The inhibits / does not inhibit label "
    "ignores potency, so a weak and a potent inhibitor score alike. Jamu is a "
    "multi-compound extract: one compound inhibiting CYP3A4 does not make a "
    "product containing that plant dangerous, because dose, actual content, "
    "absorption and the rest of the extract are not modelled here. Every output "
    "is a hypothesis for laboratory testing, never a safety claim."
)
