# Herb-Drug Cardiac Risk Screening (jamu x prescription drugs)

Computational biology course project. Scaffold stage.

## The question

Jamu (Indonesian herbal medicine) is often taken at the same time as prescription
drugs. Some plant compounds inhibit CYP3A4, the liver enzyme that clears a large
share of all drugs. If a drug depends on CYP3A4 for clearance, and that same drug
also blocks the hERG potassium channel, then slower clearance could raise its blood
level and lengthen the QT interval.

This project builds a narrow screen that lists which (herb compound, drug) pairs sit
on that path, so they can be prioritised for laboratory testing.

## The three conditions

A pair is flagged only if all three are true:

1. **The herb compound inhibits CYP3A4.** Predicted by Model A, run on the herb compound.
2. **The drug is cleared by CYP3A4.** Looked up in the FDA CYP3A4 substrate table. This is
   a lookup, not a prediction.
3. **The drug blocks hERG or prolongs QT.** Predicted by Model B, or taken from the
   CredibleMeds QT list.

The two model probabilities are never multiplied together. Neither model is calibrated,
so their product has no meaning. The screen reports a tier instead:

| Tier | Meaning |
| --- | --- |
| High | All three conditions hold, and both model legs are confident. |
| Medium | Two of three hold, or at least one leg is low confidence. |
| Unknown | The herb compound falls outside the training applicability domain, so the model has no basis for an answer. |

A continuous score exists only to rank pairs inside a single tier. It never moves a pair
between tiers.

## Models and targets

| Model | Dataset | Size | Target | Achieved |
| --- | --- | --- | --- | --- |
| Model A | TDC `CYP3A4_Veith` | 12,328 compounds | ROC-AUC >= 0.80 | **0.802** scaffold split (0.913 random) |
| Model B | TDC `hERG` | 655 compounds | ROC-AUC >= 0.85, scaffold split or cross validation because the set is small | not trained yet |

**Report the scaffold number, not the random one.** A random split scatters analogue
series across train and test, so the model scores well partly by recognising cores it has
already seen. On Model A that inflated ROC-AUC by 0.11. A Bemis-Murcko scaffold split
keeps every shared ring core on one side, so the test set is chemistry the model has
genuinely never met.

### Features

Each molecule becomes 2058 numbers: 2048 Morgan fingerprint **counts** (radius 2) plus 10
whole-molecule descriptors (weight, logP, TPSA, ring count and so on). Counts beat plain
0/1 bits because six aromatic rings and one do not behave alike in the liver, and the
descriptors capture CYP3A4's preference for large, greasy molecules, which no single
fragment bit expresses. Both choices were picked on a validation split before the test set
was touched; the comparison is in `data/processed/model_a_feature_search.csv`.

The applicability domain runs on plain 0/1 fingerprints instead, because Tanimoto
similarity is only defined for sets, and molecular weight would otherwise swamp it.

### Algorithm

Histogram gradient boosting, not Random Forest. On the scaffold split it beat a forest by
about 0.04 ROC-AUC, which was the difference between missing and meeting the target. It is
also the only one of the two that `shap.TreeExplainer` can explain: on our Random Forest
SHAP fails its own additivity check, while on the boosted model it runs in a fraction of a
second.

### Data

The TDC `.tab` files are downloaded from Harvard Dataverse and read with `pandas`. The
`pytdc` package is deliberately **not** a dependency: it pins numpy<2, pandas<3 and
rdkit<2024.3.1, and on Windows it pulls in `tiledbsoma`, which has no wheel and fails to
build. The download URLs are in notebook 01.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order:

1. `notebooks/01_data_loading.ipynb` downloads both TDC datasets and builds features. Runs end to end.
2. `notebooks/02_model_a_cyp3a4.ipynb` trains and evaluates Model A, and fits the applicability domain. Runs end to end.
3. `notebooks/03_model_b_herg.ipynb` trains and evaluates Model B.
4. `notebooks/04_shap_analysis.ipynb` SHAP, bit to substructure mapping, Fisher test.
5. `notebooks/05_lookup_tables.ipynb` CredibleMeds QT list intersected with the FDA CYP3A4 substrate table.
6. `notebooks/06_herbal_screening.ipynb` COCONUT filtering, applicability domain, tiers.

Demo app:

```bash
streamlit run app/streamlit_app.py
```

Downloaded files go in `data/raw/`. Cleaned files go in `data/processed/`. Both are
gitignored, so nothing large or license restricted ends up in the repo.

## Limitations

These are not footnotes. They decide what the output is allowed to mean.

- **Chaining two imperfect models compounds errors.** Model A and Model B each have their
  own false positive and false negative rate. Putting them in series multiplies the ways a
  flagged pair can be wrong, and the tier label does not repair that.
- **Binary inhibits / does not inhibit ignores strength.** A weak inhibitor and a potent one
  get the same label. Real risk depends on how strongly and at what concentration.
- **Jamu is a multi-compound extract.** "Compound X inhibits CYP3A4" does not mean a jamu
  product containing plant X is dangerous. Dose, how much of the compound is actually in the
  product, absorption, and the rest of the extract all matter and none of them are modelled here.
- **Every output is a hypothesis for lab testing, never a safety claim.** Nothing this project
  produces says a specific herb, product, or combination is safe or unsafe for a person. It
  only says which pairs look worth measuring first.
