# Herb-Drug Cardiac Risk Screening (jamu x prescription drugs)

Computational biology course project.

## Status

Every notebook runs end to end with no errors, and the screen is complete.

| Part | State |
| --- | --- |
| `src/features.py` 2048 Morgan counts + 10 descriptors | implemented |
| `src/models.py` gradient boosting (A) and random forest (B) | implemented |
| `src/applicability.py` domain gate | implemented, tested |
| `src/stats.py` Fisher + Benjamini-Hochberg | implemented, tested (BH checked against SciPy) |
| `src/names.py` drug name matching | implemented, tested |
| `src/combine.py` three conditions and tiers | implemented, tested |
| `src/coconut.py` COCONUT species filter | **run**: 1,861 compounds from 6 plants |
| `src/explain.py` SHAP + bit to substructure | **run**: additivity error 3.6e-13 |
| Notebooks 01-06 | all execute clean |
| `app/streamlit_app.py` | wired to the pipeline |

`python -m pytest -q` runs 33 tests. All pass.

### Results

| Measure | Value | Target |
| --- | --- | --- |
| Model A, random split ROC-AUC | 0.913 | - |
| Model A, **scaffold split** ROC-AUC | **0.802** | 0.80 |
| Model B, scaffold-grouped CV ROC-AUC | **0.869 +/- 0.033** | 0.85 |
| Jamu compounds outside Model A's domain | 45.8% (852 of 1,861) | reported, not hidden |
| Held-out drugs outside the same domain | 4.7% | - |
| Jamu compounds reaching High tier | 117 | - |

Top hits are Hinokinin (0.979, *Phyllanthus niruri*) and rac-Cubebin (0.976,
*Zingiber officinale*), both dibenzylbutyrolactone lignans, and in both cases the
largest positive SHAP contribution falls on the fused aromatic/5-ring oxygenated
system rather than on the periphery.

**The positive controls did not validate the screen.** Of five literature CYP3A4
reference compounds, only two are in COCONUT under the searched name: quercetin
scores 0.106, a miss against its documented in vitro inhibition, and
andrographolide falls outside the applicability domain and is correctly reported
as Unknown. Piperine, bergamottin and curcumin are absent under those names, so
they were never scored. The screen therefore has no working positive control, and
the ranked hits above should be read as untested hypotheses.

### Data, all downloaded automatically

Notebook 01 pulls four files from Harvard Dataverse; COCONUT is the only manual
download. Nothing is hand-typed and nothing needs a registration.

| Dataset | Used for |
| --- | --- |
| `CYP3A4_Veith` | Model A training, condition 1 |
| `CYP3A4_Substrate_CarbonMangels` | condition 2, measured substrate labels |
| `hERG` | Model B training and condition 3 |
| [COCONUT 2.0](https://coconut.naturalproducts.net/download) | the jamu compounds (manual, ~240 MB zip into `data/raw/`) |

The FDA substrate table and the CredibleMeds QTdrugs list were the original plan
for conditions 2 and 3. Neither is usable as a dataset: the FDA page is HTML whose
CYP3A row lists only midazolam and triazolam, which are *index* substrates for
designing interaction studies rather than a substrate list, and CredibleMeds puts
bulk download behind a commercial licence. Notebook 05 explains the substitution
and what it costs: the hERG label is in vitro channel blockade, not clinical
torsades risk, so the output is a mechanistic hypothesis list rather than a
clinical risk list.

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

The two model probabilities are never multiplied together. Random Forest outputs are
not calibrated, so their product has no meaning. The screen reports a tier instead:

| Tier | Meaning |
| --- | --- |
| High | All three conditions hold, and both model legs are confident. |
| Medium | Two of three hold, or at least one leg is low confidence. |
| Unknown | The herb compound falls outside the training applicability domain, so the model has no basis for an answer. |

A continuous score exists only to rank pairs inside a single tier. It never moves a pair
between tiers. The score is the **weaker of the two model legs**, so a pair ranks no
higher than its weakest evidence.

### How "borderline" is counted

A model leg is a firm yes at `p >= 0.70`, a firm no at `p <= 0.30`, and borderline
in between. A borderline leg counts as a **weak hold**, not as a failure.

That distinction is what makes the Medium tier's second case reachable at all.
The yes band and the confidence band use the same 0.70 cut-off, so if borderline
counted as a failure then "all three hold but one leg is borderline" could never
occur, and every such pair would silently fall through to the two-of-three case.

The lookup leg has no weak form. The FDA table either lists a drug as a CYP3A4
substrate or it does not, and `not in table` is a gap in the source rather than a
weak yes, so it never counts as a hold.

## Models and targets

| Model | Dataset | Size | Target |
| --- | --- | --- | --- |
| Model A | TDC `CYP3A4_Veith` | about 12k compounds | ROC-AUC >= 0.80 |
| Model B | TDC `hERG` | a few hundred compounds | ROC-AUC >= 0.85, scaffold split or cross validation because the set is small |

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order:

1. `notebooks/01_data_loading.ipynb` downloads the three TDC datasets and builds fingerprints.
2. `notebooks/02_model_a_cyp3a4.ipynb` trains Model A, runs a random **and** a scaffold split, fits the applicability domain.
3. `notebooks/03_model_b_herg.ipynb` cross validates Model B over 5 folds, plus a scaffold split, then fits the final model.
4. `notebooks/04_shap_analysis.ipynb` SHAP, bit to substructure mapping, Fisher test with BH correction, known-pharmacology check.
5. `notebooks/05_lookup_tables.ipynb` builds the 342-drug panel: measured CYP3A4 substrates, with condition 3 from measured hERG blockade where it exists and Model B elsewhere.
6. `notebooks/06_herbal_screening.ipynb` COCONUT filtering, applicability domain, tiers.

Notebooks 02 and 03 must run before 04, 06 and the app, because they write the
model and domain files everything else loads.

Tests:

```bash
python -m pytest -q
```

Demo app:

```bash
streamlit run app/streamlit_app.py
```

It reports which notebook to run if an artefact is missing, rather than
substituting a placeholder number.

Downloaded files go in `data/raw/`. Cleaned files go in `data/processed/`. Both are
gitignored, so nothing large or license restricted ends up in the repo.

### If RDKit will not import on Windows

```
ImportError: DLL load failed while importing rdBase:
An Application Control policy has blocked this file.
```

This is Windows **Smart App Control** blocking RDKit's unsigned compiled
libraries. Reinstalling does not help, because the replacement binaries are
blocked too. Check it with:

```powershell
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' |
  Select-Object VerifiedAndReputablePolicyState   # 1 = on, 0 = off, 2 = evaluation
```

Two ways round it:

- **Google Colab**, which the build plan already allows for. Nothing in this repo
  depends on running locally, and the CYP3A4 training is faster there anyway.
- **Turn Smart App Control off**, in Windows Security -> App & browser control.
  Note that this is one way: re-enabling it requires resetting Windows, so it is
  a real decision rather than a toggle.

Everything downstream of a fingerprint is pure numpy, scipy and sklearn, so the
test suite and the COCONUT filtering run with Smart App Control left on.

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
