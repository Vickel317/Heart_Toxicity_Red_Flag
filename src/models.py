# Train and score the two Random Forest models.
# Model A is CYP3A4 inhibition, Model B is hERG block. Same code, different data.

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def train_model(X, y):
    # The default trainer for both models: histogram gradient boosting.
    #
    # Trees are built one after another, each one concentrating on the molecules
    # the previous trees got wrong, instead of 500 independent trees voting at once.
    # On a scaffold split of CYP3A4_Veith this beat a Random Forest by about 0.04
    # ROC-AUC, which was the difference between missing and meeting the 0.80 target.
    #
    # It is also the better choice for notebook 04: shap.TreeExplainer runs on it
    # in a fraction of a second, while on our Random Forest it fails its own
    # additivity check outright.
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, random_state=42
    )
    model.fit(X, y)
    return model


def train_rf(X, y):
    # Kept as the comparison baseline, and because Model B trains on only a few
    # hundred hERG compounds, where a forest may well hold up better than boosting.
    # Notebook 03 decides that on the data rather than by assumption.
    model = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42
    )
    model.fit(X, y)
    return model


def scaffold_split(smiles_list, test_size=0.2):
    # Split by Bemis-Murcko scaffold: strip every side chain off a molecule and
    # keep the ring core. Molecules sharing a core always land on the same side.
    #
    # Why bother: a random split can put ibuprofen in train and a close analogue
    # of ibuprofen in test. The model then looks clever when it is really just
    # recognising something it has already seen. A scaffold split forbids that,
    # so the score it reports is the one that survives contact with new chemistry.
    # This is the split TDC itself recommends for CYP3A4_Veith.
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups = {}
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        groups.setdefault(core, []).append(i)

    # Biggest scaffold families fill the training set first, so the test set ends
    # up holding the rarer cores. That is deliberately the harder direction.
    ordered = sorted(groups.values(), key=len, reverse=True)
    n_train_target = len(smiles_list) * (1 - test_size)

    train_idx, test_idx = [], []
    for group in ordered:
        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
        else:
            test_idx.extend(group)

    return sorted(train_idx), sorted(test_idx)




def get_auc(model, X, y):
    # ROC-AUC on the positive class probability.
    # Targets: Model A (CYP3A4_Veith, about 12k compounds) >= 0.80.
    # Model B (hERG, a few hundred compounds) >= 0.85, measured with a scaffold
    # split or cross validation because a single random split of a small set is noisy.
    probs = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, probs)
