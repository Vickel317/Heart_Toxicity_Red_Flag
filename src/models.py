# Train and score the two models.
# Model A is CYP3A4 inhibition, Model B is hERG block. Same code, different data.
#
# The two models deliberately use DIFFERENT algorithms, because they are not the
# same problem:
#
#   Model A  12,328 compounds  -> gradient boosting wins
#   Model B     655 compounds  -> random forest wins
#
# That is the normal pattern. Boosting corrects its own mistakes round after
# round, which needs enough data to tell a real mistake from noise. With a few
# hundred molecules it starts fitting the noise, and a forest of independent
# trees voting is steadier. Both were measured rather than assumed; notebooks 02
# and 03 show the comparison.

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def train_model(X, y):
    # Gradient boosting. The default trainer for Model A.
    #
    # Trees are built one after another, each new tree concentrating on the
    # molecules the previous trees got wrong, instead of 500 independent trees
    # voting at once.
    #
    # Two reasons this is Model A's trainer:
    #   1. On a scaffold split of CYP3A4_Veith it beat a random forest by about
    #      0.04 ROC-AUC (0.802 against 0.787).
    #   2. SHAP can actually explain it. shap.TreeExplainer needs about 0.01
    #      seconds per molecule here, against roughly 7 seconds per molecule on
    #      a 500 tree forest grown on 12k compounds, and on that forest it also
    #      fails its own additivity check. Notebook 04 is only practical with a
    #      model SHAP can handle.
    #
    # No class_weight argument: HistGradientBoostingClassifier does not take one.
    # It is not needed here either, because CYP3A4_Veith is 41 percent positive,
    # which is close enough to balanced.
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, random_state=42
    )
    model.fit(X, y)
    return model


def train_rf(X, y):
    # Random forest. The default trainer for Model B, and the baseline for A.
    #
    # class_weight balanced because the hERG set is 69 percent blockers, so an
    # unweighted model could score well by leaning towards "blocker" for
    # everything. Balancing makes a mistake on the rarer class count for more.
    model = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42
    )
    model.fit(X, y)
    return model


def get_auc(model, X, y):
    # ROC-AUC on the positive class probability.
    # Targets: Model A (CYP3A4_Veith, about 12k compounds) >= 0.80.
    # Model B (hERG, a few hundred compounds) >= 0.85, measured with a scaffold
    # split or cross validation because a single random split of a small set is noisy.
    probs = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, probs)


def scaffold_split(smiles_list, test_size=0.2):
    # Split by Bemis-Murcko scaffold: strip every side chain off a molecule and
    # keep the ring core that remains. Molecules sharing a core always land on
    # the same side of the split.
    #
    # Why this matters more than it sounds. Public chemistry datasets are full of
    # analogue series: twenty variations on one core, differing by a methyl group
    # here or a chlorine there. A random split scatters those across train and
    # test, so the model can score well just by recognising a core it already
    # memorised. On CYP3A4_Veith that inflated ROC-AUC by 0.11 (0.913 against
    # 0.802). The scaffold number is the one that survives contact with new
    # chemistry, and therefore the one to report.
    #
    # Returns (train indices, test indices) into smiles_list.
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups = {}
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        groups.setdefault(core, []).append(i)

    # Largest scaffold families fill the training set first, so the test set ends
    # up holding the rare, awkward cores. That is deliberately the harder
    # direction: it is also the realistic one, because a herb compound we have
    # never seen is far more likely to be rare than common.
    ordered = sorted(groups.values(), key=len, reverse=True)
    n_train_target = len(smiles_list) * (1 - test_size)

    train_idx, test_idx = [], []
    for group in ordered:
        if len(train_idx) + len(group) <= n_train_target:
            train_idx.extend(group)
        else:
            test_idx.extend(group)

    return sorted(train_idx), sorted(test_idx)


def scaffold_groups(smiles_list):
    # Scaffold label per molecule, for grouped cross validation.
    # Model B has only 655 compounds, so one scaffold split leaves a test set of
    # about 130 molecules and the score swings with the seed. Passing these
    # labels to StratifiedGroupKFold keeps the scaffold honesty while averaging
    # over five folds instead of trusting one.
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    labels = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            labels.append("")
            continue
        labels.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
    return labels
