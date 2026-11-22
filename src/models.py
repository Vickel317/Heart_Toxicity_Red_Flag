# Train the two models and score them.
#
# Model A answers "does this molecule jam CYP3A4?" (the liver enzyme that clears
# about half of all prescription drugs). Model B answers "does this molecule
# block hERG?" (a channel the heart needs to reset between beats). Same code
# here, different data.
#
# The two use DIFFERENT algorithms on purpose, because they are not the same
# size of problem:
#
#   Model A  12,328 molecules  -> gradient boosting wins
#   Model B     655 molecules  -> random forest wins
#
# Why that flip happens: boosting builds trees one at a time, each one trying to
# fix the previous one's mistakes. That only helps if there is enough data to
# tell a real mistake from random noise. With a few hundred molecules it starts
# memorising the noise, and a crowd of independent trees voting is steadier.
#
# Neither of these was assumed. Notebooks 02 and 03 run both algorithms on the
# same data and print the comparison.

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def train_model(X, y):
    # Gradient boosting. This is what Model A uses.
    #
    # Trees are built in sequence, each new one paying most attention to the
    # molecules the earlier trees got wrong. Compare a random forest, which
    # grows 500 trees independently and lets them vote.
    #
    # Two reasons this is Model A's trainer:
    #   1. It scores better: 0.802 against the forest's 0.787 on a scaffold
    #      split (the honest kind of split, explained below).
    #   2. SHAP can actually explain it. Working out why one molecule got its
    #      score takes about 0.01 seconds here, against roughly 7 seconds on a
    #      500 tree forest of this size, and on that forest SHAP also fails its
    #      own self-check. Notebook 04 only works with a model SHAP can handle.
    #
    # No class_weight setting: this algorithm does not accept one, and it is not
    # needed. The CYP3A4 data is 41% "jams it", which is close enough to even.
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, random_state=42
    )
    model.fit(X, y)
    return model


def train_rf(X, y):
    # Random forest. This is what Model B uses, and the thing Model A is
    # compared against.
    #
    # class_weight="balanced" because the hERG data is 69% blockers. Without it
    # a model could score well by answering "blocker" for nearly everything.
    # Balancing makes mistakes on the rarer answer count for more.
    model = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=42
    )
    model.fit(X, y)
    return model


def get_auc(model, X, y):
    # ROC-AUC: the chance the model scores a real positive above a real
    # negative. 0.5 is coin-flipping, 1.0 is perfect.
    #
    # Targets from the proposal: Model A >= 0.80, Model B >= 0.85. Model B's is
    # measured by cross validation rather than one split, because one split of
    # 655 molecules leaves a test set small enough that the score moves around
    # depending on which molecules happened to land in it.
    probs = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, probs)


def scaffold_split(smiles_list, test_size=0.2):
    # Split the data so that similar molecules cannot sit on both sides.
    #
    # How: strip every side chain off a molecule and keep the ring skeleton that
    # is left (its "Bemis-Murcko scaffold"). Molecules with the same skeleton
    # always go to the same side.
    #
    # Why bother, instead of splitting at random? Chemistry datasets are full of
    # families: twenty variations on one skeleton, differing by a methyl group
    # here, a chlorine there. Split at random and those get scattered across
    # both sides, so the model can score well by recognising a skeleton it
    # already memorised. It looks like it learned chemistry; it learned names.
    #
    # On our CYP3A4 data this inflated the score by 0.11 (0.913 random vs 0.802
    # scaffold). The scaffold number is the one that holds up on new chemistry,
    # so it is the one to report, even though it looks worse.
    #
    # Returns (train positions, test positions) into smiles_list.
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups = {}
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        groups.setdefault(core, []).append(i)

    # Biggest families go into training first, so the test set is left holding
    # the rare, odd skeletons. That is the harder direction on purpose, and the
    # realistic one: a jamu compound we have never seen is far more likely to be
    # rare than common. Filling the test set first instead would make the score
    # look better and mean less.
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
    # One skeleton label per molecule, for cross validation.
    #
    # Model B has only 655 molecules, so a single scaffold split leaves about
    # 130 to test on and the score swings depending on the random seed. Handing
    # these labels to StratifiedGroupKFold keeps the no-similar-molecules-on-
    # both-sides rule while averaging over five folds instead of trusting one.
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
