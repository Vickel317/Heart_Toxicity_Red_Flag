# Train and score the two Random Forest models.
# Model A is CYP3A4 inhibition, Model B is hERG block. Same code, different data.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def train_rf(X, y):
    # class_weight balanced because both label sets are uneven.
    # Random Forest also gives us SHAP TreeExplainer support for free later on.
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
