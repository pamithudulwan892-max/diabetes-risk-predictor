# trains the diabetes risk model on the CDC health indicators data
# target: Diabetes_binary (1 = prediabetes or diabetes)
# saves model/model.joblib, model/metrics.json and the charts in assets/
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/raw.csv")
TARGET = "Diabetes_binary"
RANDOM_STATE = 42

# things a person can answer about their own health (no income/education/insurance)
FEATURES = {
    "HighBP": "high_blood_pressure",
    "HighChol": "high_cholesterol",
    "CholCheck": "cholesterol_check_5y",
    "BMI": "bmi",
    "Smoker": "smoked_100_cigarettes",
    "Stroke": "had_stroke",
    "HeartDiseaseorAttack": "heart_disease_or_attack",
    "PhysActivity": "physically_active",
    "Fruits": "fruit_daily",
    "Veggies": "veggies_daily",
    "HvyAlcoholConsump": "heavy_drinker",
    "GenHlth": "general_health",        # 1 excellent .. 5 poor
    "MentHlth": "bad_mental_health_days",
    "PhysHlth": "bad_physical_health_days",
    "DiffWalk": "difficulty_walking",
    "Sex": "male",
    "Age": "age_group",                 # 1 = 18-24 .. 13 = 80+
}
OTHER = ["AnyHealthcare", "NoDocbcCost", "Education", "Income"]

Path("model").mkdir(exist_ok=True)
Path("assets").mkdir(exist_ok=True)

raw = pd.read_csv(DATA)
y = raw[TARGET].astype(int)
X = raw[list(FEATURES)].rename(columns=FEATURES)
print("shape:", X.shape, "| positive rate:", round(y.mean(), 4))
print("missing values:", int(X.isna().sum().sum()), "| duplicate rows:", int(raw.drop(columns="ID").duplicated().sum()))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# all features are already numbers (0/1 flags or ordered scales), so just scale them
pre = ColumnTransformer([("num", StandardScaler(), list(X.columns))])
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=20, n_jobs=-1,
                                            random_state=RANDOM_STATE),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = {}
for name, m in models.items():
    s = cross_val_score(Pipeline([("pre", pre), ("model", m)]), X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
    cv_scores[name] = round(float(s.mean()), 4)
    print(f"{name:20s} CV ROC-AUC {s.mean():.4f} +- {s.std():.4f}")

best_name = max(cv_scores, key=cv_scores.get)
best = Pipeline([("pre", pre), ("model", models[best_name])]).fit(X_tr, y_tr)
proba = best.predict_proba(X_te)[:, 1]

# 0.5 is a bad cut-off when only ~14% are positive, so also pick a screening
# threshold on the training data (best F1 on out-of-fold style holdout)
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=1, stratify=y_tr)
tmp = Pipeline([("pre", pre), ("model", models[best_name])]).fit(X_fit, y_fit)
pv = tmp.predict_proba(X_val)[:, 1]
grid = np.arange(0.05, 0.6, 0.01)
f1s = [f1_score(y_val, pv >= t) for t in grid]
threshold = round(float(grid[int(np.argmax(f1s))]), 2)


def at(t):
    p = (proba >= t).astype(int)
    return {"threshold": t, "accuracy": round(accuracy_score(y_te, p), 4),
            "precision": round(precision_score(y_te, p), 4), "recall": round(recall_score(y_te, p), 4),
            "f1": round(f1_score(y_te, p), 4)}


test = {"roc_auc": round(roc_auc_score(y_te, proba), 4),
        "pr_auc": round(average_precision_score(y_te, proba), 4),
        "at_0.5": at(0.5), "at_screening": at(threshold)}
baseline_acc = round(float(1 - y_te.mean()), 4)
print("TEST:", best_name, json.dumps(test))
print("baseline accuracy (always 'no diabetes'):", baseline_acc, "| screening threshold:", threshold)

# experiment: does adding income / education / insurance change much?
X_all = pd.concat([X, raw[OTHER]], axis=1)
pre_all = ColumnTransformer([("num", StandardScaler(), list(X_all.columns))])
s_all = cross_val_score(Pipeline([("pre", pre_all), ("model", models[best_name])]),
                        X_all.loc[X_tr.index], y_tr, cv=cv, scoring="roc_auc")
print("CV ROC-AUC with income/education/insurance added:", round(s_all.mean(), 4))

# charts
ConfusionMatrixDisplay.from_predictions(y_te, (proba >= threshold).astype(int), display_labels=["No", "Yes"],
                                        cmap="Blues", values_format="d")
plt.title(f"Confusion matrix (threshold {threshold})")
plt.tight_layout(); plt.savefig("assets/confusion_matrix.png", dpi=150); plt.close()

RocCurveDisplay.from_predictions(y_te, proba, name=best_name)
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.title("ROC curve on the test set")
plt.tight_layout(); plt.savefig("assets/roc_curve.png", dpi=150); plt.close()

sample = X_te.sample(8000, random_state=RANDOM_STATE)
imp = permutation_importance(best, sample, y_te.loc[sample.index], n_repeats=5, random_state=RANDOM_STATE,
                             scoring="roc_auc")
imp_s = pd.Series(imp.importances_mean, index=X.columns).sort_values()
imp_s.tail(10).plot.barh()
plt.title("Top features (permutation importance, ROC-AUC drop)")
plt.tight_layout(); plt.savefig("assets/feature_importance.png", dpi=150); plt.close()

# defaults and ranges for the app inputs
schema = {c: {"min": float(X[c].min()), "max": float(X[c].max()), "default": float(X[c].median())} for c in X.columns}
coefs = None
if best_name == "Logistic Regression":
    coefs = dict(zip(X.columns, np.round(best.named_steps["model"].coef_[0], 4).tolist()))

joblib.dump(best, "model/model.joblib")
Path("model/metrics.json").write_text(json.dumps({
    "best_model": best_name, "cv_metric": "roc_auc", "cv_scores": cv_scores, "test": test,
    "baseline_accuracy": baseline_acc, "positive_rate": round(float(y.mean()), 4), "threshold": threshold,
    "n_rows": int(len(raw)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    "cv_roc_auc_with_socioeconomic": round(float(s_all.mean()), 4),
    "importance": imp_s.round(4).to_dict(), "coefficients": coefs, "schema": schema,
}, indent=2))
print("saved model/model.joblib and model/metrics.json")
