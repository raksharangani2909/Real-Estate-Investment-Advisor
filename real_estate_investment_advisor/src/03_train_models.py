"""STEP 3 - Train, evaluate and track models with MLflow.

Run:  python src/03_train_models.py
Then: mlflow ui --backend-store-uri sqlite:///mlflow.db     (open http://127.0.0.1:5000)

Classification  : Good_Investment            (LogReg, Random Forest, XGBoost)
Regression      : Future_Price_5Y_Lakhs      (Ridge, Random Forest, XGBoost)
The best model of each task (by F1 / RMSE on the held-out test set) is saved to models/.
"""
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

import config as C

warnings.filterwarnings("ignore")
BLUE, ORANGE, SURF = "#2a78d6", "#eb6834", "#fcfcfb"


def make_preprocessor():
    """Scale numeric features, one-hot encode categoricals."""
    return ColumnTransformer([
        ("num", StandardScaler(), C.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CATEGORICAL_FEATURES),
    ])


def importance_table(pipe) -> pd.DataFrame:
    """Feature importance per ORIGINAL feature (one-hot columns are summed back)."""
    model = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        imp = np.abs(np.ravel(model.coef_))
        imp = imp / imp.sum()
    rows = {}
    for n, v in zip(names, imp):
        kind, rest = n.split("__", 1)
        key = rest
        if kind == "cat":
            key = next(c for c in C.CATEGORICAL_FEATURES if rest.startswith(c + "_"))
        rows[key] = rows.get(key, 0) + float(v)
    return (pd.DataFrame({"feature": list(rows), "importance": list(rows.values())})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def plot_importance(tbl, title, path, color=BLUE):
    t = tbl.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.barh(t.feature, t.importance, color=color, height=0.7)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Relative importance")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    t0 = time.time()
    C.MODELS_DIR.mkdir(exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.CLEAN_CSV)
    X = df[C.FEATURES]
    y_clf, y_reg = df[C.CLF_TARGET], df[C.REG_TARGET]

    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_clf, y_reg, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y_clf)
    print(f"train={len(X_tr):,} test={len(X_te):,} | good-investment share={y_clf.mean():.3f}")

    mlflow.set_tracking_uri(f"sqlite:///{C.MLFLOW_DB}")
    for name in ("RealEstate_Classification", "RealEstate_Regression"):
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=C.MLFLOW_ARTIFACTS.as_uri() + "/" + name)

    # ------------------------------------------------------------------ CLASSIFICATION
    clf_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=C.RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5, max_features=0.5,
                                               n_jobs=-1, random_state=C.RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                                 tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE, eval_metric="logloss"),
    }
    mlflow.set_experiment("RealEstate_Classification")
    clf_rows, clf_fitted = [], {}

    # majority-class baseline (so 'accuracy' can be put in context)
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, yc_tr)
    base_acc = accuracy_score(yc_te, base.predict(X_te))

    for name, model in clf_models.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
            t = time.time()
            pipe.fit(X_tr, yc_tr)
            fit_s = time.time() - t
            pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
            m = {"accuracy": accuracy_score(yc_te, pred), "precision": precision_score(yc_te, pred),
                 "recall": recall_score(yc_te, pred), "f1": f1_score(yc_te, pred),
                 "roc_auc": roc_auc_score(yc_te, proba), "train_seconds": fit_s,
                 "baseline_majority_accuracy": base_acc}
            mlflow.log_params({"model": name, "n_train": len(X_tr), "n_features_raw": len(C.FEATURES),
                               "growth_threshold_pct": C.GOOD_GROWTH_THRESHOLD, **{f"hp_{k}": v for k, v in model.get_params().items()
                                                                                    if isinstance(v, (int, float, str, bool)) and v is not None}})
            mlflow.log_metrics(m)
            tbl = importance_table(pipe)
            tbl.to_csv(C.MODELS_DIR / f"_tmp_imp_clf_{name}.csv", index=False)
            mlflow.log_artifact(str(C.MODELS_DIR / f"_tmp_imp_clf_{name}.csv"), "feature_importance")
            (C.MODELS_DIR / f"_tmp_imp_clf_{name}.csv").unlink()
            sig = infer_signature(X_te.head(50), pipe.predict(X_te.head(50)))
            mlflow.sklearn.log_model(pipe, name="model", signature=sig, serialization_format="cloudpickle")
            clf_rows.append({"model": name, **m})
            clf_fitted[name] = (pipe, pred, proba)
            print(f"[clf] {name:18s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} ({fit_s:.0f}s)")

    clf_df = pd.DataFrame(clf_rows).sort_values("f1", ascending=False)
    best_clf = clf_df.iloc[0]["model"]
    pipe_c, pred_c, proba_c = clf_fitted[best_clf]
    with mlflow.start_run(run_name=f"BEST_{best_clf}"):
        mlflow.log_params({"selected_model": best_clf, "selection_metric": "f1"})
        mlflow.log_metrics({k: float(v) for k, v in clf_df.iloc[0].items() if k != "model"})
        mlflow.sklearn.log_model(pipe_c, name="model", registered_model_name="real_estate_investment_classifier", serialization_format="cloudpickle",
                                 signature=infer_signature(X_te.head(50), pipe_c.predict(X_te.head(50))))

    # ------------------------------------------------------------------ REGRESSION
    reg_models = {
        "Ridge": Ridge(alpha=1.0, random_state=C.RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=80, max_depth=14, min_samples_leaf=5, max_features=0.5,
                                              n_jobs=-1, random_state=C.RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                                tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE),
    }
    mlflow.set_experiment("RealEstate_Regression")
    reg_rows, reg_fitted = [], {}
    naive = X_te.Price_in_Lakhs.values       # baseline: "price in 5 years = price today"
    naive_rmse = float(np.sqrt(mean_squared_error(yr_te, naive)))

    def reg_metrics(y, p):
        return {"rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)),
                "r2": float(r2_score(y, p)), "mape_pct": float(np.mean(np.abs((y - p) / y)) * 100)}

    for name, model in reg_models.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
            t = time.time()
            pipe.fit(X_tr, yr_tr)
            fit_s = time.time() - t
            pred = pipe.predict(X_te)
            m = {**reg_metrics(yr_te.values, pred), "train_seconds": fit_s, "baseline_naive_rmse": naive_rmse}
            mlflow.log_params({"model": name, "n_train": len(X_tr), "n_features_raw": len(C.FEATURES),
                               "forecast_years": C.FORECAST_YEARS,
                               **{f"hp_{k}": v for k, v in model.get_params().items()
                                  if isinstance(v, (int, float, str, bool)) and v is not None}})
            mlflow.log_metrics(m)
            tbl = importance_table(pipe)
            tp = C.MODELS_DIR / f"_tmp_imp_reg_{name}.csv"
            tbl.to_csv(tp, index=False)
            mlflow.log_artifact(str(tp), "feature_importance")
            tp.unlink()
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle",
                                     signature=infer_signature(X_te.head(50), pipe.predict(X_te.head(50))))
            reg_rows.append({"model": name, **m})
            reg_fitted[name] = (pipe, pred)
            print(f"[reg] {name:18s} rmse={m['rmse']:.3f} mae={m['mae']:.3f} r2={m['r2']:.4f} ({fit_s:.0f}s)")

    reg_df = pd.DataFrame(reg_rows).sort_values("rmse")
    best_reg = reg_df.iloc[0]["model"]
    pipe_r, pred_r = reg_fitted[best_reg]
    with mlflow.start_run(run_name=f"BEST_{best_reg}"):
        mlflow.log_params({"selected_model": best_reg, "selection_metric": "rmse"})
        mlflow.log_metrics({k: float(v) for k, v in reg_df.iloc[0].items() if k != "model"})
        mlflow.sklearn.log_model(pipe_r, name="model", registered_model_name="real_estate_price_forecaster", serialization_format="cloudpickle",
                                 signature=infer_signature(X_te.head(50), pipe_r.predict(X_te.head(50))))

    # ------------------------------------------------------------------ SAVE + REPORT
    joblib.dump(pipe_c, C.MODELS_DIR / "classifier.joblib", compress=3)
    joblib.dump(pipe_r, C.MODELS_DIR / "regressor.joblib", compress=3)
    ic, ir = importance_table(pipe_c), importance_table(pipe_r)
    ic.to_csv(C.MODELS_DIR / "feature_importance_classifier.csv", index=False)
    ir.to_csv(C.MODELS_DIR / "feature_importance_regressor.csv", index=False)
    clf_df.to_csv(C.REPORTS_DIR / "model_comparison_classification.csv", index=False)
    reg_df.to_csv(C.REPORTS_DIR / "model_comparison_regression.csv", index=False)

    plot_importance(ic, f"Classifier ({best_clf}) - top 15 features", C.FIG_DIR / "model_feature_importance_classifier.png")
    plot_importance(ir, f"Regressor ({best_reg}) - top 15 features", C.FIG_DIR / "model_feature_importance_regressor.png", ORANGE)

    cm = confusion_matrix(yc_te, pred_c)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor=SURF)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"],
           xlabel="Predicted", ylabel="Actual")
    ax.set_title(f"Confusion matrix - {best_clf}", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "model_confusion_matrix.png", dpi=130)
    plt.close(fig)

    idx = np.random.RandomState(0).choice(len(yr_te), 5000, replace=False)
    fig, ax = plt.subplots(figsize=(5.2, 5), facecolor=SURF)
    ax.scatter(yr_te.values[idx], pred_r[idx], s=4, alpha=.35, color=BLUE, linewidths=0)
    lim = [yr_te.min(), yr_te.max()]
    ax.plot(lim, lim, color=ORANGE, lw=1.5)
    ax.set(xlabel="Actual 5-year price (lakhs)", ylabel="Predicted (lakhs)")
    ax.set_title(f"Regression - {best_reg}", loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "model_actual_vs_predicted.png", dpi=130)
    plt.close(fig)

    summary = {
        "classification": {"best_model": best_clf, "baseline_majority_accuracy": base_acc,
                           "results": clf_df.round(4).to_dict("records")},
        "regression": {"best_model": best_reg, "baseline_naive_rmse_lakhs": naive_rmse,
                       "results": reg_df.round(4).to_dict("records")},
        "split": {"train": len(X_tr), "test": len(X_te), "random_state": C.RANDOM_STATE},
    }
    (C.REPORTS_DIR / "model_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest classifier: {best_clf} | Best regressor: {best_reg} | total {time.time() - t0:.0f}s")
    print(clf_df.round(4).to_string(index=False))
    print(reg_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
