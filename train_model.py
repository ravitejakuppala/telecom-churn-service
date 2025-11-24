"""
Train a churn model on the telecom dataset with automatic label engineering.

Label (churn):
- active_8 = (total_ic_mou_8 + total_og_mou_8 + vol_2g_mb_8 + vol_3g_mb_8) > 0
- inactive_9 = (total_ic_mou_9 + total_og_mou_9 + vol_2g_mb_9 + vol_3g_mb_9) == 0
- churn = active_8 & inactive_9  (1 if true, else 0)

Features:
- numeric columns from months 6–8 (ending with _6, _7, _8), plus 'aon' and 'circle_id' if present
- date/id columns dropped automatically

"""
import os, joblib, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

CSV_PATH = os.getenv("CSV_PATH", "telecom_churn_data.csv")
MODEL_OUT = os.getenv("MODEL_OUT", "models/churn_pipeline.joblib")

def load_df():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    # fallback for your uploaded path when testing locally
    alt = "/mnt/data/telecom_churn_data.csv"
    if os.path.exists(alt):
        return pd.read_csv(alt)
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

def engineer_label(df: pd.DataFrame) -> pd.Series:
    # usage in month 8 and 9
    def safe_sum(cols):
        return df[cols].sum(axis=1, min_count=1)

    m8_cols = [c for c in df.columns if c.endswith("_8") and any(tok in c for tok in ["total_ic_mou","total_og_mou","vol_2g_mb","vol_3g_mb"])]
    m9_cols = [c for c in df.columns if c.endswith("_9") and any(tok in c for tok in ["total_ic_mou","total_og_mou","vol_2g_mb","vol_3g_mb"])]
    if not m8_cols or not m9_cols:
        raise ValueError("Expected month 8/9 usage columns not found. Check column names in your CSV.")

    active_8 = safe_sum(m8_cols).fillna(0) > 0
    inactive_9 = safe_sum(m9_cols).fillna(0) == 0
    churn = (active_8 & inactive_9).astype(int)
    return churn

def pick_features(df: pd.DataFrame):
    bad = ["last_date", "date", "mobile_number"]
    candidates = [c for c in df.columns if all(s not in c.lower() for s in bad)]
    num = df[candidates].select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num if c.endswith(("_6","_7","_8"))]
    for extra in ["aon","circle_id"]:
        if extra in df.columns and extra not in feats:
            feats.append(extra)
    # drop columns that are >90% missing
    na_ratio = df[feats].isna().mean()
    feats = [c for c in feats if na_ratio.get(c,0.0) <= 0.9]
    return feats

def build_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=300, n_jobs=None))
    ])

def main():
    warnings.filterwarnings("ignore")
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    df = load_df()
    y = engineer_label(df)
    X_cols = pick_features(df)
    X = df[X_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=100)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # Evaluate
    try:
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
    except Exception:
        auc = None
    preds = pipe.predict(X_test)
    print("Model trained on", len(X_cols), "features.")
    if auc is not None:
        print("Holdout ROC AUC:", round(auc, 4))
    print(classification_report(y_test, preds, digits=3))

    # Save pipeline with feature names embedded
    # We'll attach attribute so the API can confirm columns if needed
    # pipe.feature_names_in_ = np.array(X_cols, dtype=object)
    joblib.dump(pipe, MODEL_OUT)
    print("Model saved to:", MODEL_OUT)

if __name__ == "__main__":
    main()
