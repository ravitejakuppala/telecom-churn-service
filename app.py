from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os, joblib
import pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_pipeline.joblib")
SERVICE_VERSION = "1.1.0"

# Load model at startup
_pipeline = None
if os.path.exists(MODEL_PATH):
    _pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Telecom Customer Churn Prediction Service",
    version=SERVICE_VERSION,
    description=(
        "This FastAPI microservice predicts the likelihood of customer churn "
        "based on telecom usage data from months 6 to 8. "
        "The API provides endpoints to check service status, retrieve model information, "
        "and generate churn predictions for one or more customers."
    )
)


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return PlainTextResponse("", status_code=204)

@app.get("/status")
def status():
    return {"status":"ok","model_loaded": _pipeline is not None}

@app.get("/version")
def version():
    return {"service_version": SERVICE_VERSION, "model_path": MODEL_PATH}


@app.post("/predict")
def predict(req: PredictRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train and place joblib at models/ or set MODEL_PATH.")

    try:
        import numpy as np
        df = pd.DataFrame(req.instances)

        # Try to align incoming columns to what the model saw at fit time
        expected = None
        if hasattr(_pipeline, "feature_names_in_"):
            expected = list(_pipeline.feature_names_in_)
        # Some sklearn versions set it on the final estimator instead:
        elif hasattr(getattr(_pipeline, "named_steps", {}), "clf") and hasattr(_pipeline.named_steps["clf"], "feature_names_in_"):
            expected = list(_pipeline.named_steps["clf"].feature_names_in_)

        if expected is not None:
            # Reindex to expected order and add any missing columns as 0
            df = df.reindex(columns=expected, fill_value=0)

        # Predict
        if hasattr(_pipeline, "predict_proba"):
            proba = _pipeline.predict_proba(df)[:, 1]
            preds = (proba >= 0.5).astype(int)
        else:
            preds = _pipeline.predict(df)
            # Best-effort probability
            try:
                from scipy.special import expit
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(_pipeline)
                scores = getattr(_pipeline, "decision_function", lambda X: preds)(df)
                proba = expit(scores)
            except Exception:
                proba = np.array(preds, dtype=float)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    out = [{"prediction": int(p), "proba": float(pb)} for p, pb in zip(preds, proba)]
    return {"predictions": out}
