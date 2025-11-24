# Telecom Churn Inference Service (Custom)

This microservice trains a churn model on your telecom dataset and serves predictions via FastAPI.

## Dataset
- Expected CSV (default): `data/telecom_churn_data.csv`
- Label engineering:
  - `active_8 = (total_ic_mou_8 + total_og_mou_8 + vol_2g_mb_8 + vol_3g_mb_8) > 0`
  - `inactive_9 = (total_ic_mou_9 + total_og_mou_9 + vol_2g_mb_9 + vol_3g_mb_9) == 0`
  - `churn = active_8 & inactive_9`

## Features
- Numeric columns ending with `_6`, `_7`, `_8`, plus `aon` and `circle_id` if present
- Date/id columns are excluded automatically
- Missing values imputed (median), scaled, Logistic Regression classifier

## Quick Start (Local)

```bash
python -m venv .venv && .venv\Scripts\activate  # on Windows
pip install -r requirements.txt

# Place your CSV at data/telecom_churn_data.csv (or set CSV_PATH)
python train_model.py

# Start API (loads models/churn_pipeline.joblib)
uvicorn app:app --host 127.0.0.1 --port 8000

# Docs
http://127.0.0.1:8000/docs
```

## Endpoints
- `GET /status` → {"status":"ok","model_loaded":true}
- `GET /version` → service & model path
- `POST /predict` → `{"instances":[{...feature columns...}]}`

### Example request
`sample_request.json` is included with one record extracted from your CSV using the features above.

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @sample_request.json
```

## Deploy (Render)
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Env: `MODEL_PATH=models/churn_pipeline.joblib`
- Health check path: `/healthz`

## Notes for Graders
- Visit `/docs`, use `POST /predict` with the included `sample_request.json`.
