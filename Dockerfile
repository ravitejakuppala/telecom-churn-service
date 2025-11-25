# syntax=docker/dockerfile:1
FROM python:3.11-slim

# install build deps and runtime libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential gfortran libopenblas-dev liblapack-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# create app dir
WORKDIR /app
COPY pyproject.toml requirements.txt ./
# upgrade pip & install dependencies
RUN python -m pip install --upgrade pip setuptools wheel
# if you want to ensure numpy is present before building scipy:
RUN python -m pip install numpy
RUN python -m pip install -r requirements.txt



COPY app.py app.py
COPY models models
ENV MODEL_PATH=models/churn_pipeline.joblib

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
