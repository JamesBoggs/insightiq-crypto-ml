import os, mlflow, mlflow.pyfunc, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

FEATURES = [
 'ret_1d','vol_3d','vol_7d','vol_14d','vol_30d','ret_3d','ret_7d','ret_14d','ret_30d',
 'ma_3d','ma_7d','ma_14d','ma_30d',
 'comp_mean','pos_mean','neg_mean','neu_mean','n_posts'
]

class Payload(BaseModel):
    features: list

app = FastAPI(title="MemeCoin Alpha API")

@app.get("/health")
def health():
    return {"ok": True}

@app.on_event("startup")
def load_model():
    global model
    name = os.environ.get("MODEL_NAME","MemeCoinAlpha")
    version = os.environ.get("MODEL_VERSION","1")
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")

@app.post("/predict")
def predict(p: Payload):
    X = pd.DataFrame(p.features, columns=FEATURES)
    probs = model.predict(X).tolist()
    return {"probs": probs, "classes":[0,1]}
