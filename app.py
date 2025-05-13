
from pydantic import BaseModel
import joblib
import os
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Parameter Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ataraxia-msj.github.io"],  # 开发时可用 ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"

class RecommendRequest(BaseModel):
    input_param: str
    input_value: float

class RecommendResponse(BaseModel):
    recommendations: dict

# 加载所有模型和 scaler
models = {}
for fname in os.listdir(MODEL_DIR):
    if fname.endswith("_model.joblib"):
        name = fname.replace("_model.joblib", "")
        info = joblib.load(os.path.join(MODEL_DIR, fname))
        models[name] = info

# 提取共同 scaler 信息
any_scaler = next(iter(models.values()))['scaler']
numeric_features = list(any_scaler.feature_names_in_)
global_means = dict(zip(numeric_features, any_scaler.mean_))

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    iparam = req.input_param
    ival = req.input_value

    if iparam not in numeric_features:
        raise HTTPException(status_code=400, detail=f"未知参数：{iparam}")

    row = pd.DataFrame([global_means], columns=numeric_features)
    row[iparam] = ival
    scaled = pd.DataFrame(any_scaler.transform(row), columns=numeric_features)

    results = {}
    for target, info in models.items():
        if target == iparam:
            results[target] = ival
            continue
        mdl = info['model']
        feats = info['feature_names']
        y_scaled = mdl.predict(scaled[feats])[0]
        idx = numeric_features.index(target)
        mean_t = any_scaler.mean_[idx]
        std_t = np.sqrt(any_scaler.var_[idx])
        results[target] = float(y_scaled * std_t + mean_t)

    return {"recommendations": results}