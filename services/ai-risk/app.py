from pathlib import Path
from typing import Literal, Optional, List

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conlist

# Base paths (réutilise les artefacts générés par risk_model_oulad.py)
# Repo root : services/ai-risk/app.py -> parents[2] == repo root
BASE_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = BASE_DIR / "datasets" / "oulad"
RISK_DIR = DATASETS_DIR / "models" / "risk"
RISK_RNN_DIR = DATASETS_DIR / "models" / "risk_rnn"
RISK_LSTM_DIR = DATASETS_DIR / "models" / "risk_lstm"

FEATURE_ORDER = [
    "sum_click_total",
    "n_assessments",
    "eng_clicks_per_day",
    "assess_per_10days",
    "studied_credits",
    "num_of_prev_attempts",
]


# ===============================
#      Chargement des modèles
# ===============================

class RiskMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            lin = torch.nn.Linear(dims[i], dims[i + 1])
            lin = torch.nn.utils.weight_norm(lin)
            layers += [lin, torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        out = torch.nn.Linear(dims[-1], 1)
        out = torch.nn.utils.weight_norm(out)
        layers.append(out)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_mlp():
    scaler_path = RISK_DIR / "scaler.pkl"
    model_path = RISK_DIR / "risk_model.pt"
    if not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError("Scaler ou modèle MLP introuvable. Lance risk_model_oulad.py d'abord.")
    scaler = joblib.load(scaler_path)
    model = RiskMLP(input_dim=len(FEATURE_ORDER))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return scaler, model


MLP_SCALER, MLP_MODEL = load_mlp()

# Placeholders pour RNN/LSTM (chargés si présents)
RNN_MODEL = None
RNN_SCALER = None
LSTM_MODEL = None
LSTM_SCALER = None


# ===============================
#      Schémas d'entrée
# ===============================

class RiskFeatures(BaseModel):
    sum_click_total: float
    n_assessments: float
    eng_clicks_per_day: float
    assess_per_10days: float
    studied_credits: float
    num_of_prev_attempts: float


class RiskSeqPayload(BaseModel):
    # Liste de pas de temps, chaque pas = liste de features (2D)
    sequence: conlist(conlist(float, min_length=1), min_length=1) = Field(
        ..., description="Séquence temporelle: liste de pas de temps, chaque pas = liste de features"
    )


class PredictResponse(BaseModel):
    risk_score: float
    prediction: int
    model_type: Literal["mlp", "rnn", "lstm"]


app = FastAPI(title="EduPath AI Risk Service", version="1.0.0")


# ===============================
#          Helpers
# ===============================

def predict_mlp(payload: RiskFeatures) -> PredictResponse:
    feats = np.array([[getattr(payload, k) for k in FEATURE_ORDER]], dtype=np.float32)
    feats = MLP_SCALER.transform(feats)
    with torch.no_grad():
        logits = MLP_MODEL(torch.tensor(feats))
        prob = torch.sigmoid(logits).item()
    pred = 1 if prob >= 0.5 else 0
    return PredictResponse(risk_score=prob, prediction=pred, model_type="mlp")


# ===============================
#          Endpoints
# ===============================

@app.post("/predict-risk", response_model=PredictResponse)
def predict_risk(payload: RiskFeatures):
    return predict_mlp(payload)


@app.post("/predict-risk-seq", response_model=PredictResponse)
def predict_risk_seq(payload: RiskSeqPayload, model_type: Literal["rnn", "lstm"] = "rnn"):
    if model_type == "rnn":
        if RNN_MODEL is None or RNN_SCALER is None:
            raise HTTPException(status_code=501, detail="Modèle RNN non chargé (artefacts manquants).")
        model = RNN_MODEL
        scaler = RNN_SCALER
    else:
        if LSTM_MODEL is None or LSTM_SCALER is None:
            raise HTTPException(status_code=501, detail="Modèle LSTM non chargé (artefacts manquants).")
        model = LSTM_MODEL
        scaler = LSTM_SCALER

    seq = np.array(payload.sequence, dtype=np.float32)  # shape (seq_len, n_feat)
    if seq.ndim != 2:
        raise HTTPException(status_code=400, detail="sequence doit être 2D (seq_len, n_features)")
    # standardisation par pas de temps
    seq_std = scaler.transform(seq)
    seq_std = seq_std[np.newaxis, ...]  # batch=1
    with torch.no_grad():
        logits = model(torch.tensor(seq_std))
        prob = torch.sigmoid(logits).item()
    pred = 1 if prob >= 0.5 else 0
    return PredictResponse(risk_score=prob, prediction=pred, model_type=model_type)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
