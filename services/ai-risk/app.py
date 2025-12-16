from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from pathlib import Path
import os
import joblib
import torch
import torch.nn as nn

# =========================
# Paths (robustes)
# =========================
THIS_DIR = Path(__file__).resolve().parent                 # .../services/ai-risk
REPO_ROOT = THIS_DIR.parents[1]                            # .../services
REPO_ROOT = REPO_ROOT.parent                               # .../edupath-ms

# Permet override via variables d'env si besoin
OULAD_DIR = Path(os.getenv("OULAD_DIR", str(REPO_ROOT / "datasets" / "oulad")))

MLP_DIR = Path(os.getenv("RISK_MLP_DIR", str(OULAD_DIR / "models" / "risk")))
SEQ_DIR = Path(os.getenv("RISK_SEQ_DIR", str(OULAD_DIR / "models" / "risk_seq")))

MLP_SCALER_PATH = Path(os.getenv("RISK_SCALER_PATH", str(MLP_DIR / "scaler.pkl")))
MLP_MODEL_PATH  = Path(os.getenv("RISK_MLP_PATH", str(MLP_DIR / "risk_model.pt")))

RNN_MODEL_PATH  = Path(os.getenv("RISK_RNN_PATH", str(SEQ_DIR / "risk_rnn.pt")))
LSTM_MODEL_PATH = Path(os.getenv("RISK_LSTM_PATH", str(SEQ_DIR / "risk_lstm.pt")))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Models (PyTorch)
# =========================
class RiskMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class RiskRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)         # out: (B, T, H)
        last = out[:, -1, :]         # (B, H)
        return self.fc(last)         # (B, 1)

class RiskLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)        # (B, T, H)
        last = out[:, -1, :]
        return self.fc(last)

# =========================
# Lazy-loaded artifacts
# =========================
mlp_scaler = None
mlp_model = None
seq_models: Dict[str, nn.Module] = {}

def _safe_exists(p: Path) -> bool:
    return p is not None and p.exists() and p.is_file()

def load_mlp():
    global mlp_scaler, mlp_model

    if mlp_model is not None and mlp_scaler is not None:
        return

    if not _safe_exists(MLP_SCALER_PATH):
        raise FileNotFoundError(f"Scaler introuvable: {MLP_SCALER_PATH}")
    if not _safe_exists(MLP_MODEL_PATH):
        raise FileNotFoundError(f"Modèle MLP introuvable: {MLP_MODEL_PATH}")

    mlp_scaler = joblib.load(MLP_SCALER_PATH)

    in_dim = getattr(mlp_scaler, "n_features_in_", None)
    if in_dim is None:
        raise RuntimeError("Impossible de détecter n_features_in_ depuis le scaler.pkl")

    model = RiskMLP(in_dim).to(DEVICE)
    state = torch.load(MLP_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    mlp_model = model

def load_seq(model_name: str):
    model_name = model_name.lower()
    if model_name in seq_models:
        return

    path = RNN_MODEL_PATH if model_name == "rnn" else LSTM_MODEL_PATH if model_name == "lstm" else None
    if path is None:
        raise ValueError("model doit être 'rnn' ou 'lstm'")

    if not _safe_exists(path):
        raise FileNotFoundError(f"Modèle {model_name.upper()} introuvable: {path}")

    # On va charger le state_dict, mais on doit connaître input_dim.
    # Astuce: on le récupère depuis le state_dict (shape de weight_ih_l0)
    state = torch.load(path, map_location=DEVICE)
    input_dim = None

    # RNN/LSTM -> weight_ih_l0 a shape (hidden_dim, input_dim)
    for k, v in state.items():
        if "weight_ih_l0" in k and hasattr(v, "shape"):
            input_dim = int(v.shape[1])
            hidden_dim = int(v.shape[0])
            break

    if input_dim is None:
        raise RuntimeError(f"Impossible de déduire input_dim depuis le state_dict de {path}")

    if model_name == "rnn":
        model = RiskRNN(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)
    else:
        model = RiskLSTM(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)

    model.load_state_dict(state)
    model.eval()
    seq_models[model_name] = model

def sigmoid_score(logit: torch.Tensor) -> float:
    prob = torch.sigmoid(logit).item()
    return float(prob)

# =========================
# API schemas
# =========================
class PredictRiskRequest(BaseModel):
    features: List[float] = Field(..., min_items=1)

class PredictRiskResponse(BaseModel):
    model: str
    risk_score: float
    prediction: int

class PredictRiskSeqRequest(BaseModel):
    model: Literal["rnn", "lstm"] = "lstm"
    sequence: List[List[float]] = Field(..., min_items=1)  # shape: [T][D]

# =========================
# FastAPI
# =========================
app = FastAPI(title="EduPath AI Risk Service", version="1.0.0")

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "paths": {
            "OULAD_DIR": str(OULAD_DIR),
            "MLP_SCALER_PATH": str(MLP_SCALER_PATH),
            "MLP_MODEL_PATH": str(MLP_MODEL_PATH),
            "RNN_MODEL_PATH": str(RNN_MODEL_PATH),
            "LSTM_MODEL_PATH": str(LSTM_MODEL_PATH),
        }
    }

@app.get("/models")
def models():
    return {
        "mlp": {"scaler": _safe_exists(MLP_SCALER_PATH), "model": _safe_exists(MLP_MODEL_PATH)},
        "rnn": {"model": _safe_exists(RNN_MODEL_PATH)},
        "lstm": {"model": _safe_exists(LSTM_MODEL_PATH)},
    }

@app.post("/predict-risk", response_model=PredictRiskResponse)
def predict_risk(req: PredictRiskRequest):
    try:
        load_mlp()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement MLP: {e}")

    # 1) scale
    try:
        import numpy as np
        x_np = np.array(req.features, dtype=float).reshape(1, -1)
        x_scaled = mlp_scaler.transform(x_np)
        x = torch.tensor(x_scaled, dtype=torch.float32, device=DEVICE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Features invalides/scaling échoué: {e}")

    # 2) predict
    with torch.no_grad():
        logit = mlp_model(x)                 # (1,1)
        score = sigmoid_score(logit)

    pred = 1 if score >= 0.5 else 0
    return PredictRiskResponse(model="mlp", risk_score=score, prediction=pred)

@app.post("/predict-risk-seq", response_model=PredictRiskResponse)
def predict_risk_seq(req: PredictRiskSeqRequest):
    model_name = req.model.lower()

    try:
        load_seq(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement {model_name.upper()}: {e}")

    # sequence: [T][D] -> tensor (1, T, D)
    try:
        import numpy as np
        seq_np = np.array(req.sequence, dtype=float)
        if seq_np.ndim != 2:
            raise ValueError("sequence doit être une matrice 2D [T][D]")
        x = torch.tensor(seq_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Sequence invalide: {e}")

    with torch.no_grad():
        logit = seq_models[model_name](x)    # (1,1)
        score = sigmoid_score(logit)

    pred = 1 if score >= 0.5 else 0
    return PredictRiskResponse(model=model_name, risk_score=score, prediction=pred)
