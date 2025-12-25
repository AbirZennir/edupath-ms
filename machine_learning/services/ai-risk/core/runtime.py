import os
import joblib
import torch
import torch.nn as nn
from typing import List

# IMPORTANT: adapte les chemins si besoin
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_RISK_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # services/ai-risk
DATASETS_DIR = os.path.abspath(os.path.join(AI_RISK_DIR, "..", "..", "datasets", "oulad"))

MLP_DIR = os.path.join(DATASETS_DIR, "models", "risk")
SCALER_PATH = os.path.join(MLP_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MLP_DIR, "risk_model.pt")

# Mets ici tes noms de features (IMPORTANT pour l’explication)
# Si tu as déjà un fichier feature_names.json, on peut le charger aussi.
FEATURE_NAMES: List[str] = [
    "sum_click_total",
    "n_assessments",
    "studied_credits",
    "eng_clicks_per_day",
    "avg_score",
    "n_days_active",
    # ... complète avec tes vrais features dans le même ordre que l'entraînement
]


class RiskMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def load_scaler():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    return joblib.load(SCALER_PATH)


def load_mlp(in_dim: int):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = RiskMLP(in_dim)
    state = torch.load(MODEL_PATH, map_location="cpu")

    # compatible si tu as sauvegardé soit state_dict soit dict complet
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model
