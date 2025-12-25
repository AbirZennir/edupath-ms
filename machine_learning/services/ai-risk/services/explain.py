import torch
from typing import List, Dict


def predict_mlp(model, scaler, features: List[float]):
    x = scaler.transform([features])
    xt = torch.tensor(x, dtype=torch.float32)
    score = float(model(xt).item())
    pred = 1 if score >= 0.5 else 0
    return score, pred


def explain_local_variation(model, scaler, features: List[float], feature_names: List[str], top_k: int = 5):
    base_score, base_pred = predict_mlp(model, scaler, features)

    impacts: List[Dict] = []
    for i, name in enumerate(feature_names):
        x_mod = features.copy()
        # perturbation légère (+10%), si feature = 0 on met un petit delta
        x_mod[i] = x_mod[i] * 1.1 if x_mod[i] != 0 else 0.1

        s_mod, _ = predict_mlp(model, scaler, x_mod)
        impact = s_mod - base_score

        impacts.append({"feature": name, "impact": round(float(impact), 6)})

    impacts = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)[:top_k]
    return base_score, base_pred, impacts
