import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
#  Paths & constantes
# ============================================================

# __file__ = services/prepa-data/scripts/risk_model_oulad.py
BASE_DIR = Path(__file__).resolve().parents[3]

FE_DIR = BASE_DIR / "datasets" / "oulad" / "fe"
MODEL_DIR = BASE_DIR / "datasets" / "oulad" / "models" / "risk"
PRED_DIR = BASE_DIR / "datasets" / "oulad" / "predictions"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#  Chargement des splits
# ============================================================

def load_splits():
    train_path = FE_DIR / "train.csv"
    val_path = FE_DIR / "val.csv"
    test_path = FE_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv introuvable : {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"val.csv introuvable : {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv introuvable : {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # colonnes méta et cible
    meta_cols = ["id_student", "code_presentation"]
    target_col = "y"

    # 🔹 Ne garder QUE les colonnes numériques comme features
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # enlever id_student + y des features
    feature_cols = [c for c in numeric_cols if c not in meta_cols + [target_col]]

    print("\n===== DEBUG load_splits() =====")
    print("Toutes les colonnes :", list(train_df.columns))
    print("Colonnes numériques :", numeric_cols)
    print("Colonnes meta       :", meta_cols)
    print("Colonne cible       :", target_col)
    print("Features retenues   :", feature_cols)
    print("Nb features         :", len(feature_cols))
    print("================================\n")

    return train_df, val_df, test_df, feature_cols, meta_cols, target_col


# ============================================================
#  Nettoyage numérique + standardisation
# ============================================================

def prepare_numeric(train_df, val_df, test_df, feature_cols):
    """
    S'assure que toutes les colonnes de features sont bien float :
      - to_numeric(errors='coerce')
      - remplace NaN par la médiane calculée sur le train
      - applique cette médiane à val et test
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for col in feature_cols:
        # train
        train_num = pd.to_numeric(train_df[col], errors="coerce")
        med = train_num.median()
        if pd.isna(med):
            # si toute la colonne est NaN par erreur, on met 0
            med = 0.0
        train_df[col] = train_num.fillna(med)

        # val / test avec la même médiane
        for df in (val_df, test_df):
            col_num = pd.to_numeric(df[col], errors="coerce")
            df[col] = col_num.fillna(med)

    return train_df, val_df, test_df


def standardize(train_df, val_df, test_df, feature_cols):
    """
    Nettoyage + StandardScaler
    """
    # 1) Nettoyage numérique
    train_df, val_df, test_df = prepare_numeric(train_df, val_df, test_df, feature_cols)

    # 2) Standardisation
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_s = train_df.copy()
    val_s = val_df.copy()
    test_s = test_df.copy()

    train_s[feature_cols] = scaler.transform(train_df[feature_cols])
    val_s[feature_cols] = scaler.transform(val_df[feature_cols])
    test_s[feature_cols] = scaler.transform(test_df[feature_cols])

    return scaler, train_s, val_s, test_s


# ============================================================
#  Modèle MLP avec WeightNorm
# ============================================================

class RiskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)

        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            lin = nn.utils.weight_norm(lin)
            layers.append(lin)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        out = nn.Linear(dims[-1], 1)
        out = nn.utils.weight_norm(out)

        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # sortie logits (on appliquera sigmoid plus tard)
        return self.net(x).squeeze(1)


# ============================================================
#  Helpers PyTorch
# ============================================================

def make_loader(df, feature_cols, target_col, batch_size=256, shuffle=False):
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ============================================================
#  Entraînement du modèle
# ============================================================

def train_model(model, train_loader, val_loader, device,
                lr=1e-3, epochs=20, patience=4):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # pas de verbose (ta version de PyTorch ne le supporte pas)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # ----------------- validation -----------------
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_losses.append(loss.item())
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        scheduler.step(val_loss)

        print(f"[{epoch:02d}/{epochs:02d}] "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("⏹️  Early stopping déclenché.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ============================================================
#  Évaluation finale + export prédictions
# ============================================================

def evaluate_and_save(model, test_df, feature_cols, target_col, device):
    X = test_df[feature_cols].values.astype(np.float32)
    y = test_df[target_col].values.astype(np.float32)

    X_tensor = torch.from_numpy(X).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= 0.5).astype(int).ravel()

    acc = accuracy_score(y, preds)
    print("\n===== Performance finale sur le test =====")
    print("Accuracy :", acc)
    print("\nClassification report :")
    print(classification_report(y, preds))

    # Export CSV des prédictions
    out_df = pd.DataFrame({
        "id_student": test_df["id_student"],
        "code_presentation": test_df["code_presentation"],
        "risk_score": probs.ravel(),
        "prediction": preds,
        "cluster": np.nan,  # colonne optionnelle pour futur lien avec les clusters
    })

    out_path = PRED_DIR / "pred_risk.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n📁 Fichier de prédictions généré : {out_path}")


# ============================================================
#  MAIN
# ============================================================

def main():
    print(f"📂 FE_DIR    = {FE_DIR}")
    print(f"📂 MODEL_DIR = {MODEL_DIR}")
    print(f"📂 PRED_DIR  = {PRED_DIR}")

    train_df, val_df, test_df, feature_cols, meta_cols, target_col = load_splits()

    print(f"🔹 {len(train_df)} lignes train, {len(val_df)} val, {len(test_df)} test")
    print(f"🔹 {len(feature_cols)} features numériques : {feature_cols[:5]}...")

    # Nettoyage + standardisation
    scaler, train_s, val_s, test_s = standardize(
        train_df, val_df, test_df, feature_cols
    )

    # Sauvegarde scaler
    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler sauvegardé : {scaler_path}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Device utilisé : {device}")

    # Dataloaders
    train_loader = make_loader(train_s, feature_cols, target_col,
                               batch_size=256, shuffle=True)
    val_loader = make_loader(val_s, feature_cols, target_col,
                             batch_size=512, shuffle=False)

    # Modèle
    model = RiskMLP(input_dim=len(feature_cols))
    model.to(device)

    # Entraînement
    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        lr=1e-3,
        epochs=20,
        patience=4,
    )

    # Sauvegarde du modèle
    model_path = MODEL_DIR / "risk_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Modèle sauvegardé : {model_path}")

    # Évaluation + export prédictions
    evaluate_and_save(model, test_s, feature_cols, target_col, device)


if __name__ == "__main__":
    main()
