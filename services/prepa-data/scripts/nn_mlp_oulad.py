import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils import weight_norm


# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

FE_DIR = os.path.join(REPO_ROOT, "datasets", "oulad", "fe")


def detect_meta_and_features(df: pd.DataFrame):
    possible_meta = [
        "id_student",
        "code_module",
        "code_presentation",
        "final_result",
        "final_result_num",
        "cluster",
        "cluster_gmm",
    ]
    target_col = "y"  # variable de sortie
    meta_cols = [c for c in possible_meta if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_cols + [target_col]]
    return target_col, meta_cols, feat_cols


# --------------------------------------------------------------------
# Modèle MLP “normé”
# --------------------------------------------------------------------
class MLPNorm(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        # Layer 1
        self.fc1 = weight_norm(nn.Linear(in_dim, 64))
        self.bn1 = nn.BatchNorm1d(64)
        # Layer 2
        self.fc2 = weight_norm(nn.Linear(64, 32))
        self.bn2 = nn.BatchNorm1d(32)
        # Output
        self.out = nn.Linear(32, 1)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.out(x)  # pas de sigmoid ici (on utilise BCEWithLogitsLoss)
        return x


# --------------------------------------------------------------------
# Training
# --------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb).squeeze(1)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_targets = np.concatenate(all_targets)

    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(all_targets, preds)
    return acc, all_targets, preds


def main():
    print("🔹 FE_DIR =", FE_DIR.replace("\\", "/"))

    feat_path = os.path.join(FE_DIR, "features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"features.csv introuvable : {feat_path}")

    df = pd.read_csv(feat_path)

    # Détection des colonnes
    target_col, meta_cols, feat_cols = detect_meta_and_features(df)
    if target_col not in df.columns:
        raise RuntimeError(f"La colonne cible '{target_col}' est introuvable dans features.csv")

    print("🔹 Colonnes méta :", meta_cols)
    print("🔹 #features      :", len(feat_cols))

    # X, y
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    y = df[target_col].astype(float).values  # 0/1

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Standardisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔹 Device utilisé :", device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Modèle
    model = MLPNorm(in_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Entraînement
    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_acc, _, _ = eval_epoch(model, train_loader, device)
        test_acc, _, _ = eval_epoch(model, test_loader, device)
        print(
            f"[{epoch:02d}/{n_epochs}] "
            f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}"
        )

    # Rapport final sur le test
    test_acc, y_true, y_pred = eval_epoch(model, test_loader, device)
    print("\n===== Performance finale sur le test =====")
    print("Accuracy :", test_acc)
    print("\nClassification report :")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
