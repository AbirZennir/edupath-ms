"""
RNN / LSTM pour le risque d'échec OULAD.

Hypothèses de données séquentielles :
- Des tenseurs numpy (npz) présents dans datasets/oulad/fe_seq/ :
  - train_seq.npz : X (n_samples, seq_len, n_features), y (n_samples,)
  - val_seq.npz   : idem
  - test_seq.npz  : idem
- Les features sont numériques (clics par jour/semaine, etc.).

Commandes (exemples) :
    python rnn_lstm_oulad.py --model rnn
    python rnn_lstm_oulad.py --model lstm
"""

from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[3]
SEQ_DIR = BASE_DIR / "datasets" / "oulad" / "fe_seq"
MODEL_ROOT = BASE_DIR / "datasets" / "oulad" / "models"
PRED_DIR = BASE_DIR / "datasets" / "oulad" / "predictions"


def load_seq_split(split: str):
    path = SEQ_DIR / f"{split}_seq.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} manquant. Attendu un npz avec X (n, seq_len, n_feat) et y (n,)."
        )
    data = np.load(path)
    return data["X"], data["y"]


def standardize_sequences(X_train, X_val, X_test):
    """
    Standardisation feature-wise sur la dimension feature (en écrasant seq_len).
    """
    n_train, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))

    def transform(X):
        return scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape)

    return scaler, transform(X_train), transform(X_val), transform(X_test)


class RiskRNN(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(1)


class RiskLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(1)


def make_loader(X, y, batch_size=128, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_loop(model, train_loader, val_loader, device, lr=1e-3, epochs=20, patience=4):
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    best_state = None
    best_val = float("inf")
    patience_left = patience

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        print(f"[{ep:02d}/{epochs}] train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = model.state_dict()
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            all_logits.append(logits)
            all_y.append(yb.numpy())
    logits = np.concatenate(all_logits).ravel()
    y = np.concatenate(all_y).ravel()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y, preds)
    clf = classification_report(y, preds, output_dict=True)
    cm = confusion_matrix(y, preds)
    auc = roc_auc_score(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)
    return {"acc": acc, "clf": clf, "cm": cm, "auc": auc, "fpr": fpr, "tpr": tpr, "probs": probs, "preds": preds, "y": y}


def save_artifacts(metrics, out_dir: Path, model_path: Path, scaler_path: Path, preds_path: Path, probs, preds, y_true):
    out_dir.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Courbe ROC
    roc_path = out_dir / "roc_curve.png"
    plt.figure(figsize=(6, 5))
    plt.plot(metrics["fpr"], metrics["tpr"], label=f"ROC (AUC={metrics['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Risk RNN/LSTM ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    # Rapport
    report_path = out_dir / "report.md"
    f1_0 = metrics["clf"].get("0", {}).get("f1-score", float("nan"))
    f1_1 = metrics["clf"].get("1", {}).get("f1-score", float("nan"))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Risk RNN/LSTM Report\n\n")
        f.write(f"- Model path: {model_path}\n")
        f.write(f"- Scaler path: {scaler_path}\n")
        f.write(f"- Predictions: {preds_path}\n")
        f.write(f"- ROC curve: {roc_path}\n\n")
        f.write("## Metrics (test)\n")
        f.write(f"- Accuracy: {metrics['acc']:.4f}\n")
        f.write(f"- F1 class 0: {f1_0:.4f}\n")
        f.write(f"- F1 class 1: {f1_1:.4f}\n")
        f.write(f"- ROC-AUC: {metrics['auc']:.4f}\n")
        cm = metrics["cm"]
        f.write(f"- Confusion matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}\n")

    # Predictions CSV
    df_pred = pd.DataFrame({"proba": probs, "prediction": preds, "y_true": y_true})
    df_pred.to_csv(preds_path, index=False)

    print(f"Rapport: {report_path}")
    print(f"ROC: {roc_path}")
    print(f"Preds: {preds_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rnn", "lstm"], default="rnn", help="Choix du modèle séquentiel")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    X_train, y_train = load_seq_split("train")
    X_val, y_val = load_seq_split("val")
    X_test, y_test = load_seq_split("test")

    scaler, X_train_s, X_val_s, X_test_s = standardize_sequences(X_train, X_val, X_test)

    n_feat = X_train.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model == "rnn":
        model = RiskRNN(input_dim=n_feat)
        out_dir = MODEL_ROOT / "risk_rnn"
    else:
        model = RiskLSTM(input_dim=n_feat)
        out_dir = MODEL_ROOT / "risk_lstm"

    model.to(device)

    train_loader = make_loader(X_train_s, y_train, batch_size=args.batch, shuffle=True)
    val_loader = make_loader(X_val_s, y_val, batch_size=args.batch, shuffle=False)
    test_loader = make_loader(X_test_s, y_test, batch_size=args.batch, shuffle=False)

    model = train_loop(model, train_loader, val_loader, device, epochs=args.epochs, patience=args.patience)

    metrics = evaluate(model, test_loader, device)

    model_path = out_dir / f"risk_{args.model}.pt"
    scaler_path = out_dir / f"scaler_{args.model}.pkl"
    preds_path = PRED_DIR / f"pred_risk_{args.model}.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    torch.save(model.state_dict(), model_path)

    save_artifacts(metrics, out_dir, model_path, scaler_path, preds_path, metrics["probs"], metrics["preds"], metrics["y"])


if __name__ == "__main__":
    main()
