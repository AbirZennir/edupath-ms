from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
OULAD_DIR = REPO_ROOT / "datasets" / "oulad"
FE_DIR = OULAD_DIR / "fe"
MODEL_DIR = OULAD_DIR / "models" / "risk_seq"
PRED_DIR = OULAD_DIR / "predictions"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

NPZ_PATH = FE_DIR / "seq_features.npz"
TRAIN_META = FE_DIR / "seq_train_meta.csv"
VAL_META = FE_DIR / "seq_val_meta.csv"
TEST_META = FE_DIR / "seq_test_meta.csv"

SCALER_PATH = MODEL_DIR / "scaler_seq.pkl"
MODEL_PATH = MODEL_DIR / "risk_rnn.pt"
PRED_PATH = PRED_DIR / "pred_risk_rnn.csv"

BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
PATIENCE = 4
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class RNNClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, nonlinearity="tanh", dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)          # (B,T,H)
        last = out[:, -1, :]          # (B,H)
        last = self.dropout(last)
        logits = self.fc(last)        # (B,1)
        return logits

def standardize_seq(X_train, X_val, X_test):
    # Fit scaler sur toutes les frames du train: (N*T, F)
    scaler = StandardScaler()
    N, T, F = X_train.shape
    scaler.fit(X_train.reshape(N*T, F))

    def transform(X):
        n, t, f = X.shape
        X2 = scaler.transform(X.reshape(n*t, f)).reshape(n, t, f)
        return X2.astype(np.float32)

    return scaler, transform(X_train), transform(X_val), transform(X_test)

def train_loop(model, train_loader, val_loader, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        correct = 0
        total = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()

        train_loss = float(np.mean(train_losses))
        train_acc = correct / total

        model.eval()
        val_losses = []
        vcorrect, vtotal = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                preds = (torch.sigmoid(logits) >= 0.5).float()
                vcorrect += (preds == yb).sum().item()
                vtotal += yb.numel()

        val_loss = float(np.mean(val_losses))
        val_acc = vcorrect / vtotal
        scheduler.step(val_loss)

        print(f"[{epoch:02d}/{EPOCHS}] loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("⏹️ Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def main():
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Introuvable: {NPZ_PATH}. Lance d’abord build_sequence_dataset.py")

    data = np.load(NPZ_PATH, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"📂 FE_DIR = {FE_DIR}")
    print(f"🔹 Shapes: train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    scaler, X_train_s, X_val_s, X_test_s = standardize_seq(X_train, X_val, X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"💾 Scaler saved: {SCALER_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔹 Device: {device}")

    train_loader = DataLoader(SeqDataset(X_train_s, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val_s, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SeqDataset(X_test_s, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = RNNClassifier(input_size=X_train_s.shape[-1]).to(device)
    model = train_loop(model, train_loader, val_loader, device)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Model saved: {MODEL_PATH}")

    # ===== Test eval + export preds =====
    model.eval()
    all_probs = []
    all_true = []
    with torch.no_grad():
        for Xb, yb in test_loader:
            logits = model(Xb.to(device))
            probs = torch.sigmoid(logits).cpu().view(-1)
            all_probs.extend(probs.tolist())
            all_true.extend(yb.view(-1).tolist())

    probs = np.array(all_probs, dtype=np.float32)
    y_true = np.array(all_true, dtype=np.float32)
    y_pred = (probs >= 0.5).astype(np.float32)

    acc = accuracy_score(y_true, y_pred)
    print("\n===== Test performance (RNN) =====")
    print("Accuracy:", acc)
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))

    meta_test = pd.read_csv(TEST_META)
    out = meta_test.copy()
    out["risk_score"] = probs
    out["prediction"] = y_pred
    out.to_csv(PRED_PATH, index=False)
    print(f"📁 Predictions saved: {PRED_PATH}")

if __name__ == "__main__":
    main()
