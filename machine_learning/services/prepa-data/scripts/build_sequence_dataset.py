import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =======================
# Paths (compatibles repo)
# =======================
REPO_ROOT = Path(__file__).resolve().parents[3]  # .../services/prepa-data/scripts -> repo root
OULAD_DIR = REPO_ROOT / "datasets" / "oulad"
FE_DIR = OULAD_DIR / "fe"
FE_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_VLE = OULAD_DIR / "studentVle.csv"
STUDENT_INFO = OULAD_DIR / "studentInfo.csv"

OUT_NPZ = FE_DIR / "seq_features.npz"
OUT_TRAIN_META = FE_DIR / "seq_train_meta.csv"
OUT_VAL_META = FE_DIR / "seq_val_meta.csv"
OUT_TEST_META = FE_DIR / "seq_test_meta.csv"

# ============
# Config
# ============
T_WEEKS = 20         # longueur séquence
RANDOM_STATE = 42

def map_final_result_to_y(x: str):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    # OULAD final_result: Pass, Distinction, Fail, Withdrawn
    if s in ["pass", "distinction"]:
        return 1.0
    if s in ["fail", "withdrawn"]:
        return 0.0
    return np.nan

def main():
    print(f"📂 OULAD_DIR = {OULAD_DIR}")
    print("🔹 Chargement studentVle + studentInfo ...")

    if not STUDENT_VLE.exists():
        raise FileNotFoundError(f"Introuvable: {STUDENT_VLE}")
    if not STUDENT_INFO.exists():
        raise FileNotFoundError(f"Introuvable: {STUDENT_INFO}")

    sv = pd.read_csv(STUDENT_VLE)
    si = pd.read_csv(STUDENT_INFO)

    # Colonnes attendues OULAD:
    # studentVle: code_module, code_presentation, id_student, id_site, date, sum_click
    # studentInfo: code_module, code_presentation, id_student, final_result, ...
    required_sv = {"id_student", "code_presentation", "date", "sum_click"}
    required_si = {"id_student", "code_presentation", "final_result"}

    if not required_sv.issubset(set(sv.columns)):
        raise ValueError(f"studentVle missing columns: {required_sv - set(sv.columns)}")
    if not required_si.issubset(set(si.columns)):
        raise ValueError(f"studentInfo missing columns: {required_si - set(si.columns)}")

    # Label y depuis final_result
    si = si[["id_student", "code_presentation", "final_result"]].copy()
    si["y"] = si["final_result"].apply(map_final_result_to_y)
    si = si.dropna(subset=["y"]).copy()

    # Merge label sur studentVle
    sv = sv[["id_student", "code_presentation", "date", "sum_click"]].copy()
    sv["date"] = pd.to_numeric(sv["date"], errors="coerce")
    sv["sum_click"] = pd.to_numeric(sv["sum_click"], errors="coerce").fillna(0.0)
    sv = sv.dropna(subset=["date"]).copy()

    df = sv.merge(si[["id_student", "code_presentation", "y"]], on=["id_student", "code_presentation"], how="inner")
    print(f"🔹 Lignes après merge label: {len(df):,}")

    # Binning par semaine
    df["week"] = (df["date"] // 7).astype(int)
    df = df[(df["week"] >= 0) & (df["week"] < T_WEEKS)].copy()

    # Agrégation clicks par (student, pres, week)
    g = (
        df.groupby(["id_student", "code_presentation", "y", "week"], as_index=False)["sum_click"]
        .sum()
        .rename(columns={"sum_click": "clicks_week"})
    )

    # Construire matrice séquentielle (N,T,2) : clicks_week + cum_clicks_week
    # On pivot en (N,T)
    pivot = g.pivot_table(
        index=["id_student", "code_presentation", "y"],
        columns="week",
        values="clicks_week",
        aggfunc="sum",
        fill_value=0.0
    )

    # Assurer colonnes weeks 0..T-1
    for w in range(T_WEEKS):
        if w not in pivot.columns:
            pivot[w] = 0.0
    pivot = pivot[sorted(pivot.columns)]

    clicks = pivot.values.astype(np.float32)               # (N,T)
    cum_clicks = np.cumsum(clicks, axis=1).astype(np.float32)

    X_seq = np.stack([clicks, cum_clicks], axis=-1)        # (N,T,2)

    idx = pivot.index.to_frame(index=False)
    meta = idx[["id_student", "code_presentation", "y"]].copy()
    y = meta["y"].values.astype(np.float32)

    print(f"✅ X_seq shape = {X_seq.shape} (N,T,F)")
    print(f"✅ y shape    = {y.shape}")

    # Split stratifié : 70/15/15
    X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
        X_seq, y, meta, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    val_size = 0.15 / 0.85  # pour que val soit 15% du total
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp, y_temp, meta_temp, test_size=val_size, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"🔹 train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}")

    # Sauvegarde NPZ
    np.savez_compressed(
        OUT_NPZ,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        T=T_WEEKS, F=X_seq.shape[-1]
    )

    meta_train.to_csv(OUT_TRAIN_META, index=False)
    meta_val.to_csv(OUT_VAL_META, index=False)
    meta_test.to_csv(OUT_TEST_META, index=False)

    print(f"💾 NPZ saved: {OUT_NPZ}")
    print(f"💾 Meta train/val/test saved in: {FE_DIR}")

if __name__ == "__main__":
    main()
