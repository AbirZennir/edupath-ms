#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_engineering_oulad.py
- Lit cleaned/oulad_merged_feature_base.csv (+ studentInfo_clean.csv)
- S'assure de récupérer la colonne final_result
- Crée des features simples (engagement, crédits, tentatives, etc.)
- Crée le label binaire y (pass=1, fail/withdrawn=0)
- Split stratifié train/val/test
- Sauvegarde dans datasets/oulad/fe/
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
CLEAN_DIR = os.path.join(BASE_DIR, "datasets", "oulad", "cleaned")
FE_DIR = os.path.join(BASE_DIR, "datasets", "oulad", "fe")
os.makedirs(FE_DIR, exist_ok=True)

def log(x): print(f"🔹 {x}")
def ok(x): print(f"✅ {x}")
def warn(x): print(f"⚠️  {x}")

def load_sources():
    merged_path = os.path.join(CLEAN_DIR, "oulad_merged_feature_base.csv")
    sinfo_path  = os.path.join(CLEAN_DIR, "studentInfo_clean.csv")

    if not os.path.exists(merged_path):
        raise FileNotFoundError(
            f"{merged_path} introuvable. Lance d'abord clean_oulad.py."
        )
    if not os.path.exists(sinfo_path):
        raise FileNotFoundError(
            f"{sinfo_path} introuvable. Vérifie le nettoyage de studentInfo."
        )

    merged = pd.read_csv(merged_path)
    sinfo  = pd.read_csv(sinfo_path)

    # Harmoniser final_result côté merged (si final_result_x / final_result_y)
    if "final_result" not in merged.columns:
        for col in ["final_result_y", "final_result_x"]:
            if col in merged.columns:
                merged.rename(columns={col: "final_result"}, inplace=True)
                warn(f"Colonne {col} renommée en final_result dans merged.")
                break

    # Préparer studentInfo (sans recréer un conflit sur final_result)
    cols_keep = [
        "id_student","code_module","code_presentation",
        "studied_credits","num_of_prev_attempts",
        "imd_band","age_band","highest_education",
        "region","disability","final_result",
    ]
    sinfo = sinfo[[c for c in cols_keep if c in sinfo.columns]].copy()

    # Si merged a déjà final_result, on ne la reprend pas depuis sinfo
    if "final_result" in merged.columns and "final_result" in sinfo.columns:
        sinfo = sinfo.drop(columns=["final_result"])

    merged = merged.merge(
        sinfo,
        on=["id_student","code_module","code_presentation"],
        how="left",
        suffixes=("", "_info")
    )

    return merged

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Récupérer final_result après tous les merges
    final_col = None
    for col in ["final_result", "final_result_info", "final_result_x", "final_result_y"]:
        if col in df.columns:
            final_col = col
            break

    if final_col is None:
        warn("Aucune colonne final_result trouvée, y sera mis à 0 (pas idéal).")
        df["y"] = 0
    else:
        df["final_result_tmp"] = df[final_col].astype(str).str.lower().str.strip()
        df["y"] = df["final_result_tmp"].map({"pass":1, "fail":0, "withdrawn":0})
        df["y"] = df["y"].fillna(0).astype(int)

    # Engagement
    if "sum_click_total" in df.columns:
        df["sum_click_total"] = pd.to_numeric(df["sum_click_total"], errors="coerce").fillna(0)
    else:
        df["sum_click_total"] = 0

    if "n_assessments" in df.columns:
        df["n_assessments"] = pd.to_numeric(df["n_assessments"], errors="coerce").fillna(0).astype(int)
    else:
        df["n_assessments"] = 0

    if "date_registration" in df.columns:
        df["date_registration"] = pd.to_numeric(df["date_registration"], errors="coerce")
    else:
        df["date_registration"] = np.nan

    df["eng_clicks_per_day"] = np.where(
        (df["date_registration"].notna()) & (df["date_registration"] > 0),
        df["sum_click_total"] / df["date_registration"].clip(lower=1),
        df["sum_click_total"]
    )

    df["assess_per_10days"] = np.where(
        (df["date_registration"].notna()) & (df["date_registration"] > 0),
        df["n_assessments"] / (df["date_registration"].clip(lower=10) / 10.0),
        df["n_assessments"]
    )

    # Charge académique
    df["studied_credits"] = pd.to_numeric(df.get("studied_credits", 0), errors="coerce").fillna(0).astype(int)
    df["num_of_prev_attempts"] = pd.to_numeric(df.get("num_of_prev_attempts", 0), errors="coerce").fillna(0).astype(int)

    # Encodage de quelques variables catégorielles
    for c in ["imd_band","age_band","highest_education","region","disability","code_module","code_presentation"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip().fillna("unknown")

    cats = ["imd_band","highest_education","disability","code_module"]
    df = pd.get_dummies(
        df,
        columns=[c for c in cats if c in df.columns],
        drop_first=True
    )

    id_cols = ["id_student","code_module","code_presentation"]
    base_feats = [
        "sum_click_total",
        "n_assessments",
        "eng_clicks_per_day",
        "assess_per_10days",
        "studied_credits",
        "num_of_prev_attempts",
    ]
    ohe_feats = [
        c for c in df.columns
        if c.startswith("imd_band_")
        or c.startswith("highest_education_")
        or c.startswith("disability_")
        or c.startswith("code_module_")
    ]

    keep = id_cols + ["y"] + base_feats + ohe_feats
    keep = [c for c in keep if c in df.columns]

    return df[keep].copy()

def do_splits(fe: pd.DataFrame):
    if "y" not in fe.columns:
        raise ValueError("Colonne y absente des features.")
    X = fe.drop(columns=["y"])
    y = fe["y"]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=0.50,
        stratify=y_tmp,
        random_state=42,
    )

    X_train.assign(y=y_train.values).to_csv(os.path.join(FE_DIR, "train.csv"), index=False)
    X_val.assign(y=y_val.values).to_csv(os.path.join(FE_DIR, "val.csv"), index=False)
    X_test.assign(y=y_test.values).to_csv(os.path.join(FE_DIR, "test.csv"), index=False)
    ok(f"Splits train/val/test sauvegardés dans {FE_DIR}")

def main():
    log("Chargement des sources (merged + studentInfo)…")
    merged = load_sources()
    log(f"merged: {len(merged):,} lignes")

    log("Construction des features…")
    fe = make_features(merged)
    fe_path = os.path.join(FE_DIR, "features.csv")
    fe.to_csv(fe_path, index=False)
    ok(f"Features sauvegardées: {fe_path} ({len(fe):,} lignes, {fe.shape[1]} colonnes)")

    log("Création des splits train/val/test…")
    do_splits(fe)

if __name__ == "__main__":
    main()
