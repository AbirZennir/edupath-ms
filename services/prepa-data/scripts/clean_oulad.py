#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_oulad.py (version corrigée)
Nettoyage complet du dataset OULAD (Open University Learning Analytics).
- Charge 7 CSV
- Nettoie (doublons, types, NaN, valeurs extrêmes)
- Normalise les clés (code_module, code_presentation) en minuscules partout
- Valide clés communes et cohérence inter-tables
- Sauvegarde versions *_clean.csv
- Génère des rapports HTML (optionnels) de profiling, en évitant les erreurs sur DF vides

Arborescence attendue :
edupath-ms/
├─ datasets/
│   └─ oulad/
│       ├─ assessments.csv
│       ├─ courses.csv
│       ├─ studentAssessment.csv
│       ├─ studentInfo.csv
│       ├─ studentRegistration.csv
│       ├─ studentVle.csv
│       ├─ vle.csv
│       └─ cleaned/  (sera créé si absent)
└─ services/prepa-data/scripts/clean_oulad.py
"""

import os
import sys
import json
import argparse
import traceback
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ----------------------------------
# Chemins
# ----------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
RAW_DIR = os.path.join(BASE_DIR, "datasets", "oulad")
CLEAN_DIR = os.path.join(RAW_DIR, "cleaned")
REPORTS_DIR = os.path.join(CLEAN_DIR, "reports")

# ----------------------------------
# Logs
# ----------------------------------
def log(msg: str):
    print(f"🔹 {msg}")

def warn(msg: str):
    print(f"⚠️  {msg}")

def ok(msg: str):
    print(f"✅ {msg}")

def err(msg: str):
    print(f"❌ {msg}", file=sys.stderr)

# ----------------------------------
# Helpers
# ----------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def drop_full_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if after < before:
        log(f"Suppression de {before - after} doublon(s).")
    return df

def safe_clip(df: pd.DataFrame, col: str, min_val: float = None, max_val: float = None) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if min_val is not None:
            df[col] = df[col].clip(lower=min_val)
        if max_val is not None:
            df[col] = df[col].clip(upper=max_val)
    return df

def ensure_int(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def ensure_float(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df

def normalize_str_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    return df

def to_datetime_days_from_start(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # OULAD code souvent les dates comme des offsets (jours) par rapport au début de présentation.
    # On garde l'offset sous forme d'entier.
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def save_clean(df: pd.DataFrame, name: str):
    os.makedirs(CLEAN_DIR, exist_ok=True)
    out_path = os.path.join(CLEAN_DIR, name)
    df.to_csv(out_path, index=False)
    ok(f"Sauvegardé : {out_path} ({len(df):,} lignes)")

# ----------------------------------
# Nettoyages par table
# ----------------------------------
def clean_courses(path: str) -> pd.DataFrame:
    log("Nettoyage: courses.csv")
    df = pd.read_csv(os.path.join(path, "courses.csv"))
    df = standardize_columns(df)
    df = drop_full_duplicates(df)
    df = ensure_int(df, ["module_presentation_length"])
    df = normalize_str_cols(df, ["code_module", "code_presentation"])
    df = df.sort_values(["code_module", "code_presentation"]).reset_index(drop=True)
    return df

def clean_student_info(path: str) -> pd.DataFrame:
    log("Nettoyage: studentInfo.csv")
    df = pd.read_csv(os.path.join(path, "studentInfo.csv"))
    df = standardize_columns(df)
    df = drop_full_duplicates(df)

    # Normalisation clés pour fusion 📌
    df = normalize_str_cols(df, ["code_module", "code_presentation"])

    # Remplissage valeurs manquantes / normalisation
    fill_map = {
        "imd_band": "unknown",
        "highest_education": "no info",
        "age_band": "unknown",
        "region": "unknown"
    }
    for k, v in fill_map.items():
        if k in df.columns:
            df[k] = df[k].fillna(v)

    df = normalize_str_cols(df, ["gender", "region", "highest_education", "disability", "final_result", "age_band", "imd_band"])

    # Types
    df = ensure_int(df, ["num_of_prev_attempts", "studied_credits"])
    if "id_student" in df.columns:
        df = df[df["id_student"].notna()].copy()

    # Harmoniser final_result
    if "final_result" in df.columns:
        mapping = {
            "pass": "pass",
            "fail": "fail",
            "withdrawn": "withdrawn",
            "distinction": "pass"  # regrouper distinction avec pass pour un label binaire
        }
        df["final_result"] = df["final_result"].map(lambda x: mapping.get(x, x))

    df = df.reset_index(drop=True)
    return df

def clean_assessments(path: str) -> pd.DataFrame:
    log("Nettoyage: assessments.csv")
    df = pd.read_csv(os.path.join(path, "assessments.csv"))
    df = standardize_columns(df)
    df = drop_full_duplicates(df)
    df = normalize_str_cols(df, ["assessment_type", "code_module", "code_presentation"])
    df = to_datetime_days_from_start(df, "date")
    df = ensure_int(df, ["date"])
    if "id_assessment" in df.columns:
        df = df[df["id_assessment"].notna()].copy()
    df = df.sort_values(["code_module", "code_presentation", "date"]).reset_index(drop=True)
    return df

def clean_student_assessment(path: str, assessments_df: pd.DataFrame = None) -> pd.DataFrame:
    log("Nettoyage: studentAssessment.csv")
    df = pd.read_csv(os.path.join(path, "studentAssessment.csv"))
    df = standardize_columns(df)
    df = drop_full_duplicates(df)

    df = ensure_float(df, ["score"])
    df = df[df["score"].notna()].copy()
    df = safe_clip(df, "score", 0, 100)

    if "id_student" in df.columns:
        before = len(df)
        df = df[df["id_student"].notna()]
        if len(df) < before:
            warn(f"studentAssessment: {before - len(df)} ligne(s) sans id_student supprimée(s).")

    # Validation croisée : id_assessment doit exister
    if assessments_df is not None and "id_assessment" in df.columns:
        before = len(df)
        df = df.merge(assessments_df[["id_assessment"]], on="id_assessment", how="inner")
        if len(df) < before:
            warn(f"studentAssessment: {before - len(df)} ligne(s) avec id_assessment inconnu supprimée(s).")

    df = df.reset_index(drop=True)
    return df

def clean_vle(path: str) -> pd.DataFrame:
    log("Nettoyage: vle.csv")
    df = pd.read_csv(os.path.join(path, "vle.csv"))
    df = standardize_columns(df)
    df = drop_full_duplicates(df)
    df = normalize_str_cols(df, ["activity_type", "code_module", "code_presentation"])
    df = ensure_int(df, ["id_site", "week_from", "week_to"])
    df = df.fillna({"week_from": 0, "week_to": 0})
    df = df.reset_index(drop=True)
    return df

def clean_student_vle(path: str) -> pd.DataFrame:
    log("Nettoyage: studentVle.csv")
    df = pd.read_csv(os.path.join(path, "studentVle.csv"))
    df = standardize_columns(df)
    df = drop_full_duplicates(df)

    # Normalisation clés pour fusion 📌
    df = normalize_str_cols(df, ["code_module", "code_presentation"])

    # Nettoyage des clicks + outliers grossiers
    if "sum_click" in df.columns:
        df = df[df["sum_click"].notna()].copy()
        df = ensure_int(df, ["sum_click"])
        MAX_CLICKS = 10_000
        before = len(df)
        df = df[df["sum_click"] <= MAX_CLICKS]
        if len(df) < before:
            warn(f"studentVle: {before - len(df)} ligne(s) outliers sum_click supprimée(s) (> {MAX_CLICKS}).")

    # 'date' offset (int)
    df = ensure_int(df, ["date"])

    # Supprimer lignes sans id_student/id_site
    for key in ["id_student", "id_site"]:
        if key in df.columns:
            before = len(df)
            df = df[df[key].notna()]
            if len(df) < before:
                warn(f"studentVle: {before - len(df)} ligne(s) sans {key} supprimée(s).")

    df = df.reset_index(drop=True)
    return df

def clean_student_registration(path: str) -> pd.DataFrame:
    log("Nettoyage: studentRegistration.csv")
    df = pd.read_csv(os.path.join(path, "studentRegistration.csv"))
    df = standardize_columns(df)
    df = drop_full_duplicates(df)
    df = normalize_str_cols(df, ["code_module", "code_presentation"])
    df = ensure_int(df, ["date_registration", "date_unregistration"])

    if "id_student" in df.columns:
        before = len(df)
        df = df[df["id_student"].notna()]
        if len(df) < before:
            warn(f"studentRegistration: {before - len(df)} ligne(s) sans id_student supprimée(s).")

    df = df.reset_index(drop=True)
    return df

# ----------------------------------
# Profiling (optionnel, safe si DF vide)
# ----------------------------------
def try_generate_profiling(df: pd.DataFrame, title: str, filename: str):
    if df is None or df.empty:
        warn(f"Profiling sauté : DataFrame vide pour {title}.")
        return
    try:
        from ydata_profiling import ProfileReport
    except Exception:
        warn("ydata-profiling non installé - rapport HTML non généré.")
        return
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(REPORTS_DIR, filename)
    profile = ProfileReport(df, title=title, explorative=True, minimal=True)
    profile.to_file(out)
    ok(f"Profiling généré : {out}")

# ----------------------------------
# Orchestration
# ----------------------------------
def run_all():
    os.makedirs(CLEAN_DIR, exist_ok=True)

    # 1) Courses
    courses = clean_courses(RAW_DIR)
    save_clean(courses, "courses_clean.csv")

    # 2) Student Info
    sinfo = clean_student_info(RAW_DIR)
    save_clean(sinfo, "studentInfo_clean.csv")

    # 3) Assessments
    assess = clean_assessments(RAW_DIR)
    save_clean(assess, "assessments_clean.csv")

    # 4) Student Assessment (avec check assessments)
    s_assess = clean_student_assessment(RAW_DIR, assessments_df=assess)
    save_clean(s_assess, "studentAssessment_clean.csv")

    # 5) VLE
    vle = clean_vle(RAW_DIR)
    save_clean(vle, "vle_clean.csv")

    # 6) Student VLE
    s_vle = clean_student_vle(RAW_DIR)
    save_clean(s_vle, "studentVle_clean.csv")

    # 7) Student Registration
    sreg = clean_student_registration(RAW_DIR)
    save_clean(sreg, "studentRegistration_clean.csv")

    # ---- Fusion globale pour base features IA ----
    common_keys = ["id_student", "code_module", "code_presentation"]
    merged = None
    try:
        # Fusion student info + registration
        merged = sinfo.merge(
            sreg[common_keys + ["date_registration", "date_unregistration"]],
            on=common_keys,
            how="inner"
        )
        log(f"Fusion sinfo ⨝ sreg → {len(merged):,} lignes")

        # Agrégats clicks
        if all(k in s_vle.columns for k in common_keys + ["sum_click"]):
            clicks = (
                s_vle.groupby(common_keys, as_index=False)["sum_click"]
                .sum()
                .rename(columns={"sum_click": "sum_click_total"})
            )
            merged = merged.merge(clicks, on=common_keys, how="left")
            log(f"Ajout sum_click_total → {len(merged):,} lignes")

        # Nombre d'évaluations rendues
        if "id_student" in s_assess.columns:
            nb_ass = (
                s_assess.groupby("id_student", as_index=False)["id_assessment"]
                .count()
                .rename(columns={"id_assessment": "n_assessments"})
            )
            merged = merged.merge(nb_ass, on="id_student", how="left")
            log(f"Ajout n_assessments → {len(merged):,} lignes")

    except Exception as e:
        warn(f"Fusion globale partielle (non bloquante) : {e}")

    if merged is not None:
        save_clean(merged, "oulad_merged_feature_base.csv")
        try_generate_profiling(merged, "OULAD Cleaned (Merged Feature Base)", "oulad_merged_profile.html")

    # Rapports par table (optionnels)
    try_generate_profiling(sinfo, "studentInfo_clean", "studentInfo_profile.html")
    try_generate_profiling(s_vle, "studentVle_clean", "studentVle_profile.html")

    ok("Nettoyage complet terminé.")

def run_single(table: str):
    cleaners = {
        "courses": lambda: clean_courses(RAW_DIR),
        "studentinfo": lambda: clean_student_info(RAW_DIR),
        "assessments": lambda: clean_assessments(RAW_DIR),
        "studentassessment": lambda: clean_student_assessment(RAW_DIR),
        "vle": lambda: clean_vle(RAW_DIR),
        "studentvle": lambda: clean_student_vle(RAW_DIR),
        "studentregistration": lambda: clean_student_registration(RAW_DIR),
    }
    key = table.lower().replace("_", "")
    if key not in cleaners:
        err(f"Table inconnue: {table}")
        err(f"Tables possibles: {', '.join(cleaners.keys())}")
        sys.exit(1)
    df = cleaners[key]()
    save_clean(df, f"{table}_clean.csv")
    ok(f"Nettoyage partiel terminé pour {table}.")

# ----------------------------------
# CLI
# ----------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Nettoyage du dataset OULAD")
    p.add_argument("--table", type=str, default=None,
                   help="Nettoyer une table spécifique (ex: studentInfo, studentVle, ...). Par défaut: tout.")
    return p.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        log(f"RAW_DIR = {RAW_DIR}")
        log(f"CLEAN_DIR = {CLEAN_DIR}")
        if not os.path.isdir(RAW_DIR):
            err("Le dossier datasets/oulad est introuvable. Vérifie l'arborescence.")
            sys.exit(1)

        if args.table:
            run_single(args.table)
        else:
            run_all()

    except Exception as e:
        err("Échec du nettoyage :")
        err(str(e))
        traceback.print_exc()
        sys.exit(2)
