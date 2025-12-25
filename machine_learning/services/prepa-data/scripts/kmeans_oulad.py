#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kmeans_oulad.py

Applique K-Means sur les features OULAD :

- Lit datasets/oulad/fe/features.csv
- Convertit toutes les features en numérique (coerce les strings)
- Standardise les features
- Cherche le meilleur k (2..8) via inertia + silhouette
- Entraîne le KMeans final
- Sauvegarde :
    - centres de clusters
    - labels par étudiant
    - profils numériques par cluster
    - graphiques (elbow, silhouette, PCA 2D)

Les résultats sont écrits dans :
datasets/oulad/models/kmeans/
"""

import os
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 📁 Chemins (même logique que feature_engineering_oulad.py)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
FE_DIR = os.path.join(BASE_DIR, "datasets", "oulad", "fe")
OUT_DIR = os.path.join(BASE_DIR, "datasets", "oulad", "models", "kmeans")
os.makedirs(OUT_DIR, exist_ok=True)


def log(x: str) -> None:
    print(f"🔹 {x}")


def ok(x: str) -> None:
    print(f"✅ {x}")


def warn(x: str) -> None:
    print(f"⚠️  {x}")


@dataclass
class KMResult:
    k: int
    inertia: float
    silhouette: float


def load_features():
    """
    Charge features.csv et sépare :
    - X : features
    - meta : colonnes d'identification + y (si présente)
    """
    fe_path = os.path.join(FE_DIR, "features.csv")
    if not os.path.exists(fe_path):
        raise FileNotFoundError(
            f"features.csv introuvable : {fe_path}\n"
            "➡️ Lance d’abord: python services/prepa-data/scripts/feature_engineering_oulad.py"
        )

    df = pd.read_csv(fe_path)

    id_cols = [c for c in ["id_student", "code_module", "code_presentation"] if c in df.columns]
    y_col = "y" if "y" in df.columns else None

    drop_cols = id_cols + ([y_col] if y_col else [])
    feat_cols = df.drop(columns=drop_cols, errors="ignore").columns.tolist()

    X = df[feat_cols].copy()
    if id_cols or y_col:
        meta = df[id_cols + ([y_col] if y_col else [])].copy()
    else:
        meta = pd.DataFrame()

    # Remplacer NA par 0 dans les features (premier niveau)
    X = X.fillna(0)

    return df, X, feat_cols, meta


def standardize(X: pd.DataFrame):
    """
    Convertit toutes les colonnes en numérique (coerce) puis standardise.
    Toute valeur non numérique (ex: '90-100%') devient NaN puis 0.0.
    """
    # Forcer au format numérique
    X_num = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_num.values)
    return scaler, Xs, X_num


def try_ks(Xs: np.ndarray, k_min: int = 2, k_max: int = 8, random_state: int = 42):
    ks = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []
    results: list[KMResult] = []

    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(Xs)
        inertia = km.inertia_
        inertias.append(inertia)

        if 1 < k < Xs.shape[0]:
            sil = silhouette_score(Xs, labels)
        else:
            sil = np.nan
        silhouettes.append(sil)

        results.append(KMResult(k=k, inertia=inertia, silhouette=sil))
        log(f"k={k} | inertia={inertia:.2f} | silhouette={sil:.4f}")

    return ks, inertias, silhouettes, results


def pick_best_k(results: list[KMResult]) -> int:
    """
    Choix du k :
    - max silhouette
    - en cas d'égalité, inertia minimale
    """
    valid = [r for r in results if not np.isnan(r.silhouette)]
    if not valid:
        best = min(results, key=lambda x: x.inertia)
        warn("Silhouette non disponible, choix du k par inertia minimale.")
        return best.k

    best_sil = max(r.silhouette for r in valid)
    candidates = [r for r in valid if abs(r.silhouette - best_sil) < 1e-6]
    best = min(candidates, key=lambda x: x.inertia)
    return best.k


def plot_elbow(ks, inertias, out_path):
    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.title("KMeans — Elbow (Inertia)")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_silhouette(ks, silhouettes, out_path):
    plt.figure()
    plt.plot(ks, silhouettes, marker="o")
    plt.title("KMeans — Silhouette")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_pca_scatter(Xs, labels, out_path):
    pca = PCA(n_components=2, random_state=42)
    comp = pca.fit_transform(Xs)
    plt.figure()
    plt.scatter(comp[:, 0], comp[:, 1], c=labels, s=6, alpha=0.8)
    plt.title("KMeans — PCA(2D) des clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def profile_clusters(df_feat: pd.DataFrame, labels: np.ndarray, feat_cols, meta: pd.DataFrame):
    res = df_feat.copy()
    res["cluster"] = labels

    # Profil numérique : moyennes par cluster
    num_stats = res.groupby("cluster")[feat_cols].mean().reset_index()

    # Si y présent → taux de réussite par cluster
    if "y" in meta.columns:
        y_df = meta.copy()
        y_df["cluster"] = labels
        acc = (
            y_df.groupby("cluster")["y"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "pass_rate", "count": "n"})
        )
    else:
        acc = pd.DataFrame()

    return res, num_stats, acc


def main():
    log(f"FE_DIR  = {FE_DIR}")
    log(f"OUT_DIR = {OUT_DIR}")

    df, X, feat_cols, meta = load_features()
    log(f"Features: {X.shape[0]:,} lignes × {X.shape[1]} colonnes")

    # 🔹 conversion -> numérique + standardisation
    scaler, Xs, X_num = standardize(X)

    ks, inertias, silhouettes, results = try_ks(Xs, k_min=2, k_max=8)
    # Graphiques de sélection de k
    plot_elbow(ks, inertias, os.path.join(OUT_DIR, "k_elbow_inertia.png"))
    plot_silhouette(ks, silhouettes, os.path.join(OUT_DIR, "k_silhouette.png"))

    best_k = pick_best_k(results)
    ok(f"k retenu = {best_k}")

    # Entraînement final
    km = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
    labels = km.fit_predict(Xs)
    centers = km.cluster_centers_

    # 1) centres standardisés
    centers_df = pd.DataFrame(centers, columns=feat_cols)
    centers_df.insert(0, "cluster", range(best_k))
    centers_df.to_csv(os.path.join(OUT_DIR, "cluster_centers_standardized.csv"), index=False)

    # 2) labels par étudiant
    labels_df = pd.DataFrame({"cluster": labels})
    if not meta.empty:
        out_labels = pd.concat([meta.reset_index(drop=True), labels_df], axis=1)
    else:
        out_labels = labels_df
    out_labels.to_csv(os.path.join(OUT_DIR, "labels.csv"), index=False)

    # 3) profiling simple (sur les X numériques)
    df_feat = X_num.copy()
    res, num_stats, acc = profile_clusters(df_feat, labels, feat_cols, meta)
    res.to_csv(os.path.join(OUT_DIR, "features_with_cluster.csv"), index=False)
    num_stats.to_csv(os.path.join(OUT_DIR, "cluster_numeric_profile.csv"), index=False)
    if not acc.empty:
        acc.to_csv(os.path.join(OUT_DIR, "cluster_outcome_summary.csv"), index=False)

    # 4) PCA 2D
    plot_pca_scatter(Xs, labels, os.path.join(OUT_DIR, "clusters_pca2d.png"))

    # 5) méta
    meta_info = {
        "best_k": int(best_k),
        "inertia_by_k": {int(k): float(i) for k, i in zip(ks, inertias)},
        "silhouette_by_k": {
            int(k): (None if np.isnan(s) else float(s)) for k, s in zip(ks, silhouettes)
        },
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feat_cols": feat_cols,
    }
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)

    ok(f"Fini. Résultats dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
