import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

FE_DIR = os.path.join(REPO_ROOT, "datasets", "oulad", "fe")
KMEANS_DIR = os.path.join(REPO_ROOT, "datasets", "oulad", "models", "kmeans")
GMM_DIR = os.path.join(REPO_ROOT, "datasets", "oulad", "models", "gmm")


def detect_meta_and_features(df: pd.DataFrame):
    """Détecte les colonnes méta et retourne aussi la liste des features."""
    possible_meta = [
        "id_student",
        "code_module",
        "code_presentation",
        "final_result",
        "final_result_num",
        "y",
    ]
    meta_cols = [c for c in possible_meta if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    return meta_cols, feat_cols


def load_all():
    # Features
    feat_path = os.path.join(FE_DIR, "features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"features.csv introuvable : {feat_path}")

    features = pd.read_csv(feat_path)
    meta_cols, feat_cols = detect_meta_and_features(features)

    # Labels KMeans
    kmeans_labels_path = os.path.join(KMEANS_DIR, "labels.csv")
    if not os.path.exists(kmeans_labels_path):
        raise FileNotFoundError(f"labels.csv introuvable (KMeans) : {kmeans_labels_path}")
    kmeans_labels = pd.read_csv(kmeans_labels_path)

    # Labels GMM
    gmm_labels_path = os.path.join(GMM_DIR, "labels_gmm.csv")
    if not os.path.exists(gmm_labels_path):
        raise FileNotFoundError(f"labels_gmm.csv introuvable (GMM) : {gmm_labels_path}")
    gmm_labels = pd.read_csv(gmm_labels_path)

    # Colonnes communes pour faire le join (id_student, code_presentation…)
    join_keys = [c for c in ["id_student", "code_presentation"] if c in features.columns
                 and c in kmeans_labels.columns and c in gmm_labels.columns]
    if not join_keys:
        raise RuntimeError("Impossible de trouver des clés communes (id_student, code_presentation).")

    # Détection des colonnes de cluster dans les fichiers labels
    kmeans_cluster_col = [c for c in kmeans_labels.columns if "cluster" in c.lower()][0]
    gmm_cluster_col = [c for c in gmm_labels.columns if "cluster" in c.lower()][0]

    # Merge
    df = features.merge(
        kmeans_labels[join_keys + [kmeans_cluster_col]],
        on=join_keys,
        how="inner",
    ).merge(
        gmm_labels[join_keys + [gmm_cluster_col]],
        on=join_keys,
        how="inner",
    )

    print(f"🔹 Lignes après fusion: {len(df)}")

    # X numérique pour les scores
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df, X, feat_cols, meta_cols, kmeans_cluster_col, gmm_cluster_col


def main():
    print("📂 FE_DIR   =", FE_DIR.replace("\\", "/"))
    print("📂 KMEANS   =", KMEANS_DIR.replace("\\", "/"))
    print("📂 GMM      =", GMM_DIR.replace("\\", "/"))

    df, X, feat_cols, meta_cols, kmeans_cluster_col, gmm_cluster_col = load_all()

    # Standardisation pour les scores
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    # Scores silhouette
    sil_kmeans = silhouette_score(Xs, df[kmeans_cluster_col])
    sil_gmm = silhouette_score(Xs, df[gmm_cluster_col])

    print("\n===== Scores de qualité des clusters (silhouette) =====")
    print(f"KMeans (k={df[kmeans_cluster_col].nunique()}): {sil_kmeans:.4f}")
    print(f"GMM    (k={df[gmm_cluster_col].nunique()}): {sil_gmm:.4f}")

    # Matrice de correspondance des clusters
    print("\n===== Correspondance KMeans vs GMM (table croisée) =====")
    crosstab = pd.crosstab(df[kmeans_cluster_col], df[gmm_cluster_col])
    print(crosstab)

    # Lien avec la réussite si 'y' existe
    if "y" in df.columns:
        print("\n===== Moyenne de y (proba / label de réussite) par cluster =====")
        print("KMeans :")
        print(df.groupby(kmeans_cluster_col)["y"].mean())
        print("\nGMM :")
        print(df.groupby(gmm_cluster_col)["y"].mean())
    else:
        print("\n⚠ Colonne 'y' absente : pas de comparaison directe avec la réussite.")

    print("\n🎉 Comparaison terminée.")


if __name__ == "__main__":
    main()
