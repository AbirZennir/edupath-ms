import os
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

FE_DIR = os.path.join(REPO_ROOT, "datasets", "oulad", "fe")
OUT_DIR = os.path.join(REPO_ROOT, "datasets", "oulad", "models", "gmm")

os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def load_features():
    path = os.path.join(FE_DIR, "features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"features.csv introuvable : {path}")

    df = pd.read_csv(path)

    # Colonnes "méta" possibles (on garde seulement celles qui existent)
    possible_meta = [
        "id_student",
        "code_module",
        "code_presentation",
        "final_result",
        "final_result_num",
        "y",
    ]
    meta_cols = [c for c in possible_meta if c in df.columns]

    if not meta_cols:
        print("⚠ Aucune colonne méta trouvée, on considère que toutes les colonnes sont des features.")
        meta = pd.DataFrame(index=df.index)
    else:
        print("🔹 Colonnes méta utilisées :", meta_cols)
        meta = df[meta_cols]

    feat_cols = [c for c in df.columns if c not in meta_cols]

    if not feat_cols:
        raise RuntimeError("Aucune colonne de features trouvée dans features.csv !")

    # Conversion numérique + NaN -> 0.0
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    print(f"🔹 {len(df)} lignes, {len(feat_cols)} features")

    return df, X, feat_cols, meta


def standardize(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    return scaler, Xs


def visualize_pca(Xs, labels, out_path):
    pca = PCA(n_components=2)
    pts = pca.fit_transform(Xs)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="tab10", s=5)
    plt.colorbar(scatter)
    plt.title("GMM Clusters (PCA 2D)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    print("🔹 FE_DIR  =", FE_DIR.replace("\\", "/"))
    print("🔹 OUT_DIR =", OUT_DIR.replace("\\", "/"))

    df, X, feat_cols, meta = load_features()

    # Standardisation
    scaler, Xs = standardize(X)

    # On garde le même k que KMeans pour comparer
    best_k = 8
    print(f"🔹 Entraînement du GMM avec k={best_k}…")

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=42,
    )
    labels = gmm.fit_predict(Xs)

    # ------------------------------- #
    # 1) Sauvegarde des labels
    # ------------------------------- #
    meta_with_labels = meta.copy()
    meta_with_labels["cluster_gmm"] = labels
    labels_path = os.path.join(OUT_DIR, "labels_gmm.csv")
    meta_with_labels.to_csv(labels_path, index=False)
    print("✔ labels_gmm.csv généré ->", labels_path)

    # ------------------------------- #
    # 2) Profil numérique par cluster
    #    (uniquement colonnes numériques)
    # ------------------------------- #
    df_with_labels = pd.concat([meta_with_labels, X], axis=1)

    # On sélectionne uniquement les colonnes numériques
    numeric_cols = df_with_labels.select_dtypes(include=[np.number]).columns
    numeric_profile = df_with_labels.groupby("cluster_gmm")[numeric_cols].mean()

    profile_path = os.path.join(OUT_DIR, "cluster_numeric_profile_gmm.csv")
    numeric_profile.to_csv(profile_path)
    print("✔ cluster_numeric_profile_gmm.csv généré ->", profile_path)

    # ------------------------------- #
    # 3) Visualisation PCA 2D
    # ------------------------------- #
    pca_path = os.path.join(OUT_DIR, "clusters_pca2d_gmm.png")
    visualize_pca(Xs, labels, pca_path)
    print("✔ clusters_pca2d_gmm.png généré ->", pca_path)

    print("🎉 Fini : résultats GMM dans", OUT_DIR.replace("\\", "/"))


if __name__ == "__main__":
    main()
