#  Branch `data` — Nettoyage et Préparation du Dataset OULAD

##  Objectif
Cette branche contient **tout le code nécessaire pour nettoyer, préparer et profiler** le dataset **OULAD (Open University Learning Analytics Dataset)**.  
⚠️ **Les fichiers CSV ne sont pas inclus** car ils sont trop volumineux — chacun doit les télécharger et exécuter les scripts localement.

## 🚀 Étapes pour reproduire le même résultat que Abir

### 1️⃣ Cloner le projet et basculer sur la branche `data`

git clone https://github.com/AbirZennir/edupath-ms.git
cd edupath-ms
git checkout data

### 2️⃣ Télécharger le dataset OULAD

📥 Lien officiel :
👉 https://analyse.kmi.open.ac.uk/open-dataset

Placez les fichiers téléchargés dans le dossier suivant :

datasets/oulad/
  assessments.csv
  courses.csv
  studentAssessment.csv
  studentInfo.csv
  studentRegistration.csv
  studentVle.csv
  vle.csv

### 3️⃣ Installer les dépendances

Depuis la racine du projet :

pip install -r services/prepa-data/requirements.txt

### 4️⃣ Lancer le pipeline de nettoyage

python services/prepa-data/scripts/clean_oulad.py

Ce script :

1. Charge les CSV bruts depuis datasets/oulad/

2. Applique un nettoyage de base :

suppression des doublons

uniformisation des colonnes

gestion de certaines valeurs manquantes

3. Fusionne les tables clés (studentInfo + studentRegistration + agrégats VLE + assessments)

4. Crée un fichier fusionné :
datasets/oulad/cleaned/oulad_merged_feature_base.csv
5. Génère des rapports de profiling dans :
<img width="371" height="163" alt="Capture d&#39;écran 2025-11-16 113547" src="https://github.com/user-attachments/assets/d9ff2160-756f-4952-ac5b-07b108b97b40" />
6.Exemple d’arborescence attendue :
<img width="402" height="885" alt="Capture d&#39;écran 2025-11-16 113944" src="https://github.com/user-attachments/assets/75428716-a341-4159-9507-62960133302a" />


https://github.com/user-attachments/assets/4085e09f-1808-4763-b95f-01ca0bb6c2c8


