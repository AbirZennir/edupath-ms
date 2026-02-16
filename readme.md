EduPath-MS

Microservices-based Learning Analytics platform combining Spring Boot, React, Flutter, and Machine Learning to predict student risk and generate academic insights.

EduPath-MS is a complete educational analytics and prediction platform powered by Machine Learning. It helps institutions identify at-risk students, analyze performance trends, and generate actionable academic insights through a modern multi-platform interface.

📋 Overview

EduPath-MS follows a microservices architecture composed of four main components:

Backend API – Spring Boot 3.5.7 (Java 17) REST service

Mobile App – Cross-platform Flutter application (iOS & Android)

Web App – Modern web interface built with React + Vite

ML Engine – Python FastAPI module for predictions and analytics

🏗️ Project Architecture
edupath-ms/
├── backend/              # Spring Boot REST API
├── frontendmobile/       # Flutter mobile application
├── frontendweb/          # React/Vite web application
└── nn/                   # Machine Learning module (Python)


🚀 Quick Start
Prerequisites

Java 17+

Node.js 18+

Python 3.9+

Flutter SDK 3.9.2+

Maven

npm

Installation & Run
Backend
cd backend
./mvnw spring-boot:run


Server runs on:
http://localhost:8080

Web Frontend
cd frontendweb
npm install
npm run dev


Application runs on:
http://localhost:5173

Mobile Frontend (Flutter)
cd frontendmobile
flutter pub get
flutter run

ML Engine (Python)
cd nn
pip install -r requirements.txt
python main.py


FastAPI runs on:
http://localhost:8000

📦 Main Dependencies
Backend (Spring Boot)

Spring Boot Web

Spring Data JPA

H2 Database (development)

Spring Boot DevTools

Mobile App (Flutter)

http: 1.2.2

shared_preferences: 2.3.2

url_launcher: 6.3.2

confetti: 0.8.0

file_picker: 10.3.8

cupertino_icons: 1.0.8

Web App (React)

React + Vite

State management & HTTP client

ML Engine (Python)

FastAPI ≥ 0.110.0 – Async web framework

Uvicorn ≥ 0.27.0 – ASGI server

scikit-learn ≥ 1.4.0 – ML algorithms

pandas ≥ 2.2.0 – Data manipulation

numpy ≥ 1.26.0 – Numerical computing

SQLAlchemy ≥ 2.0.0 – ORM

PyMySQL ≥ 1.1.0 – MySQL connector

Pydantic ≥ 2.6.0 – Data validation

🗂️ Detailed Structure
Backend (backend/)

src/main/java/ – Java source code

src/main/resources/ – Configuration files

src/test/ – Unit tests

pom.xml – Maven dependencies

Mobile Frontend (frontendmobile/)

lib/ – Flutter/Dart source code

main.dart – Entry point

api_client.dart – HTTP client

screens/ – Application screens

android/ – Android config

ios/ – iOS config

pubspec.yaml – Flutter dependencies

Web Frontend (frontendweb/)

src/ – React source code

components/ – Reusable components

api/ – API clients

styles/ – Styling files

index.html – HTML entry

vite.config.js – Vite configuration

ML Engine (nn/)

models/ – ML models & architectures

db_models.py

ml_model.py

services/

prediction_service.py

analysis_service.py

schemas/ – Pydantic schemas

utils/ – Utilities

datasets/ – Training datasets

main.py – FastAPI entry point

🔌 API Endpoints

Main backend endpoints:

/api/students – Student management

/api/predictions – ML predictions

/api/analysis – Academic analytics

🤖 Machine Learning Module

The nn/ module uses scikit-learn algorithms to:

Predict student dropout risk

Analyze academic performance

Recommend targeted interventions

Services

PredictionService – Generates risk predictions

AnalysisService – Performs pedagogical analysis

📱 Key Features

✅ Real-time ML predictions

✅ Web & mobile multi-channel interface

✅ Secure authentication

✅ Academic reports & analytics dashboard

✅ Risk scoring system

✅ Multi-platform mobile support

🛠 Configuration

Create a .env file at the root:

# Backend
spring.datasource.url=jdbc:h2:mem:testdb
spring.datasource.username=sa
spring.datasource.password=

# ML Engine
DATABASE_URL=mysql://user:password@localhost/edupath
API_PORT=8000

📝 Build & Deployment
Backend
cd backend
./mvnw clean package

Mobile
cd frontendmobile
flutter build apk
flutter build ios
flutter build web

Web
cd frontendweb
npm run build

🧪 Tests
Backend
cd backend
./mvnw test

Mobile
cd frontendmobile
flutter test

📚 Resources

Web Design (Figma)

Spring Boot Documentation

Flutter Documentation

FastAPI Documentation# EduPath-MS

Une plateforme complète d'analyse et de prédiction pédagogique utilisant le machine learning. EduPath aide à identifier et guider les étudiants à risque grâce à des algorithmes d'apprentissage automatique et une interface utilisateur conviviale.

## 📋 Vue d'ensemble

EduPath-MS est une architecture microservices composée de quatre composants principaux:

- **Backend API** - Service REST Spring Boot 3.5.7 (Java 17)
- **Mobile App** - Application Flutter multiplateformes (iOS, Android)
- **Web App** - Interface web moderne avec Vite + React
- **ML Engine** - Module Python FastAPI pour prédictions et analyses

## 🏗️ Architecture du projet

```
edupath-ms/
├── backend/              # API REST Spring Boot
├── frontendmobile/       # Application Flutter
├── frontendweb/          # Application React/Vite
└── nn/                   # Module Machine Learning (Python)
```
![architecture](https://github.com/user-attachments/assets/585c8643-6e75-49e9-b661-4d5ff5dc685e)

## 🚀 Démarrage rapide

### Prérequis

- Java 17+
- Node.js 18+
- Python 3.9+
- Flutter SDK 3.9.2+
- Maven
- npm

### Installation et lancement

#### Backend

```bash
cd backend
./mvnw spring-boot:run
```

Le serveur démarre sur `http://localhost:8080`

#### Frontend Web

```bash
cd frontendweb
npm install
npm run dev
```

L'application démarre sur `http://localhost:5173`

#### Frontend Mobile (Flutter)

```bash
cd frontendmobile
flutter pub get
flutter run
```

#### ML Engine (Python)

```bash
cd nn
pip install -r requirements.txt
python main.py
```

Le service FastAPI démarre sur `http://localhost:8000`

## 📦 Dépendances principales

### Backend (Spring Boot)
- Spring Boot Web
- Spring Data JPA
- H2 Database (développement)
- Spring Boot DevTools

### Frontend Mobile (Flutter)
- http: 1.2.2
- shared_preferences: 2.3.2
- url_launcher: 6.3.2
- confetti: 0.8.0
- file_picker: 10.3.8
- cupertino_icons: 1.0.8

### Frontend Web (React)
- React avec Vite
- Gestion d'état et requêtes HTTP

### ML Engine (Python)
- **FastAPI** ≥ 0.110.0 - Framework web asynchrone
- **Uvicorn** ≥ 0.27.0 - Serveur ASGI
- **scikit-learn** ≥ 1.4.0 - Algorithmes ML
- **pandas** ≥ 2.2.0 - Manipulation de données
- **numpy** ≥ 1.26.0 - Calculs numériques
- **SQLAlchemy** ≥ 2.0.0 - ORM
- **PyMySQL** ≥ 1.1.0 - Connecteur MySQL
- **Pydantic** ≥ 2.6.0 - Validation de données

## 🗂️ Structure détaillée

### Backend (`backend/`)
- `src/main/java/` - Code source Java
- `src/main/resources/` - Configuration (application.properties)
- `src/test/` - Tests unitaires
- `pom.xml` - Gestion des dépendances Maven

### Frontend Mobile (`frontendmobile/`)
- `lib/` - Code source Dart/Flutter
  - `main.dart` - Point d'entrée
  - `api_client.dart` - Client HTTP
  - `screens/` - Écrans de l'app
- `android/` - Configuration Android
- `ios/` - Configuration iOS
- `pubspec.yaml` - Dépendances Flutter

### Frontend Web (`frontendweb/`)
- `src/` - Code source React
  - `components/` - Composants réutilisables
  - `api/` - Clients API
  - `styles/` - Feuilles de style
- `index.html` - Point d'entrée HTML
- `vite.config.js` - Configuration Vite

### ML Engine (`nn/`)
- `models/` - Modèles et architectures ML
  - `db_models.py` - Modèles de base de données
  - `ml_model.py` - Modèles d'apprentissage
- `services/` - Services applicatifs
  - `prediction_service.py` - Service de prédiction
  - `analysis_service.py` - Service d'analyse
- `schemas/` - Schémas Pydantic
- `utils/` - Utilitaires
- `datasets/` - Données d'entraînement
- `main.py` - Point d'entrée FastAPI

## 🔌 API Endpoints

Le backend expose les endpoints REST suivants (consulter [application.properties](backend/src/main/resources/application.properties) pour la configuration):

- `/api/students` - Gestion des étudiants
- `/api/predictions` - Prédictions ML
- `/api/analysis` - Analyses pédagogiques

## 🤖 Module Machine Learning

Le module `nn/` utilise des algorithmes de scikit-learn pour:
- Prédire le risque d'abandon
- Analyser les performances
- Recommander des interventions

### Services disponibles:
- **PredictionService** - Génère les prédictions
- **AnalysisService** - Analyse les données pédagogiques

## 📱 Fonctionnalités clés

- ✅ Prédictions ML en temps réel
- ✅ Interface multichannel (web + mobile)
- ✅ Synchronisation des données
- ✅ Authentification sécurisée
- ✅ Rapports d'analyse détaillés
- ✅ Support multi-plateforme mobile

## 🛠️ Configuration

### Variables d'environnement

Créer un fichier `.env` à la racine ou configurer:

```properties
# Backend
spring.datasource.url=jdbc:h2:mem:testdb
spring.datasource.username=sa
spring.datasource.password=

# ML Engine
DATABASE_URL=mysql://user:password@localhost/edupath
API_PORT=8000
```

## 📝 Build et Déploiement

### Build du Backend
```bash
cd backend
./mvnw clean package
```

### Build de l'app mobile Flutter
```bash
cd frontendmobile
flutter build apk          # Android
flutter build ios          # iOS
flutter build web          # Web version
```

### Build du Frontend Web
```bash
cd frontendweb
npm run build
```

## 🧪 Tests

### Backend
```bash
cd backend
./mvnw test
```

### Frontend Mobile
```bash
cd frontendmobile
flutter test
```

## 📚 Ressources

- [Design Web Figma](https://www.figma.com/design/hb6gPr7JfdG5wQEmEE0op3/Web-App-Design-for-EduPath-MS)
- [Documentation Spring Boot](https://spring.io/projects/spring-boot)
- [Documentation Flutter](https://flutter.dev/docs)
- [Documentation FastAPI](https://fastapi.tiangolo.com/)
  
# Capture d'ecran

## application web

<img width="1908" height="1119" alt="web_login" src="https://github.com/user-attachments/assets/cb3f49a3-7248-4a20-8b39-a604147d482b" />
<img width="1908" height="1582" alt="web_register" src="https://github.com/user-attachments/assets/abe911bd-3dce-4095-9ced-a4aae8fd6b17" />
<img width="1908" height="1020" alt="web_reset_password" src="https://github.com/user-attachments/assets/7d28323d-29e9-438b-aaa8-3d66c4d449ba" />
<img width="1908" height="2784" alt="web_analytics_rapports" src="https://github.com/user-attachments/assets/46726879-4c5e-466a-b571-4eff362972d5" />
<img width="1908" height="868" alt="web_class_details" src="https://github.com/user-attachments/assets/7b5fc2ea-6502-4a88-ad5b-47a8481aaf3f" />
<img width="1908" height="926" alt="web_class_history" src="https://github.com/user-attachments/assets/caffa0b8-5b36-46f9-8f40-96b5f91d6c70" />
<img width="1908" height="926" alt="web_class_history" src="https://github.com/user-attachments/assets/049d28cc-db20-4b26-b5a3-edb2b1070aae" />
<img width="1908" height="1463" alt="web_class_students" src="https://github.com/user-attachments/assets/c2a088e9-4765-4ccd-87d2-eb3092c1bd7d" />
<img width="1908" height="1767" alt="web_classes" src="https://github.com/user-attachments/assets/02fe40ec-aa2f-4d87-83e6-23bc6d97a411" />
<img width="1908" height="1992" alt="web_dashboard" src="https://github.com/user-attachments/assets/47999123-72d0-4933-b610-c0bc0c30aef6" />
<img width="1908" height="1304" alt="web_parametre_integrationLMS" src="https://github.com/user-attachments/assets/78f68cd1-c1d5-4189-9b88-be64c222ab22" />
<img width="1908" height="1314" alt="web_parametre_notifications" src="https://github.com/user-attachments/assets/9554ba44-cb43-4e4c-bc5b-c8057b34eb88" />
<img width="1908" height="1537" alt="web_parametre_profile" src="https://github.com/user-attachments/assets/6aaf8590-e335-48d3-bbae-7a1e3bee8c2f" />
<img width="1908" height="3243" alt="web_recommendations" src="https://github.com/user-attachments/assets/d755f1e3-1e64-48c0-8907-5014cc55d334" />

## application mobile

<img width="441" height="590" alt="mobile_login" src="https://github.com/user-attachments/assets/ca1137c4-f1b0-4ecd-8277-92bbad64bd98" />
<img width="487" height="560" alt="mobile_register" src="https://github.com/user-attachments/assets/1443a5e0-53ac-43d1-a4d4-14cc01222d25" />
<img width="463" height="326" alt="mobile_reset_password" src="https://github.com/user-attachments/assets/6a892176-d253-4c9d-9b15-bdc5e65ad69e" />
<img width="455" height="673" alt="mobile_cours" src="https://github.com/user-attachments/assets/87a3f6e5-2d5e-4133-a17f-fb62ca6594b7" />
<img width="453" height="713" alt="mobile_dashboard" src="https://github.com/user-attachments/assets/441e283d-7d8c-4ddb-bbad-9b94b8978d98" />
<img width="474" height="658" alt="mobile_homework_all" src="https://github.com/user-attachments/assets/9a79e052-411e-496b-ae84-3d6c683d6c81" />
<img width="459" height="666" alt="mobile_homework_given" src="https://github.com/user-attachments/assets/9103dedc-7ef2-4d38-aaa5-d0c6f6b716c7" />
<img width="464" height="667" alt="mobile_homework_to_give" src="https://github.com/user-attachments/assets/188fcae8-cb91-46f2-8875-24f0086f5c35" />
<img width="463" height="740" alt="mobile_notes" src="https://github.com/user-attachments/assets/b27bb5bc-5a35-42a1-a4fc-27140b97ffde" />
<img width="464" height="764" alt="mobile_profil" src="https://github.com/user-attachments/assets/af69068f-7b4e-4c74-a279-6915ccfe8053" />


https://github.com/user-attachments/assets/0e6051b8-c2e4-4991-a6ab-8fc7bae3b70c





https://github.com/user-attachments/assets/49dc7658-7f77-4e14-a6c8-aac611c5768b

