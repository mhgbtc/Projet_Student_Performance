# Student Performance Prediction

## Problématique

Ce projet vise à prédire les performances académiques des étudiants afin d'identifier ceux à risque et d'aider les éducateurs à intervenir de manière proactive.

## Dataset

- Source : Kaggle - Student Performance Factors
- 6607 étudiants
- 20 variables (19 features + 1 cible)
- Variables : habitudes d'étude, assiduité, facteurs socio-économiques, etc.

## Démarche

### 1. Analyse Exploratoire (EDA)
- Nettoyage des données
- Analyse des distributions
- Détection d'outliers
- Étude des corrélations

**Insights clés :**
- Attendance (0.58) et Hours_Studied (0.45) sont les facteurs les plus corrélés
- Variables socio-économiques ont un impact limité
- Distribution de la cible concentrée entre 65-70

### 2. Preprocessing
- Imputation des valeurs manquantes (mode)
- Encodage ordinal pour variables à ordre naturel
- One-hot encoding pour variables nominales
- Correction des outliers (score > 100)

### 3. Modélisation
**Modèles testés :**
- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest
- Gradient Boosting

**Meilleur modèle : Ridge Regression**
- R² = 0.7709 (77% de variance expliquée)
- RMSE = 1.87 points
- MAE = 1.47 points
- Pas d'overfitting (Train ≈ Test)

## Résultats

Le modèle Ridge optimisé permet de prédire les scores d'examen avec une erreur moyenne de 1.47 points, soit une amélioration de 52% par rapport au baseline.

## Recommandations

1. Prioriser l'assiduité (Attendance) - facteur #1
2. Encourager le temps d'étude
3. Mettre en place un système d'alerte précoce basé sur les prédictions

## Structure du projet
```
student-performance/
├── data/
│   └── raw/
│       └── StudentPerformanceFactors.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Modeling.ipynb
├── src/
│   └── preprocessing.py
├── models/
│   ├── final_model.pkl
│   └── model_metrics.csv
├── README.md
└── requirements.txt
```

## Installation et utilisation
```bash
pip install -r requirements.txt
jupyter notebook
```

## Auteurs

ADIGBONON Mahoutondji Thérèse Rodica
CAMARA Moussa
DIOP Seynabou Mbayé Ba Souna
DJIDOHOKPIN Samuel

## Date

Février 2026