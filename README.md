# Viral Outbreak Early Warning System

> End-to-end machine learning pipeline for synthetic epidemiological surveillance and early outbreak prediction using spatio-temporal simulation, environmental variables and explainable AI.

---

## Project Overview

This project presents a complete epidemiological surveillance system capable of predicting regional viral outbreak risk **7 days in advance** using synthetic epidemiological data.

The dataset is generated through a stochastic spatial simulation that incorporates:

- Regional disease transmission
- Human mobility
- Vaccination dynamics
- Environmental conditions
- Healthcare system pressure
- Imported cases
- Public health interventions

The resulting dataset is then used to train multiple machine learning models capable of identifying future outbreak risk.

---

## Motivation

Early detection of infectious disease outbreaks is essential for public health decision-making.

Real surveillance data are often:

- confidential
- incomplete
- geographically limited

This project generates a realistic synthetic dataset that reproduces many epidemiological processes while remaining completely reproducible.

---

# Project Pipeline

```
Synthetic Epidemic Simulation
            │
            ▼
Generated Epidemiological Dataset
            │
            ▼
Feature Engineering
            │
            ▼
Machine Learning Models
            │
            ▼
Model Evaluation
            │
            ▼
Explainability (SHAP + Feature Importance)
            │
            ▼
Figures & Report
```

---

# Repository Structure

```
Viral-Outbreak-Early-Warning-System/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── simulate_data.py
│   ├── train_model.py
│   ├── visualize_results.py
│   └── explainability.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Dataset

The synthetic dataset includes variables describing:

### Epidemiological

- Reported cases
- True infections
- Effective reproduction number (Rt)
- Effective transmission rate
- Spatial transmission pressure
- Imported infections

### Environmental

- Temperature
- Humidity
- Rainfall

### Demographic

- Population
- Population density

### Healthcare

- Healthcare capacity
- Healthcare pressure
- Testing rate

### Mobility

- Mobility index
- Vaccination rate
- Intervention status

### Temporal Features

- Lagged cases
- Rolling averages
- Growth rates

### Prediction Target

```
outbreak_7d
```

Binary prediction indicating whether the region is expected to experience an outbreak within the next seven days.

---

# Machine Learning Models

Three different algorithms are evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting

The best-performing model is automatically saved for later use.

---

# Model Evaluation

Performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Precision-Recall AUC

A temporal train/test split is used to avoid data leakage.

---

# Explainable AI

Model interpretation includes:

- Permutation Importance
- Model Feature Importance
- SHAP Summary Plot

These analyses help identify the variables contributing most strongly to outbreak prediction.

---

# Generated Figures

Running the pipeline automatically generates:

```
reports/figures/

outbreak_distribution.png

reported_cases_over_time.png

regional_case_trajectories.png

feature_correlation_matrix.png

model_performance.png

model_ranking_pr_auc.png

permutation_importance.png

model_feature_importance.png

shap_summary.png
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/your_username/Viral-Outbreak-Early-Warning-System.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Generate the dataset

```bash
python src/simulate_data.py
```

Train the models

```bash
python src/train_model.py
```

Generate visualizations

```bash
python src/visualize_results.py
```

Run explainability analysis

```bash
python src/explainability.py
```

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SHAP
- Git
- GitHub

---

# Future Improvements

Possible future extensions include:

- Graph Neural Networks
- XGBoost
- LightGBM
- Bayesian epidemiological models
- Real-world public datasets
- Docker deployment
- Interactive dashboard with Streamlit
- Azure Machine Learning deployment

---

# Disclaimer

The dataset generated in this project is entirely synthetic.

It has been designed exclusively for educational purposes and machine learning experimentation and should not be interpreted as representing any real-world epidemiological event.

---

# Author

**David Lafuente Pérez**

Biotechnology Graduate | MSc in Virology

Interested in:

- Data Science
- Machine Learning
- Bioinformatics
- Epidemiology
- Healthcare Analytics
