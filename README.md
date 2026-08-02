# AUTOML-Python-
# AutoML Engine

An end-to-end AutoML pipeline that automatically preprocesses tabular datasets,
trains and tunes multiple classical machine learning models, and surfaces the
best-performing model through an interactive web dashboard — no manual model
selection required.

## Problem
Choosing and tuning the right ML model for a dataset is time-consuming and
requires expertise most beginners/analysts don't have. This project automates
that entire workflow.

## Features
- Upload any tabular dataset (CSV)
- Automatic preprocessing (missing values, encoding, scaling)
- Trains multiple classical models: Logistic Regression, Random Forest, SVM,
  XGBoost, LightGBM, KNN
- Automated hyperparameter tuning via Optuna
- Model leaderboard with accuracy/F1/training-time comparison
- Download the best trained model + a generated performance report

## Tech Stack
- **Backend:** Python, FastAPI, scikit-learn, XGBoost, LightGBM, Optuna
- **Database:** SQLite (dev) / PostgreSQL (prod)
- **Frontend:** React (or Streamlit)
- **Other:** Docker, joblib

## Architecture
CSV Upload → Preprocessing → Model Training (parallel) → Hyperparameter
Tuning → Evaluation → Leaderboard → Best Model Export

## Setup

```bash
git clone https://github.com/<your-username>/automl-engine.git
cd automl-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## Usage
1. Start the backend server
2. Upload a CSV via `/upload` endpoint (or dashboard once built)
3. Trigger training via `/train`
4. View leaderboard and download the best model via `/results`

## Project Structure
*(paste the folder tree here)*

## Team
| Name | Role |
|---|---|
| Gaurav | ML Core / Backend Lead |
| ... | ... |
| ... | ... |

## License
MIT
