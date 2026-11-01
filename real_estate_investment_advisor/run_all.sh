#!/usr/bin/env bash
# Runs the whole pipeline end to end (about 5 minutes on a laptop).
set -e
cd "$(dirname "$0")"
python src/01_data_preprocessing.py   # clean + engineer features -> data/processed/housing_cleaned.csv
python src/02_eda.py                  # 23 charts + reports/eda_findings.md
python src/03_train_models.py         # train, compare, log to MLflow, save best models to models/
echo
echo "Done. Launch the app with:   streamlit run app/streamlit_app.py"
echo "Browse experiments with:     mlflow ui --backend-store-uri sqlite:///mlflow.db"
