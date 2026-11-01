# Real Estate Investment Advisor: Predicting Property Profitability & Future Value

A machine-learning application that helps investors decide whether a property is a **Good Investment**
(classification) and estimates its **price after 5 years** (regression). Built on the supplied
`india_housing_prices.csv` (250,000 listings, 23 columns, 20 states, 42 cities) - no other data is used.

**Stack:** Python, Pandas, Scikit-learn, XGBoost, MLflow, Streamlit, Plotly.

## Project layout

```
real_estate_investment_advisor/
├── data/
│   ├── raw/india_housing_prices.csv        your original dataset (unchanged)
│   └── processed/housing_cleaned.csv       cleaned + engineered dataset with both targets
├── src/
│   ├── config.py                           all business rules & feature lists (edit rules HERE)
│   ├── features.py                         cleaning, feature engineering, target rules
│   ├── 01_data_preprocessing.py            Step 1 - preprocessing
│   ├── 02_eda.py                           Step 2 - the 20 EDA questions (+3 extras)
│   └── 03_train_models.py                  Step 3 - training, evaluation, MLflow tracking
├── app/streamlit_app.py                    Step 4 - Streamlit application
├── models/                                 classifier.joblib, regressor.joblib, reference.json, importances
├── reports/
│   ├── figures/                            27 PNG charts
│   ├── eda_findings.md                     written answer to every EDA question
│   ├── model_metrics.json, model_comparison_*.csv, data_quality_report.json, training_log.txt
├── docs/PROJECT_REPORT.md (+ .docx)        methodology, findings, use cases, limitations
├── requirements.txt
└── run_all.sh
```

## Quick start

```bash
pip install -r requirements.txt
# Everything is already generated, so you can launch the app straight away:
streamlit run app/streamlit_app.py

# To regenerate everything from the raw CSV (~5 min):
bash run_all.sh                                       # or run the 3 scripts in src/ one by one
mlflow ui --backend-store-uri sqlite:///mlflow.db     # experiment tracking UI -> http://127.0.0.1:5000
```
Run all commands from the project root folder.

## What the app does

| Tab | Content |
|---|---|
| Investment Advisor | Property form -> "Is this a Good Investment?" with probability gauge, estimated price after 5 years with trend line, key facts, feature importance |
| Property Explorer | Filter all 250k properties by state, city, BHK, type, price, area, availability; download results |
| Market Insights | Location-wise heatmaps (choose metric and dimension), city price trends, top cities by good-investment share |
| Model Performance | Metrics for all models, confusion matrix, actual vs predicted, and an interpretation note |

## Results (held-out 20% test set = 50,000 properties)

| Task | Best model | Key metrics | Baseline |
|---|---|---|---|
| Classification (Good Investment) | XGBoost | Accuracy 0.9964, F1 0.9942, ROC-AUC 0.9999 | majority class: 0.689 accuracy |
| Regression (price in 5 yrs, lakhs) | XGBoost | RMSE 1.78, MAE 1.32, R² 0.9999 | "no growth": RMSE 138.6 |

All six models (LogReg/RF/XGBoost and Ridge/RF/XGBoost) are logged to MLflow with parameters, metrics,
feature importances and the model artifact; the winners are registered in the MLflow Model Registry.

## IMPORTANT - please read before presenting these numbers

1. **The dataset has no "price after 5 years" column and no "good investment" label.** Both targets were built from
   the documented domain rule in `src/config.py` (expected annual appreciation from transport, infrastructure,
   new-build status, value vs city median, security, parking, amenities, and property age; "Good" = growth >= 8.5%/yr
   and price/sq ft <= 1.2x the city median). 31.1% of properties are labelled Good.
2. Because the targets are a function of the same features, the near-perfect scores show that the models **learned the
   rule**. They do not show that the models can predict the real market. State this in your viva/report.
3. The original columns are statistically independent of price (all pairwise |r| < 0.01) - typical of synthetic data -
   so most EDA answers are "no meaningful relationship". These are genuine findings about this dataset.
4. The brief mentions *crime rate*; the dataset has no such column. The **Security** flag is used as the safety proxy.
5. The dataset has no dates, so "price trend" charts use **construction period** as the time axis.

To use a different business rule, change the numbers in `config.py` (`GROWTH`, `GOOD_GROWTH_THRESHOLD`,
`MAX_PPSF_VS_CITY`) and re-run `run_all.sh`.

## Data cleaning decisions

- No missing values or duplicates were found (the code still handles both: median/mode imputation, de-duplication ignoring ID).
- `Price_per_SqFt` in the raw file is in **lakhs** per sq ft (values like 0.05). It was recomputed in INR/sq ft (`Price_per_SqFt_INR`).
- 116,304 rows (46.5%) had `Floor_No` greater than `Total_Floors` (impossible). These were capped at the top floor and flagged (`Floor_Fixed_Flag`).
- 19,723 price/sq ft outliers (IQR rule) are flagged, not deleted - they are valid small, expensive homes.
- Scaling (StandardScaler) and one-hot encoding happen inside the model pipeline, so the saved models accept raw property inputs.

## Deploying on Streamlit Cloud

Push the folder to GitHub (the cleaned CSV is ~62 MB, below GitHub's 100 MB limit), then create an app from
`app/streamlit_app.py`. Keep `data/processed/housing_cleaned.csv`, `models/` and `reports/` in the repository.
