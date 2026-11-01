"""STEP 1 - Data preprocessing & feature engineering.

Run:  python src/01_data_preprocessing.py
Out:  data/processed/housing_cleaned.csv
      models/reference.json               (city lookup tables used by the app)
      reports/data_quality_report.json
"""
import json

import pandas as pd

import config as C
import features as F


def main():
    raw = pd.read_csv(C.RAW_CSV)
    print(f"Loaded raw data: {raw.shape}")

    df, rep = F.clean(raw)
    ref = F.compute_reference(df)
    df = F.add_targets(F.engineer(df, ref))

    # Final column order: original fields first, then engineered fields, then targets
    C.CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_CSV, index=False)

    C.MODELS_DIR.mkdir(exist_ok=True)
    (C.MODELS_DIR / "reference.json").write_text(json.dumps(ref, indent=1))

    rep["rows_final"] = int(len(df))
    rep["good_investment_share"] = round(float(df[C.CLF_TARGET].mean()), 4)
    rep["growth_rate_summary_pct"] = df.Annual_Growth_Rate_Pct.describe().round(3).to_dict()
    rep["engineered_columns"] = [c for c in df.columns if c not in raw.columns]
    C.REPORTS_DIR.mkdir(exist_ok=True)
    (C.REPORTS_DIR / "data_quality_report.json").write_text(json.dumps(rep, indent=2))

    print(json.dumps({k: v for k, v in rep.items() if k != "engineered_columns"}, indent=2))
    print(f"Saved {C.CLEAN_CSV}  shape={df.shape}")


if __name__ == "__main__":
    main()
