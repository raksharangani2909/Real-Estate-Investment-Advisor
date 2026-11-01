"""Cleaning + feature engineering shared by the training scripts and the app."""
import numpy as np
import pandas as pd

import config as C

NUM_COLS = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
            "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals"]
CAT_COLS = ["State", "City", "Locality", "Property_Type", "Furnished_Status",
            "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities",
            "Facing", "Owner_Type", "Availability_Status"]


def iqr_flag(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return ((s < q1 - k * iqr) | (s > q3 + k * iqr)).astype(int)


# ---------------------------------------------------------------------------
# 1. CLEANING
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame):
    """Return (clean_df, report). Handles types, missing values, duplicates,
    inconsistent values and flags outliers."""
    rep = {"rows_raw": int(len(df))}
    df = df.copy()

    # whitespace / type clean-up
    for c in CAT_COLS:
        df[c] = df[c].astype("string").str.strip()
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # missing values: drop rows missing the essentials, impute the rest
    rep["missing_before"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    df = df.dropna(subset=["Price_in_Lakhs", "Size_in_SqFt", "City"])
    for c in NUM_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in CAT_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    df["Amenities"] = df["Amenities"].fillna("")

    # duplicates (ignoring the ID column)
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"])
    rep["duplicates_removed"] = int(before - len(df))

    # invalid values
    before = len(df)
    df = df[(df.Price_in_Lakhs > 0) & (df.Size_in_SqFt > 0) & (df.BHK > 0)]
    df = df[df.Year_Built <= C.CURRENT_YEAR]
    rep["invalid_rows_removed"] = int(before - len(df))

    # Age is recomputed so it is always consistent with Year_Built
    df["Age_of_Property"] = C.CURRENT_YEAR - df["Year_Built"]

    # Floor_No can never exceed Total_Floors -> cap it and keep a flag
    bad = df.Floor_No > df.Total_Floors
    df["Floor_Fixed_Flag"] = bad.astype(int)
    df.loc[bad, "Floor_No"] = df.loc[bad, "Total_Floors"]
    rep["floor_no_capped_to_total_floors"] = int(bad.sum())

    # The raw Price_per_SqFt is in LAKHS per sq ft (values like 0.05) -> recompute in INR
    df["Price_per_SqFt_INR"] = (df.Price_in_Lakhs * 1e5 / df.Size_in_SqFt).round(2)
    df = df.drop(columns=["Price_per_SqFt"])

    # outlier flags (kept, not deleted: they are legitimate small/expensive homes)
    for col, name in [("Price_in_Lakhs", "Price"), ("Size_in_SqFt", "Size"),
                      ("Price_per_SqFt_INR", "PPSF")]:
        df[f"{name}_Outlier"] = iqr_flag(df[col])
        rep[f"outliers_{name}"] = int(df[f"{name}_Outlier"].sum())

    rep["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), rep


# ---------------------------------------------------------------------------
# 2. REFERENCE TABLES (needed to engineer features for a single new property)
# ---------------------------------------------------------------------------
def compute_reference(df: pd.DataFrame) -> dict:
    g = df.groupby("City")
    return {
        "city_median_ppsf": g.Price_per_SqFt_INR.median().round(2).to_dict(),
        "city_avg_schools": g.Nearby_Schools.mean().round(3).to_dict(),
        "city_avg_price": g.Price_in_Lakhs.mean().round(2).to_dict(),
        "city_state": g.State.first().to_dict(),
        "city_localities": {c: sorted(x.Locality.unique().tolist(),
                                      key=lambda s: int(s.split("_")[-1]))
                            for c, x in g},
    }


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def engineer(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    df = df.copy()
    df["Transport_Score"] = df.Public_Transport_Accessibility.map(C.TRANSPORT_MAP)
    df["Parking_Flag"] = (df.Parking_Space == "Yes").astype(int)
    df["Security_Flag"] = (df.Security == "Yes").astype(int)
    df["Under_Construction_Flag"] = (df.Availability_Status == "Under_Construction").astype(int)

    for a in C.AMENITIES:
        df[a] = df.Amenities.str.contains(a, regex=False).astype(int)
    df["Amenity_Count"] = df[C.AMENITIES].sum(axis=1)

    df["Floor_Ratio"] = (df.Floor_No / df.Total_Floors).round(4)
    df["Size_per_BHK"] = (df.Size_in_SqFt / df.BHK).round(1)

    # Infrastructure score 0-1: schools, hospitals and transport equally weighted
    df["Infrastructure_Score"] = ((df.Nearby_Schools / 10 + df.Nearby_Hospitals / 10
                                   + (df.Transport_Score - 1) / 2) / 3).round(4)
    # School density score: schools relative to the city average (1.0 = city average)
    df["School_Density_Score"] = (df.Nearby_Schools / df.City.map(ref["city_avg_schools"])).round(4)

    med = df.City.map(ref["city_median_ppsf"])
    df["City_Median_PPSF"] = med
    df["PPSF_vs_City_Median"] = (df.Price_per_SqFt_INR / med).round(4)
    return df


# ---------------------------------------------------------------------------
# 4. DOMAIN RULES -> TARGETS
# ---------------------------------------------------------------------------
def annual_growth_rate(df: pd.DataFrame) -> pd.Series:
    """Rule-based expected appreciation (% / year). See config.GROWTH."""
    G = C.GROWTH
    g = (G["base"]
         + df.Public_Transport_Accessibility.map(G["transport_bonus"])
         + G["infra_max"] * (df.Nearby_Schools + df.Nearby_Hospitals) / 20
         + G["under_construction"] * df.Under_Construction_Flag
         + G["undervalued"] * (df.PPSF_vs_City_Median <= 1.0)
         + G["security"] * df.Security_Flag
         + G["parking"] * df.Parking_Flag
         + G["per_amenity"] * df.Amenity_Count
         - G["age_penalty_max"] * df.Age_of_Property / 35)
    return g.clip(G["min"], G["max"]).round(3)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Annual_Growth_Rate_Pct"] = annual_growth_rate(df)
    df[C.REG_TARGET] = (df.Price_in_Lakhs
                        * (1 + df.Annual_Growth_Rate_Pct / 100) ** C.FORECAST_YEARS).round(2)
    df[C.CLF_TARGET] = ((df.Annual_Growth_Rate_Pct >= C.GOOD_GROWTH_THRESHOLD)
                        & (df.PPSF_vs_City_Median <= C.MAX_PPSF_VS_CITY)).astype(int)
    return df


# ---------------------------------------------------------------------------
# 5. SINGLE-PROPERTY INPUT (used by the Streamlit app)
# ---------------------------------------------------------------------------
def frame_from_user_input(u: dict, ref: dict) -> pd.DataFrame:
    """u: dict with the raw property fields entered in the app."""
    row = {
        "ID": 0, "State": ref["city_state"][u["City"]], "City": u["City"],
        "Locality": u.get("Locality", ""), "Property_Type": u["Property_Type"],
        "BHK": u["BHK"], "Size_in_SqFt": u["Size_in_SqFt"], "Price_in_Lakhs": u["Price_in_Lakhs"],
        "Year_Built": u["Year_Built"], "Furnished_Status": u["Furnished_Status"],
        "Floor_No": min(u["Floor_No"], u["Total_Floors"]), "Total_Floors": u["Total_Floors"],
        "Age_of_Property": C.CURRENT_YEAR - u["Year_Built"],
        "Nearby_Schools": u["Nearby_Schools"], "Nearby_Hospitals": u["Nearby_Hospitals"],
        "Public_Transport_Accessibility": u["Public_Transport_Accessibility"],
        "Parking_Space": u["Parking_Space"], "Security": u["Security"],
        "Amenities": ", ".join(u["Amenities"]), "Facing": u["Facing"],
        "Owner_Type": u["Owner_Type"], "Availability_Status": u["Availability_Status"],
    }
    df = pd.DataFrame([row])
    df["Price_per_SqFt_INR"] = df.Price_in_Lakhs * 1e5 / df.Size_in_SqFt
    return add_targets(engineer(df, ref))
