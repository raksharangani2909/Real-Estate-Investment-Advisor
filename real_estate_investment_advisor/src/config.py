"""Central configuration for the Real Estate Investment Advisor project.

Every business rule (growth-rate model, 'Good Investment' threshold, etc.) lives
here so it can be reviewed and changed in ONE place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "india_housing_prices.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "housing_cleaned.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# The dataset stores Age_of_Property = 2025 - Year_Built (verified on all rows).
CURRENT_YEAR = 2025
FORECAST_YEARS = 5

AMENITIES = ["Clubhouse", "Gym", "Garden", "Pool", "Playground"]
TRANSPORT_MAP = {"Low": 1, "Medium": 2, "High": 3}

# ---------------------------------------------------------------------------
# Domain rule: annual appreciation model (percent per year)
# The raw data has NO future price, so the 5-year target is built from this
# transparent, editable rule (documented in docs/PROJECT_REPORT.md).
# ---------------------------------------------------------------------------
GROWTH = {
    "base": 5.0,                  # baseline market appreciation
    "transport_bonus": {"Low": 0.0, "Medium": 1.0, "High": 2.0},
    "infra_max": 2.0,             # scaled by (schools + hospitals) / 20
    "under_construction": 1.0,    # pre-completion appreciation
    "undervalued": 1.0,           # price/sqft <= city median (catch-up potential)
    "security": 0.3,
    "parking": 0.3,
    "per_amenity": 0.2,           # x number of amenities (0-5)
    "age_penalty_max": 1.5,       # scaled by age / 35
    "min": 2.0,
    "max": 15.0,
}

# 'Good Investment' = appreciation >= threshold AND not clearly over-priced
GOOD_GROWTH_THRESHOLD = 8.5       # % per year
MAX_PPSF_VS_CITY = 1.20           # price/sqft may be at most 20% above city median

# ---------------------------------------------------------------------------
# Model feature lists (shared by training and the Streamlit app)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_INR",
    "Age_of_Property", "Floor_No", "Total_Floors", "Floor_Ratio",
    "Nearby_Schools", "Nearby_Hospitals", "Transport_Score",
    "Parking_Flag", "Security_Flag", "Under_Construction_Flag",
    "Amenity_Count", "Clubhouse", "Gym", "Garden", "Pool", "Playground",
    "Size_per_BHK", "Infrastructure_Score", "School_Density_Score",
    "PPSF_vs_City_Median",
]
CATEGORICAL_FEATURES = ["City", "Property_Type", "Furnished_Status", "Facing", "Owner_Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLF_TARGET = "Good_Investment"
REG_TARGET = "Future_Price_5Y_Lakhs"
