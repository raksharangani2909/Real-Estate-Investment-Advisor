"""Real Estate Investment Advisor - Streamlit app.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
import features as F

BLUE, ORANGE = "#2a78d6", "#eb6834"
st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

# ----------------------------------------------------------------------------- loaders
@st.cache_resource
def load_models():
    return joblib.load(C.MODELS_DIR / "classifier.joblib"), joblib.load(C.MODELS_DIR / "regressor.joblib")


@st.cache_data
def load_reference():
    return json.loads((C.MODELS_DIR / "reference.json").read_text())


@st.cache_data
def load_metrics():
    return json.loads((C.REPORTS_DIR / "model_metrics.json").read_text())


@st.cache_data
def load_data():
    cols = ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
            "Price_per_SqFt_INR", "Year_Built", "Age_of_Property", "Furnished_Status", "Availability_Status",
            "Public_Transport_Accessibility", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count",
            "Infrastructure_Score", "Annual_Growth_Rate_Pct", C.REG_TARGET, C.CLF_TARGET]
    df = pd.read_csv(C.CLEAN_CSV, usecols=cols)
    df["Build_Period"] = pd.cut(df.Year_Built, [1989, 1995, 2000, 2005, 2010, 2015, 2020, 2023],
                                labels=["1990-95", "1996-00", "2001-05", "2006-10", "2011-15", "2016-20", "2021-23"])
    return df


for f in (C.MODELS_DIR / "classifier.joblib", C.MODELS_DIR / "regressor.joblib", C.CLEAN_CSV):
    if not f.exists():
        st.error(f"Missing `{f.relative_to(ROOT)}`. Run the pipeline first: "
                 "`python src/01_data_preprocessing.py` then `python src/03_train_models.py`.")
        st.stop()

clf, reg = load_models()
ref = load_reference()
metrics = load_metrics()
data = load_data()

st.title("🏠 Real Estate Investment Advisor")
st.caption("Is this property a good investment, and what could it be worth in 5 years? "
           f"Trained on {len(data):,} Indian property listings.")

tab_adv, tab_explore, tab_insight, tab_model = st.tabs(
    ["🔮 Investment Advisor", "🔎 Property Explorer", "📊 Market Insights", "🧪 Model Performance"])

# ============================================================================ TAB 1
with tab_adv:
    st.subheader("Enter property details")
    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            city = st.selectbox("City", sorted(ref["city_state"]), index=sorted(ref["city_state"]).index("Pune"))
            st.caption(f"State: **{ref['city_state'][city]}**")
            ptype = st.selectbox("Property type", ["Apartment", "Independent House", "Villa"])
            bhk = st.slider("BHK", 1, 5, 3)
            size = st.number_input("Size (sq ft)", 500, 5000, 1800, step=50)
            price = st.number_input("Current price (INR lakhs)", 10.0, 500.0, 120.0, step=5.0)
            year_built = st.number_input("Year built", 1990, C.CURRENT_YEAR, 2012)
        with c2:
            furnished = st.selectbox("Furnished status", ["Unfurnished", "Semi-furnished", "Furnished"])
            total_floors = st.number_input("Total floors in building", 1, 30, 12)
            floor_no = st.number_input("Floor number", 0, 30, 5)
            facing = st.selectbox("Facing", ["North", "South", "East", "West"])
            owner = st.selectbox("Owner type", ["Owner", "Builder", "Broker"])
            avail = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])
        with c3:
            schools = st.slider("Nearby schools", 1, 10, 6)
            hospitals = st.slider("Nearby hospitals", 1, 10, 5)
            transport = st.select_slider("Public transport accessibility", ["Low", "Medium", "High"], value="Medium")
            parking = st.radio("Parking space", ["Yes", "No"], horizontal=True)
            security = st.radio("Security (gated / CCTV / guard)", ["Yes", "No"], horizontal=True)
            amen = st.multiselect("Amenities", C.AMENITIES, default=["Gym", "Garden"])
        go_btn = st.form_submit_button("Analyse this property", type="primary")

    if go_btn:
        user = dict(City=city, Property_Type=ptype, BHK=bhk, Size_in_SqFt=size, Price_in_Lakhs=price,
                    Year_Built=int(year_built), Furnished_Status=furnished, Floor_No=int(floor_no),
                    Total_Floors=int(total_floors), Nearby_Schools=schools, Nearby_Hospitals=hospitals,
                    Public_Transport_Accessibility=transport, Parking_Space=parking, Security=security,
                    Amenities=amen, Facing=facing, Owner_Type=owner, Availability_Status=avail)
        row = F.frame_from_user_input(user, ref)
        X = row[C.FEATURES]
        p_good = float(clf.predict_proba(X)[0, 1])
        is_good = p_good >= 0.5
        future = float(reg.predict(X)[0])
        gain = future - price
        cagr = ((future / price) ** (1 / C.FORECAST_YEARS) - 1) * 100
        ppsf = float(row.Price_per_SqFt_INR.iloc[0])
        city_med = ref["city_median_ppsf"][city]
        rmse = metrics["regression"]["results"][0]["rmse"]

        if floor_no > total_floors:
            st.info("Floor number is higher than total floors - it was capped to the top floor.")
        if floor_no <= total_floors and not (10 <= price <= 500 and 500 <= size <= 5000):
            st.warning("Price or size is outside the range the model was trained on (10-500 lakhs, 500-5,000 sq ft); treat the result with caution.")

        st.divider()
        a, b = st.columns([1, 1.2])
        with a:
            st.markdown("### Is this a Good Investment?")
            if is_good:
                st.success(f"✅ **GOOD INVESTMENT** - {p_good * 100:.0f}% model confidence")
            else:
                st.error(f"⚠️ **NOT RECOMMENDED** - {(1 - p_good) * 100:.0f}% model confidence")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=p_good * 100, number={"suffix": "%"},
                title={"text": "Probability of 'Good Investment'"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": BLUE if is_good else ORANGE},
                       "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
            fig.update_layout(height=260, margin=dict(t=60, b=10, l=30, r=30))
            st.plotly_chart(fig, width="stretch")
        with b:
            st.markdown("### Estimated price after 5 years")
            m1, m2, m3 = st.columns(3)
            m1.metric("Today", f"₹{price:,.1f} L")
            m2.metric(f"In {C.FORECAST_YEARS} yrs", f"₹{future:,.1f} L", f"{gain:+,.1f} L")
            m3.metric("Implied growth", f"{cagr:.1f}% / yr")
            yrs = list(range(0, C.FORECAST_YEARS + 1))
            vals = [price * (1 + cagr / 100) ** y for y in yrs]
            fig = go.Figure(go.Scatter(x=[C.CURRENT_YEAR + y for y in yrs], y=vals, mode="lines+markers",
                                       line=dict(color=BLUE, width=3), name="Projected price"))
            fig.update_layout(height=260, margin=dict(t=20, b=10), xaxis_title="Year", yaxis_title="Price (lakhs)",
                              template="plotly_white")
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Typical model error on unseen data: about ±{rmse:.1f} lakhs (RMSE).")

        st.markdown("### Why? Key facts about this property")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Price per sq ft", f"₹{ppsf:,.0f}", f"{(ppsf / city_med - 1) * 100:+.0f}% vs {city} median",
                  delta_color="inverse")
        k2.metric("Infrastructure score", f"{row.Infrastructure_Score.iloc[0]:.2f} / 1")
        k3.metric("Rule-based growth", f"{row.Annual_Growth_Rate_Pct.iloc[0]:.1f}% / yr",
                  help=f"Used to define 'Good Investment': needs >= {C.GOOD_GROWTH_THRESHOLD}% / yr and a price/sq ft "
                       f"no more than {round((C.MAX_PPSF_VS_CITY - 1) * 100)}% above the city median.")
        k4.metric("Amenities", f"{len(amen)} / 5")

        st.markdown("### What drives the model? (feature importance)")
        ic, ir = st.columns(2)
        for col, name, color, title in ((ic, "classifier", BLUE, "Classification (Good Investment)"),
                                        (ir, "regressor", ORANGE, "Regression (5-year price)")):
            imp = pd.read_csv(C.MODELS_DIR / f"feature_importance_{name}.csv").head(10).iloc[::-1]
            f = px.bar(imp, x="importance", y="feature", orientation="h", title=title, template="plotly_white")
            f.update_traces(marker_color=color)
            f.update_layout(height=330, margin=dict(t=40, b=10), yaxis_title="", xaxis_title="Relative importance")
            col.plotly_chart(f, width="stretch")
    else:
        st.info("Fill in the form and press **Analyse this property**.")

# ============================================================================ TAB 2
with tab_explore:
    st.subheader("Filter properties")
    f1, f2, f3, f4 = st.columns(4)
    states = f1.multiselect("State", sorted(data.State.unique()))
    cities_pool = sorted(data[data.State.isin(states)].City.unique()) if states else sorted(data.City.unique())
    cities = f2.multiselect("City", cities_pool)
    bhks = f3.multiselect("BHK", [1, 2, 3, 4, 5])
    types = f4.multiselect("Property type", sorted(data.Property_Type.unique()))
    g1, g2, g3 = st.columns(3)
    pr = g1.slider("Price (lakhs)", 10, 500, (10, 500))
    ar = g2.slider("Area (sq ft)", 500, 5000, (500, 5000), step=50)
    only_good = g3.checkbox("Only 'Good Investment' properties")
    avail_f = g3.multiselect("Availability", sorted(data.Availability_Status.unique()))

    m = data.Price_in_Lakhs.between(*pr) & data.Size_in_SqFt.between(*ar)
    if states: m &= data.State.isin(states)
    if cities: m &= data.City.isin(cities)
    if bhks: m &= data.BHK.isin(bhks)
    if types: m &= data.Property_Type.isin(types)
    if avail_f: m &= data.Availability_Status.isin(avail_f)
    if only_good: m &= data[C.CLF_TARGET] == 1
    res = data[m]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching properties", f"{len(res):,}")
    k2.metric("Avg price", f"₹{res.Price_in_Lakhs.mean():,.1f} L" if len(res) else "-")
    k3.metric("Avg price / sq ft", f"₹{res.Price_per_SqFt_INR.mean():,.0f}" if len(res) else "-")
    k4.metric("Good-investment share", f"{res[C.CLF_TARGET].mean() * 100:.0f}%" if len(res) else "-")
    show = res.sort_values("Annual_Growth_Rate_Pct", ascending=False).head(500).drop(columns=["Build_Period"])
    st.caption("Showing the top 500 matches, best expected growth first.")
    st.dataframe(show, width="stretch", hide_index=True)
    st.download_button("Download all matches (CSV)", res.drop(columns=["Build_Period"]).to_csv(index=False),
                       "filtered_properties.csv", "text/csv")

# ============================================================================ TAB 3
with tab_insight:
    st.subheader("Location-wise heatmap")
    h1, h2 = st.columns(2)
    metric = h1.selectbox("Metric", {"Avg price (lakhs)": "Price_in_Lakhs", "Avg price per sq ft (INR)": "Price_per_SqFt_INR",
                                     "Good-investment share (%)": C.CLF_TARGET, "Avg 5-yr price (lakhs)": C.REG_TARGET,
                                     "Avg modelled growth (% / yr)": "Annual_Growth_Rate_Pct"}.keys())
    mcol = {"Avg price (lakhs)": "Price_in_Lakhs", "Avg price per sq ft (INR)": "Price_per_SqFt_INR",
            "Good-investment share (%)": C.CLF_TARGET, "Avg 5-yr price (lakhs)": C.REG_TARGET,
            "Avg modelled growth (% / yr)": "Annual_Growth_Rate_Pct"}[metric]
    colvar = h2.selectbox("Compare across", ["BHK", "Property_Type", "Public_Transport_Accessibility",
                                             "Furnished_Status", "Availability_Status"])
    st.caption("Note: the colour scale is stretched between each metric's min and max, so tiny differences can look large - "
               "read the numbers in the cells. Price metrics in this dataset are nearly flat across states (see the EDA report).")
    hm = data.pivot_table(index="State", columns=colvar, values=mcol, aggfunc="mean")
    if mcol == C.CLF_TARGET:
        hm = hm * 100
    fig = px.imshow(hm, aspect="auto", color_continuous_scale="Blues", text_auto=".1f", template="plotly_white")
    fig.update_layout(height=560, coloraxis_colorbar_title=metric)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Trend charts")
    t1, t2 = st.columns(2)
    pick = t1.multiselect("Cities to compare", sorted(data.City.unique()), default=["Mumbai", "Pune", "Bangalore"])
    if pick:
        tr = (data[data.City.isin(pick)].groupby(["Build_Period", "City"], observed=True).Price_in_Lakhs.mean().reset_index())
        f = px.line(tr, x="Build_Period", y="Price_in_Lakhs", color="City", markers=True, template="plotly_white",
                    title="Avg price by construction period")
        f.update_layout(height=380, xaxis_title="Construction period (no listing dates in the data)", yaxis_title="Avg price (lakhs)")
        t1.plotly_chart(f, width="stretch")
    top = (data.groupby("City")[C.CLF_TARGET].mean().mul(100).sort_values(ascending=False).head(12).iloc[::-1].reset_index())
    f = px.bar(top, x=C.CLF_TARGET, y="City", orientation="h", template="plotly_white",
               title="Cities with the highest share of Good Investments")
    f.update_traces(marker_color=BLUE)
    f.update_layout(height=380, xaxis_title="% of listings", yaxis_title="")
    t2.plotly_chart(f, width="stretch")

# ============================================================================ TAB 4
with tab_model:
    st.subheader("Model comparison (held-out test set, 50,000 properties)")
    cm_, rm_ = metrics["classification"], metrics["regression"]
    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"**Classification** - best: `{cm_['best_model']}`")
        st.dataframe(pd.DataFrame(cm_["results"])[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]],
                     hide_index=True, width="stretch")
        st.caption(f"Majority-class baseline accuracy: {cm_['baseline_majority_accuracy']:.3f}")
    with cb:
        st.markdown(f"**Regression** - best: `{rm_['best_model']}`")
        st.dataframe(pd.DataFrame(rm_["results"])[["model", "rmse", "mae", "r2", "mape_pct"]], hide_index=True, width="stretch")
        st.caption(f"Naive baseline (price in 5 yrs = price today) RMSE: {rm_['baseline_naive_rmse_lakhs']:.1f} lakhs")
    i1, i2, i3 = st.columns(3)
    i1.image(str(C.FIG_DIR / "model_confusion_matrix.png"))
    i2.image(str(C.FIG_DIR / "model_actual_vs_predicted.png"))
    i3.image(str(C.FIG_DIR / "model_feature_importance_classifier.png"))

    with st.expander("⚠️ How to interpret these results (please read)", expanded=True):
        st.markdown(f"""
- The dataset has **no future price and no 'good investment' label**. Both were created from the transparent domain
  rule in `src/config.py` (expected annual growth = base {C.GROWTH['base']}% + transport, infrastructure, new-build,
  value, security, parking and amenity bonuses - age penalty; *Good* = growth >= {C.GOOD_GROWTH_THRESHOLD}% and price/sq ft
  not more than {round((C.MAX_PPSF_VS_CITY - 1) * 100)}% above the city median).
- Because the targets come from a rule over the same features, the high scores show that the models **learned the rule**,
  not that they can predict the real property market.
- The original columns in the dataset are statistically independent of price, so no model could genuinely forecast
  real prices from them.
- *Locality* is not a model input (500 anonymous labels per city, no signal); *City* is.
- Use this as a decision-support demo, not financial advice.
""")
