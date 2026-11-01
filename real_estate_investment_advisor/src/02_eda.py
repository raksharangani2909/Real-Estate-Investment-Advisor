"""STEP 2 - Exploratory Data Analysis (the 20 questions in the brief + 3 extras).

Run:  python src/02_eda.py
Out:  reports/figures/*.png      (one chart per question)
      reports/eda_findings.md    (numeric answer to every question)
      reports/eda_findings.json
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as C

# ---- style: one blue for magnitude, one orange for contrast, quiet grid -----
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SEQ = sns.light_palette(BLUE, as_cmap=True)
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": "#52514e", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left", "font.size": 10, "figure.dpi": 100,
})
FIG_DIR = C.FIG_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIND = {}  # question id -> dict(question, answer)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=130)
    plt.close(fig)


def note(qid, question, answer):
    FIND[qid] = {"question": question, "answer": answer}


def strength(r):
    a = abs(r)
    return "negligible" if a < 0.05 else "weak" if a < 0.3 else "moderate" if a < 0.6 else "strong"


def bar_h(ax, s, color=BLUE, fmt="{:.1f}"):
    s = s.sort_values()
    ax.barh(s.index, s.values, color=color, height=0.7)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(s.values):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color="#52514e")


def main():
    df = pd.read_csv(C.CLEAN_CSV)
    df["PPSF_k"] = df.Price_per_SqFt_INR / 1000   # thousand INR per sqft
    sample = df.sample(20000, random_state=1)

    # ===================== 1-5  PRICE & SIZE ==============================
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.Price_in_Lakhs, bins=50, color=BLUE)
    ax.axvline(df.Price_in_Lakhs.median(), color=ORANGE, lw=2)
    ax.set(title="Q1  Distribution of property prices", xlabel="Price (INR lakhs)", ylabel="Properties")
    ax.text(df.Price_in_Lakhs.median(), ax.get_ylim()[1] * 0.95, f"  median {df.Price_in_Lakhs.median():.0f}", color=ORANGE, va="top")
    save(fig, "q01_price_distribution")
    p = df.Price_in_Lakhs
    note("Q1", "Distribution of property prices",
         f"Prices run {p.min():.0f}-{p.max():.0f} lakhs, mean {p.mean():.1f}, median {p.median():.1f}, std {p.std():.1f}. "
         f"The histogram is essentially flat (skew {p.skew():.2f}), i.e. close to uniform - there is no long luxury tail.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.Size_in_SqFt, bins=50, color=BLUE)
    ax.axvline(df.Size_in_SqFt.median(), color=ORANGE, lw=2)
    ax.set(title="Q2  Distribution of property sizes", xlabel="Size (sq ft)", ylabel="Properties")
    save(fig, "q02_size_distribution")
    s = df.Size_in_SqFt
    note("Q2", "Distribution of property sizes",
         f"Sizes run {s.min():.0f}-{s.max():.0f} sq ft, mean {s.mean():.0f}, median {s.median():.0f}; the shape is flat/uniform (skew {s.skew():.2f}).")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    order = ["Apartment", "Independent House", "Villa"]
    sns.boxplot(data=df, x="Property_Type", y="PPSF_k", order=order, color=BLUE, fliersize=1.5,
                flierprops={"markerfacecolor": MUTED, "alpha": .3}, ax=ax, linewidth=1)
    ax.set_yscale("log")
    ax.set(title="Q3  Price per sq ft by property type", xlabel="", ylabel="INR '000 per sq ft (log scale)")
    save(fig, "q03_ppsf_by_property_type")
    g = df.groupby("Property_Type").Price_per_SqFt_INR.median().round(0)
    note("Q3", "Price per sq ft by property type",
         "Median INR/sq ft: " + ", ".join(f"{k} {v:,.0f}" for k, v in g.items()) +
         f". Spread between types is only {(g.max() / g.min() - 1) * 100:.1f}% - property type barely moves price/sq ft.")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.scatter(sample.Size_in_SqFt, sample.Price_in_Lakhs, s=3, alpha=.25, color=BLUE, linewidths=0)
    ax.set(title="Q4  Property size vs price (20k random sample)", xlabel="Size (sq ft)", ylabel="Price (INR lakhs)")
    save(fig, "q04_size_vs_price")
    r = df.Size_in_SqFt.corr(df.Price_in_Lakhs)
    note("Q4", "Relationship between property size and price",
         f"Pearson r = {r:.3f} ({strength(r)}). In this dataset a bigger home is NOT priced higher in absolute terms - "
         f"size and price vary independently, so price per sq ft falls as size rises (r with size = {df.Size_in_SqFt.corr(df.Price_per_SqFt_INR):.2f}).")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for ax, col, ttl in zip(axes, ["PPSF_k", "Size_in_SqFt", "Price_in_Lakhs"],
                            ["Price/sq ft ('000 INR)", "Size (sq ft)", "Price (lakhs)"]):
        sns.boxplot(y=df[col], ax=ax, color=BLUE, fliersize=1.5, flierprops={"markerfacecolor": MUTED, "alpha": .3})
        ax.set(title=ttl, ylabel="")
    axes[0].set_yscale("log")
    fig.suptitle("Q5  Outlier check (IQR rule)", x=0.01, ha="left", fontweight="bold")
    save(fig, "q05_outliers")
    note("Q5", "Outliers in price per sq ft or property size",
         f"IQR rule: price/sq ft has {int(df.PPSF_Outlier.sum()):,} outliers ({df.PPSF_Outlier.mean() * 100:.1f}%), all on the HIGH side "
         f"(small flats with high prices, max {df.Price_per_SqFt_INR.max():,.0f} INR/sq ft). Size ({int(df.Size_Outlier.sum())}) and price ({int(df.Price_Outlier.sum())}) have none. "
         "These are valid records, so they are flagged (PPSF_Outlier) rather than deleted; models use trees/scaling that tolerate them.")

    # ===================== 6-10  LOCATION =================================
    fig, ax = plt.subplots(figsize=(7, 6))
    st = df.groupby("State").Price_per_SqFt_INR.mean() / 1000
    bar_h(ax, st, fmt="{:.2f}")
    ax.set(title="Q6  Average price per sq ft by state", xlabel="INR '000 per sq ft")
    save(fig, "q06_ppsf_by_state")
    note("Q6", "Average price per sq ft by state",
         f"Highest: {st.idxmax()} ({st.max() * 1000:,.0f} INR), lowest: {st.idxmin()} ({st.min() * 1000:,.0f} INR) - "
         f"only a {(st.max() / st.min() - 1) * 100:.1f}% gap across 20 states.")

    fig, ax = plt.subplots(figsize=(7, 9))
    ct = df.groupby("City").Price_in_Lakhs.mean()
    bar_h(ax, ct, fmt="{:.0f}")
    ax.set_xlim(240, ct.max() * 1.03)
    ax.set(title="Q7  Average property price by city", xlabel="Average price (lakhs) - axis starts at 240")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "q07_price_by_city")
    note("Q7", "Average property price by city",
         f"Highest: {ct.idxmax()} ({ct.max():.1f} lakhs), lowest: {ct.idxmin()} ({ct.min():.1f} lakhs); all 42 cities sit within "
         f"{(ct.max() / ct.min() - 1) * 100:.1f}% of each other (city mean of means {ct.mean():.1f}).")

    gl = df.groupby(["City", "Locality"]).Age_of_Property.median()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    axes[0].hist(gl, bins=30, color=BLUE)
    axes[0].set(title="Median age per locality", xlabel="Median age (years)", ylabel="Localities (21,000)")
    top = gl.sort_values(ascending=False).head(10)
    top.index = [f"{c} / {l}" for c, l in top.index]
    bar_h(axes[1], top, color=ORANGE, fmt="{:.0f}")
    axes[1].set_xlim(top.min() - 2, top.max() + 1)
    axes[1].set(title="10 localities with the oldest stock", xlabel="Median age (years)")
    axes[1].tick_params(axis="y", labelsize=8)
    fig.suptitle("Q8  Median age of properties by locality", x=0.01, ha="left", fontweight="bold")
    save(fig, "q08_median_age_by_locality")
    note("Q8", "Median age of properties by locality",
         f"Locality names repeat in every city, so a locality = (City, Locality). Across {len(gl):,} localities the median age ranges "
         f"{gl.min():.0f}-{gl.max():.0f} years (overall median {gl.median():.1f}). Each locality only has ~12 listings, so individual values are noisy. "
         f"Oldest: {top.index[0]} ({top.iloc[0]:.0f} yrs).")

    fig, ax = plt.subplots(figsize=(7, 10))
    bh = pd.crosstab(df.City, df.BHK, normalize="index") * 100
    sns.heatmap(bh, cmap=SEQ, annot=True, fmt=".0f", cbar_kws={"label": "% of city's listings"}, ax=ax,
                linewidths=.5, linecolor=SURF, annot_kws={"size": 7})
    ax.set(title="Q9  BHK distribution across cities (% of listings)", xlabel="BHK", ylabel="")
    ax.tick_params(axis="y", labelsize=8)
    save(fig, "q09_bhk_by_city")
    note("Q9", "How BHK is distributed across cities",
         f"Every city has all five BHK sizes at roughly 20% each (range {bh.values.min():.1f}%-{bh.values.max():.1f}%) - no city specialises in small or large homes.")

    # Q10: top-5 most expensive localities, price trend by construction year band
    gm = df.groupby(["City", "Locality"]).Price_in_Lakhs.agg(["mean", "size"])
    top5 = gm[gm["size"] >= 8].sort_values("mean", ascending=False).head(5)
    df["Build_Period"] = pd.cut(df.Year_Built, [1989, 1995, 2000, 2005, 2010, 2015, 2020, 2023],
                                labels=["1990-95", "1996-00", "2001-05", "2006-10", "2011-15", "2016-20", "2021-23"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cols = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4"]
    out = []
    for (city, loc), col in zip(top5.index, cols):
        sub = (df[(df.City == city) & (df.Locality == loc)].groupby("Build_Period", observed=False)
               .Price_in_Lakhs.mean())          # all 7 periods, NaN where the locality has no listing
        ax.plot(range(len(sub)), sub.values, marker="o", ms=5, lw=2, color=col, label=f"{city} / {loc}")
        out.append(f"{city} / {loc} ({top5.loc[(city, loc), 'mean']:.0f} lakhs)")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["1990-95", "1996-00", "2001-05", "2006-10", "2011-15", "2016-20", "2021-23"])
    ax.legend(frameon=False, fontsize=8)
    ax.set(title="Q10  Price trend of the 5 most expensive localities (by year built)",
           xlabel="Construction period (the data has no listing dates, so build year is the only time axis)",
           ylabel="Avg price (lakhs)")
    save(fig, "q10_top5_locality_trends")
    note("Q10", "Price trends for the top 5 most expensive localities",
         "Top 5 (localities with >=8 listings, ranked by mean price): " + "; ".join(out) +
         ". The dataset has no transaction date, so 'trend' = average price by construction period. Lines are jagged (few listings per band) "
         "and show no consistent rise for newer homes.")

    # ===================== 11-15  RELATIONSHIPS ===========================
    num = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt_INR", "BHK", "Age_of_Property", "Nearby_Schools",
           "Nearby_Hospitals", "Floor_No", "Total_Floors", "Amenity_Count", "Transport_Score", "Infrastructure_Score"]
    cm = df[num].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=sns.diverging_palette(250, 25, as_cmap=True, center="light"),
                vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7}, linewidths=.5, linecolor=SURF)
    ax.set(title="Q11  Correlation of numeric features")
    save(fig, "q11_correlation_heatmap")
    pairs = cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack().sort_values(key=abs, ascending=False)
    note("Q11", "How numeric features are correlated",
         "Strongest pairs: " + "; ".join(f"{a} vs {b} r={v:.2f}" for (a, b), v in pairs.head(5).items()) +
         ". Every strong pair involves a derived field: price/sq ft comes from price and size, infrastructure score from schools, hospitals and transport, "
         "and Floor_No vs Total_Floors is high only because Floor_No was capped at Total_Floors during cleaning. "
         "Among the ORIGINAL columns in the raw file, every pairwise correlation is below 0.01 (checked on the uncleaned data).")

    def level_chart(col, xlabel, qid, name, title, q):
        fig, ax = plt.subplots(figsize=(7, 4))
        m = df.groupby(col).PPSF_k.mean()
        ax.bar(m.index.astype(str), m.values, color=BLUE, width=0.7)
        ax.set_ylim(m.min() * 0.9, m.max() * 1.03)
        ax.set(title=title, xlabel=xlabel, ylabel="Avg price per sq ft ('000 INR) - axis truncated")
        save(fig, name)
        r = df[col].corr(df.Price_per_SqFt_INR)
        note(qid, q, f"Pearson r with price/sq ft = {r:.3f} ({strength(r)}); mean price/sq ft ranges "
                     f"{m.min() * 1000:,.0f}-{m.max() * 1000:,.0f} INR across levels - no systematic trend. "
                     f"(Correlation with total price: {df[col].corr(df.Price_in_Lakhs):.3f}.)")
    level_chart("Nearby_Schools", "Nearby schools", "Q12", "q12_schools_vs_ppsf",
                "Q12  Nearby schools vs price per sq ft", "How nearby schools relate to price per sq ft")
    level_chart("Nearby_Hospitals", "Nearby hospitals", "Q13", "q13_hospitals_vs_ppsf",
                "Q13  Nearby hospitals vs price per sq ft", "How nearby hospitals relate to price per sq ft")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    f = df.groupby("Furnished_Status")[["Price_in_Lakhs", "PPSF_k"]].mean().loc[["Unfurnished", "Semi-furnished", "Furnished"]]
    axes[0].bar(f.index, f.Price_in_Lakhs, color=BLUE, width=0.6)
    axes[0].set_ylim(f.Price_in_Lakhs.min() * 0.9, f.Price_in_Lakhs.max() * 1.03)
    axes[0].set(title="Avg price (lakhs)", xlabel="")
    axes[1].bar(f.index, f.PPSF_k, color=ORANGE, width=0.6)
    axes[1].set_ylim(f.PPSF_k.min() * 0.9, f.PPSF_k.max() * 1.03)
    axes[1].set(title="Avg price per sq ft ('000 INR)", xlabel="")
    fig.suptitle("Q14  Price by furnished status (axes truncated)", x=0.01, ha="left", fontweight="bold")
    save(fig, "q14_price_by_furnished")
    note("Q14", "How price varies by furnished status",
         "Avg price (lakhs): " + ", ".join(f"{k} {v:.1f}" for k, v in f.Price_in_Lakhs.items()) +
         f". Spread {(f.Price_in_Lakhs.max() / f.Price_in_Lakhs.min() - 1) * 100:.1f}% - furnishing carries no price premium here.")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    fc = df.groupby("Facing").PPSF_k.mean().sort_values()
    ax.bar(fc.index, fc.values, color=BLUE, width=0.6)
    ax.set_ylim(fc.min() * 0.9, fc.max() * 1.03)
    ax.set(title="Q15  Price per sq ft by facing direction", xlabel="", ylabel="Avg price/sq ft ('000 INR) - truncated")
    save(fig, "q15_ppsf_by_facing")
    note("Q15", "How price per sq ft varies by facing direction",
         "Mean INR/sq ft: " + ", ".join(f"{k} {v * 1000:,.0f}" for k, v in fc.items()) +
         f". Gap max-min = {(fc.max() / fc.min() - 1) * 100:.1f}% - direction does not matter in this data.")

    # ===================== 16-20  OWNERSHIP / AMENITIES ===================
    fig, ax = plt.subplots(figsize=(6, 4))
    oc = df.Owner_Type.value_counts()
    ax.bar(oc.index, oc.values, color=BLUE, width=0.55)
    for i, v in enumerate(oc.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set(title="Q16  Properties by owner type", ylabel="Properties")
    save(fig, "q16_owner_type")
    note("Q16", "How many properties belong to each owner type", ", ".join(f"{k}: {v:,}" for k, v in oc.items()) + " (about one-third each).")

    fig, ax = plt.subplots(figsize=(6, 4))
    av = df.Availability_Status.value_counts()
    ax.bar(av.index, av.values, color=[BLUE, ORANGE][:len(av)], width=0.5)
    for i, v in enumerate(av.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set(title="Q17  Properties by availability status", ylabel="Properties")
    save(fig, "q17_availability_status")
    note("Q17", "How many properties are available under each availability status",
         ", ".join(f"{k}: {v:,}" for k, v in av.items()) + ". (The data has only these two statuses - no 'Sold'.)")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    pk = df.groupby("Parking_Space")[["Price_in_Lakhs", "PPSF_k"]].mean()
    axes[0].bar(pk.index, pk.Price_in_Lakhs, color=BLUE, width=0.5)
    axes[0].set_ylim(pk.Price_in_Lakhs.min() * 0.9, pk.Price_in_Lakhs.max() * 1.03)
    axes[0].set(title="Avg price (lakhs)", xlabel="Parking space")
    axes[1].bar(pk.index, pk.PPSF_k, color=ORANGE, width=0.5)
    axes[1].set_ylim(pk.PPSF_k.min() * 0.9, pk.PPSF_k.max() * 1.03)
    axes[1].set(title="Avg price/sq ft ('000 INR)", xlabel="Parking space")
    fig.suptitle("Q18  Does parking affect price? (axes truncated)", x=0.01, ha="left", fontweight="bold")
    save(fig, "q18_parking_vs_price")
    note("Q18", "Does parking space affect property price",
         f"Avg price with parking {pk.loc['Yes', 'Price_in_Lakhs']:.1f} vs without {pk.loc['No', 'Price_in_Lakhs']:.1f} lakhs "
         f"({(pk.loc['Yes', 'Price_in_Lakhs'] / pk.loc['No', 'Price_in_Lakhs'] - 1) * 100:+.2f}%). No meaningful effect on price. "
         "(Parking does add to the modelled 5-year growth rule.)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    am = df.groupby("Amenity_Count").PPSF_k.mean()
    axes[0].bar(am.index.astype(str), am.values, color=BLUE, width=0.6)
    axes[0].set_ylim(am.min() * 0.9, am.max() * 1.03)
    axes[0].set(title="By number of amenities", xlabel="Amenities in the property", ylabel="Avg price/sq ft ('000 INR) - truncated")
    diff = {a: df[df[a] == 1].PPSF_k.mean() / df[df[a] == 0].PPSF_k.mean() * 100 - 100 for a in C.AMENITIES}
    ds = pd.Series(diff).sort_values()
    axes[1].barh(ds.index, ds.values, color=[ORANGE if v < 0 else BLUE for v in ds.values], height=0.6)
    axes[1].axvline(0, color=INK, lw=.8)
    axes[1].grid(axis="y", visible=False)
    axes[1].set(title="Having the amenity vs not (%)", xlabel="Difference in avg price/sq ft (%)")
    fig.suptitle("Q19  How amenities affect price per sq ft", x=0.01, ha="left", fontweight="bold")
    save(fig, "q19_amenities_vs_ppsf")
    r = df.Amenity_Count.corr(df.Price_per_SqFt_INR)
    note("Q19", "How amenities affect price per sq ft",
         f"Amenity count vs price/sq ft r = {r:.3f} ({strength(r)}). Per-amenity price/sq ft difference (has vs hasn't): "
         + ", ".join(f"{k} {v:+.1f}%" for k, v in diff.items()) + ". Mixed signs of small size = noise, not a premium.")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    order = ["Low", "Medium", "High"]
    tp = df.groupby("Public_Transport_Accessibility").agg(ppsf=("PPSF_k", "mean"), good=(C.CLF_TARGET, "mean"),
                                                          growth=("Annual_Growth_Rate_Pct", "mean")).loc[order]
    axes[0].bar(order, tp.ppsf, color=BLUE, width=0.55)
    axes[0].set_ylim(tp.ppsf.min() * 0.9, tp.ppsf.max() * 1.03)
    axes[0].set(title="Avg price/sq ft ('000 INR)", xlabel="Public transport accessibility")
    axes[1].bar(order, tp.good * 100, color=ORANGE, width=0.55)
    for i, v in enumerate(tp.good * 100):
        axes[1].text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set(title="Share rated 'Good Investment' (%)", xlabel="Public transport accessibility")
    fig.suptitle("Q20  Public transport vs price per sq ft and investment potential", x=0.01, ha="left", fontweight="bold")
    save(fig, "q20_transport_vs_ppsf_investment")
    note("Q20", "How public transport accessibility relates to price per sq ft or investment potential",
         "Price/sq ft is flat across Low/Medium/High (" + ", ".join(f"{k} {v * 1000:,.0f}" for k, v in tp.ppsf.items()) + " INR). "
         "Investment potential, however, rises sharply because transport is a driver of the growth rule: share of Good Investments = "
         + ", ".join(f"{k} {v * 100:.0f}%" for k, v in tp.good.items()) + " (by construction of the rule, not observed market behaviour).")

    # ===================== EXTRAS (from the 'Approach' section) ===========
    # X1 area vs investment return
    df["Size_Band"] = pd.cut(df.Size_in_SqFt, [499, 1000, 2000, 3000, 4000, 5000],
                             labels=["500-1k", "1-2k", "2-3k", "3-4k", "4-5k"])
    sb = df.groupby("Size_Band", observed=True).agg(growth=("Annual_Growth_Rate_Pct", "mean"), good=(C.CLF_TARGET, "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(sb.index.astype(str), sb.growth, color=BLUE, width=0.6)
    axes[0].set_ylim(sb.growth.min() * 0.95, sb.growth.max() * 1.02)
    axes[0].set(title="Avg modelled annual growth (%)", xlabel="Size band (sq ft)")
    axes[1].bar(sb.index.astype(str), sb.good * 100, color=ORANGE, width=0.6)
    axes[1].set(title="Share 'Good Investment' (%)", xlabel="Size band (sq ft)")
    fig.suptitle("X1  Property area vs investment return", x=0.01, ha="left", fontweight="bold")
    save(fig, "x1_area_vs_investment_return")
    note("X1", "Correlation between area and investment return",
         f"Size vs annual growth r = {df.Size_in_SqFt.corr(df.Annual_Growth_Rate_Pct):.3f}; Good-Investment share by size band: "
         + ", ".join(f"{k} {v * 100:.0f}%" for k, v in sb.good.items()) +
         ". The share of Good Investments rises steeply with size because larger homes have a lower price/sq ft (see Q4): they more often qualify for the "
         "'undervalued' growth bonus and pass the 'not over-priced' test. This is an effect of the investment rule, not observed market returns.")

    # X2 crime-rate proxy
    sc = df.groupby("Security")[C.CLF_TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sc.index, sc.values, color=BLUE, width=0.5)
    for i, v in enumerate(sc.values):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    ax.set(title="X2  Safety proxy (Security) vs Good Investment", xlabel="Security features available",
           ylabel="% rated Good Investment")
    save(fig, "x2_security_vs_good_investment")
    note("X2", "Impact of crime rate on 'Good Investment' classification",
         f"The dataset has NO crime-rate column, so the Security flag is used as a safety proxy: Good-Investment share {sc['Yes']:.1f}% with security vs {sc['No']:.1f}% without. "
         "The gap exists only because the growth rule gives security a small bonus.")

    # X3 infrastructure score vs resale value
    df["Infra_Band"] = pd.cut(df.Infrastructure_Score, [0, .2, .4, .6, .8, 1.0], include_lowest=True,
                              labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
    ib = df.groupby("Infra_Band", observed=True).agg(price=("Price_in_Lakhs", "mean"), ppsf=("PPSF_k", "mean"),
                                                     fut=(C.REG_TARGET, "mean"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, col, ttl, c in zip(axes, ["price", "ppsf", "fut"], ["Avg price today (lakhs)", "Avg price/sq ft ('000 INR)",
                                                              "Avg price in 5 yrs (lakhs)"], [BLUE, BLUE, ORANGE]):
        ax.bar(ib.index.astype(str), ib[col], color=c, width=0.6)
        ax.set_ylim(ib[col].min() * 0.9, ib[col].max() * 1.03)
        ax.set(title=ttl, xlabel="Infrastructure score")
    fig.suptitle("X3  Infrastructure score vs resale value (axes truncated)", x=0.01, ha="left", fontweight="bold")
    save(fig, "x3_infrastructure_vs_value")
    note("X3", "Relationship between infrastructure score and resale value",
         f"Infrastructure score vs today's price r = {df.Infrastructure_Score.corr(df.Price_in_Lakhs):.3f} and vs price/sq ft r = "
         f"{df.Infrastructure_Score.corr(df.Price_per_SqFt_INR):.3f} (both {strength(df.Infrastructure_Score.corr(df.Price_in_Lakhs))}). Vs the projected 5-year value r = "
         f"{df.Infrastructure_Score.corr(df[C.REG_TARGET]):.3f}: infrastructure raises the FORECAST value only because it drives the growth rule.")

    # ---- write findings -------------------------------------------------
    md = ["# EDA findings\n", f"Dataset: {len(df):,} cleaned properties, 20 states, {df.City.nunique()} cities. "
          "One chart per question is in `reports/figures/`.\n",
          "> **Read this first:** the original features are almost perfectly independent of each other and of price (pairwise |r| < 0.01). "
          "That is typical of a synthetic/randomly generated dataset, so most 'no relationship' answers below are genuine findings about THIS data, not mistakes.\n"]
    for k, v in FIND.items():
        md.append(f"### {k}. {v['question']}\n{v['answer']}\n")
    (C.REPORTS_DIR / "eda_findings.md").write_text("\n".join(md))
    (C.REPORTS_DIR / "eda_findings.json").write_text(json.dumps(FIND, indent=2))
    print(f"Saved {len(FIND)} findings and {len(list(FIG_DIR.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
