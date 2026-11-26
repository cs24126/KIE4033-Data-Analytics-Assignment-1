"""
02_exploratory_data_analysis.py
-------------------------------
Exploratory Data Analysis (EDA) for KIE4033 Assignment 1:
Global Health Financing and Workforce Capacity (UN SDG 3.8 & 3.c).

Inputs:
    data/processed/cleaned_sdg3_panel_2000_2022.csv
    data/processed/cleaned_sdg3_cross_section_2018_2022.csv
    data/processed/cleaned_sdg3_covid_paired.csv

Outputs:
    figures/fig01_spearman_correlation_heatmap.png
    figures/fig02_income_tier_distributions.png
    figures/fig03_preston_diminishing_returns.png
    figures/fig04_longitudinal_trends_2000_2022.png
    figures/fig05_covid_shift_overview.png
    results/eda/*.csv & results/eda/eda_summary_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration & Styling
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results" / "eda"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INCOME_ORDER = [
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "High income",
]
INCOME_SHORT = {
    "Low income": "Low",
    "Lower middle income": "Lower\nMiddle",
    "Upper middle income": "Upper\nMiddle",
    "High income": "High",
}
# Colorblind-safe palette (Okabe-Ito inspired)
INCOME_PALETTE = {
    "Low income": "#D55E00",
    "Lower middle income": "#E69F00",
    "Upper middle income": "#009E73",
    "High income": "#0072B2",
}

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titleweight": "bold",
    "axes.labelweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

KEY_VARS = [
    "health_exp_per_capita_usd",
    "health_exp_pct_gdp",
    "gdp_per_capita_usd",
    "physicians_per_1000",
    "hospital_beds_per_1000",
    "life_expectancy",
    "infant_mortality_per_1000",
    "under5_mortality_per_1000",
]

VAR_LABELS = {
    "health_exp_per_capita_usd": "Health Exp. / Capita (US$)",
    "health_exp_pct_gdp": "Health Exp. (% GDP)",
    "gdp_per_capita_usd": "GDP / Capita (US$)",
    "physicians_per_1000": "Physicians / 1,000",
    "hospital_beds_per_1000": "Hospital Beds / 1,000",
    "life_expectancy": "Life Expectancy (yrs)",
    "infant_mortality_per_1000": "Infant Mortality / 1,000",
    "under5_mortality_per_1000": "Under-5 Mortality / 1,000",
}


def load_datasets():
    panel = pd.read_csv(PROCESSED_DIR / "cleaned_sdg3_panel_2000_2022.csv")
    cs = pd.read_csv(PROCESSED_DIR / "cleaned_sdg3_cross_section_2018_2022.csv")
    covid = pd.read_csv(PROCESSED_DIR / "cleaned_sdg3_covid_paired.csv")

    for df in (panel, cs, covid):
        df["income_group"] = pd.Categorical(
            df["income_group"], categories=INCOME_ORDER, ordered=True
        )
    return panel, cs, covid


# ---------------------------------------------------------------------------
# Summary Statistics Tables
# ---------------------------------------------------------------------------
def compute_summary_tables(cs: pd.DataFrame, panel: pd.DataFrame, covid: pd.DataFrame):
    # 1. Overall cross-sectional summary (2018-2022)
    rows = []
    for col in KEY_VARS:
        s = cs[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append({
            "variable": col,
            "n_valid": int(s.count()),
            "n_missing": int(cs[col].isna().sum()),
            "mean": round(s.mean(), 3),
            "std": round(s.std(), 3),
            "median": round(s.median(), 3),
            "q1": round(q1, 3),
            "q3": round(q3, 3),
            "iqr": round(q3 - q1, 3),
            "min": round(s.min(), 3),
            "max": round(s.max(), 3),
            "skewness": round(s.skew(), 3),
        })
    overall_df = pd.DataFrame(rows)
    overall_df.to_csv(RESULTS_DIR / "summary_stats_overall_2018_2022.csv", index=False)

    # 2. Grouped summary by World Bank income tier
    grp_rows = []
    for col in KEY_VARS:
        for ig in INCOME_ORDER:
            s = cs.loc[cs["income_group"] == ig, col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            grp_rows.append({
                "variable": col,
                "income_group": ig,
                "n_valid": int(s.count()),
                "median": round(s.median(), 3),
                "iqr": round(q3 - q1, 3),
                "mean": round(s.mean(), 3),
                "std": round(s.std(), 3),
                "min": round(s.min(), 3),
                "max": round(s.max(), 3),
            })
    by_income_df = pd.DataFrame(grp_rows)
    by_income_df.to_csv(RESULTS_DIR / "summary_stats_by_income_2018_2022.csv", index=False)

    # 3. Pairwise Spearman correlation & p-value matrices
    rho_mat = pd.DataFrame(index=KEY_VARS, columns=KEY_VARS, dtype=float)
    pval_mat = pd.DataFrame(index=KEY_VARS, columns=KEY_VARS, dtype=float)
    n_mat = pd.DataFrame(index=KEY_VARS, columns=KEY_VARS, dtype=int)
    for c1 in KEY_VARS:
        for c2 in KEY_VARS:
            sub = cs[[c1, c2]].dropna()
            n_mat.loc[c1, c2] = len(sub)
            if c1 == c2:
                rho_mat.loc[c1, c2] = 1.0
                pval_mat.loc[c1, c2] = 0.0
            else:
                r, p = stats.spearmanr(sub[c1], sub[c2])
                rho_mat.loc[c1, c2] = r
                pval_mat.loc[c1, c2] = p
    rho_mat.round(4).to_csv(RESULTS_DIR / "spearman_correlation_matrix.csv")
    pval_mat.to_csv(RESULTS_DIR / "spearman_pvalues_matrix.csv")

    # 4. Year-by-year medians across 2017-2022 for complete COVID financing countries (n=191)
    valid_isos = set(covid.loc[covid["health_exp_pct_gdp_diff"].notna(), "iso3"])
    sub_pan = panel[(panel["year"].between(2017, 2022)) & (panel["iso3"].isin(valid_isos))]
    yearly_rows = []
    for yr, g in sub_pan.groupby("year"):
        yearly_rows.append({
            "year": int(yr),
            "n_health_exp": int(g["health_exp_pct_gdp"].notna().sum()),
            "median_health_exp_pct_gdp": round(g["health_exp_pct_gdp"].median(), 3),
            "median_health_exp_per_capita_usd": round(g["health_exp_per_capita_usd"].median(), 2),
            "median_gdp_per_capita_usd": round(g["gdp_per_capita_usd"].median(), 2),
            "median_life_expectancy": round(g["life_expectancy"].median(), 3),
        })
    covid_yearly_df = pd.DataFrame(yearly_rows)
    covid_yearly_df.to_csv(RESULTS_DIR / "covid_yearly_medians_2017_2022.csv", index=False)

    return overall_df, by_income_df, rho_mat, pval_mat, n_mat, covid_yearly_df


# ---------------------------------------------------------------------------
# Figure 1: Spearman Correlation Matrix Heatmap
# ---------------------------------------------------------------------------
def plot_spearman_heatmap(rho_mat: pd.DataFrame, pval_mat: pd.DataFrame, n_mat: pd.DataFrame):
    labels = [VAR_LABELS[c] for c in KEY_VARS]
    mask = np.triu(np.ones_like(rho_mat, dtype=bool), k=1)
    off_diag_n = n_mat.values[np.tril_indices(len(KEY_VARS), k=-1)]
    n_min, n_max = int(off_diag_n.min()), int(off_diag_n.max())

    annot = pd.DataFrame("", index=KEY_VARS, columns=KEY_VARS)
    for i, c1 in enumerate(KEY_VARS):
        for j, c2 in enumerate(KEY_VARS):
            if i == j:
                annot.loc[c1, c2] = "1.00"
            elif i > j:
                r = rho_mat.loc[c1, c2]
                p = pval_mat.loc[c1, c2]
                stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                annot.loc[c1, c2] = f"{r:.2f}{stars}"

    fig, ax = plt.subplots(figsize=(9.5, 7.8))
    sns.heatmap(
        rho_mat,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"shrink": 0.78, "label": "Spearman Rank Correlation (ρ)"},
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 9.5, "weight": "semibold"},
        ax=ax,
    )
    ax.set_title(
        "Spearman Rank Correlation Matrix of Health Financing, Workforce & Outcomes\n"
        f"(2018–2022 Country Averages; pairwise complete n = {n_min}–{n_max})",
        pad=14,
        fontsize=12.5,
    )
    plt.xticks(rotation=35, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    fig.text(
        0.5,
        0.01,
        "Significance: *** p < 0.001, ** p < 0.01, * p < 0.05 (two-sided). Source: World Bank WDI (215 classified countries).",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(FIGURES_DIR / "fig01_spearman_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Income-Tier Distributions (2x2 Boxplots + Strip Points)
# ---------------------------------------------------------------------------
def plot_income_distributions(cs: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))
    panels = [
        (
            axes[0, 0],
            "health_exp_per_capita_usd",
            "(a) Per-Capita Health Expenditure (US$, log scale)",
            "Current US$ per Capita (log scale)",
            True,
            [("USA", (0.14, 0.0)), ("MYS", (0.18, -0.05))],
        ),
        (
            axes[0, 1],
            "physicians_per_1000",
            "(b) Physician Density per 1,000 Population",
            "Physicians per 1,000 Population",
            False,
            [("CUB", (-0.36, -0.2)), ("NOR", (0.14, 0.25))],
        ),
        (
            axes[1, 0],
            "life_expectancy",
            "(c) Life Expectancy at Birth (Years)",
            "Life Expectancy (Years)",
            False,
            [("CAF", (0.16, 0.8)), ("JPN", (-0.34, 1.3)), ("USA", (0.16, -2.5))],
        ),
        (
            axes[1, 1],
            "under5_mortality_per_1000",
            "(d) Under-5 Mortality Rate (per 1,000 Live Births)",
            "Deaths per 1,000 Live Births",
            False,
            [("CAF", (0.16, -8.0)), ("TCD", (0.16, 8.0))],
        ),
    ]

    palette_list = [INCOME_PALETTE[ig] for ig in INCOME_ORDER]
    short_labels = [INCOME_SHORT[ig] for ig in INCOME_ORDER]

    for ax, col, title, ylabel, log_y, annotations in panels:
        sub = cs[["iso3", "country", "income_group", col]].dropna()
        sns.boxplot(
            data=sub,
            x="income_group",
            y=col,
            hue="income_group",
            order=INCOME_ORDER,
            palette=palette_list,
            width=0.52,
            showfliers=False,
            boxprops={"alpha": 0.45, "linewidth": 1.3},
            medianprops={"color": "#111111", "linewidth": 2.0},
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="income_group",
            y=col,
            hue="income_group",
            order=INCOME_ORDER,
            palette=palette_list,
            size=4.2,
            alpha=0.72,
            jitter=0.18,
            legend=False,
            ax=ax,
        )
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:,.0f}"))

        # Sample sizes below x-tick labels
        counts = [int((sub["income_group"] == ig).sum()) for ig in INCOME_ORDER]
        ax.set_xticks(range(len(INCOME_ORDER)))
        ax.set_xticklabels(
            [f"{lbl}\n(n={n})" for lbl, n in zip(short_labels, counts)],
            fontsize=9.5,
        )
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{title}  [total n={len(sub)}]", fontsize=11, pad=8)

        # Annotate notable countries
        ig_to_x = {ig: idx for idx, ig in enumerate(INCOME_ORDER)}
        for iso, (dx, dy) in annotations:
            row = sub[sub["iso3"] == iso]
            if not row.empty:
                r = row.iloc[0]
                x_pos = ig_to_x[r["income_group"]]
                y_val = r[col]
                y_text = y_val * (10 ** dy) if log_y else y_val + dy
                ax.annotate(
                    f"{iso} ({y_val:,.1f})" if col != "health_exp_per_capita_usd" else f"{iso} (${y_val:,.0f})",
                    xy=(x_pos, y_val),
                    xytext=(x_pos + dx, y_text),
                    fontsize=8.5,
                    fontweight="bold",
                    color="#222222",
                    arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#CCCCCC", "alpha": 0.85},
                )

    fig.suptitle(
        "Global Health Financing, Workforce & Outcomes Across World Bank Income Tiers (2018–2022 Means)",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.005,
        "Note: Boxes show IQR with median bar; individual points represent classified countries (excludes Ethiopia & Venezuela, 'Not classified').",
        ha="center",
        fontsize=8.8,
        color="#444444",
    )
    plt.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(FIGURES_DIR / "fig02_income_tier_distributions.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Preston-Style Diminishing Returns Scatter Analysis
# ---------------------------------------------------------------------------
def plot_preston_curves(cs: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.4))

    def scatter_by_income(ax, df, xcol, ycol):
        for ig in INCOME_ORDER:
            g = df[df["income_group"] == ig]
            ax.scatter(
                g[xcol],
                g[ycol],
                label=ig,
                color=INCOME_PALETTE[ig],
                alpha=0.8,
                s=46,
                edgecolors="white",
                linewidths=0.5,
            )

    # (a) Raw Health Exp vs Life Expectancy (concave curve)
    sub_a = cs[["iso3", "income_group", "health_exp_per_capita_usd", "log10_health_exp_per_capita", "life_expectancy"]].dropna()
    ax = axes[0, 0]
    scatter_by_income(ax, sub_a, "health_exp_per_capita_usd", "life_expectancy")
    # Overlay log-linear curve in raw space
    slope, intercept = np.polyfit(sub_a["log10_health_exp_per_capita"], sub_a["life_expectancy"], 1)
    x_grid = np.linspace(sub_a["health_exp_per_capita_usd"].min(), sub_a["health_exp_per_capita_usd"].max(), 300)
    y_curve = intercept + slope * np.log10(x_grid)
    ax.plot(x_grid, y_curve, color="#222222", lw=2.0, ls="--", label="Log-linear trend")
    lin_r2 = stats.linregress(sub_a["health_exp_per_capita_usd"], sub_a["life_expectancy"]).rvalue ** 2
    log_r2 = stats.linregress(sub_a["log10_health_exp_per_capita"], sub_a["life_expectancy"]).rvalue ** 2
    ax.set_title(f"(a) Raw Spending vs. Life Expectancy (n={len(sub_a)})", fontsize=11)
    ax.set_xlabel("Health Expenditure per Capita (current US$)")
    ax.set_ylabel("Life Expectancy at Birth (Years)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.text(
        0.97, 0.06,
        f"Linear $R^2$ = {lin_r2:.3f}\nLog-linear $R^2$ = {log_r2:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#CCCCCC", "alpha": 0.9},
    )
    for iso, offset in [("USA", (-1700, -5.5)), ("JPN", (850, 1.0)), ("MYS", (1200, -4.5)), ("CAF", (1200, 1.5))]:
        r = sub_a[sub_a["iso3"] == iso]
        if not r.empty:
            row = r.iloc[0]
            ax.annotate(
                iso,
                xy=(row["health_exp_per_capita_usd"], row["life_expectancy"]),
                xytext=(row["health_exp_per_capita_usd"] + offset[0], row["life_expectancy"] + offset[1]),
                fontsize=8.5, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
            )

    # (b) Log10 Health Exp vs Life Expectancy (linearized Preston curve)
    ax = axes[0, 1]
    scatter_by_income(ax, sub_a, "log10_health_exp_per_capita", "life_expectancy")
    lx = np.linspace(sub_a["log10_health_exp_per_capita"].min(), sub_a["log10_health_exp_per_capita"].max(), 200)
    ax.plot(lx, intercept + slope * lx, color="#222222", lw=2.0, ls="--")
    rho_ab, p_ab = stats.spearmanr(sub_a["health_exp_per_capita_usd"], sub_a["life_expectancy"])
    ax.set_title(f"(b) Log10 Spending vs. Life Expectancy (n={len(sub_a)})", fontsize=11)
    ax.set_xlabel(r"$\log_{10}$(Health Expenditure per Capita, USD)")
    ax.set_ylabel("Life Expectancy at Birth (Years)")
    ax.text(
        0.04, 0.93,
        f"Spearman ρ = {rho_ab:.3f} (p < 0.001)\nSlope = +{slope:.2f} yrs per 10× spend",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#CCCCCC", "alpha": 0.9},
    )
    for iso, offset in [("USA", (-0.42, -5.2)), ("JPN", (-0.48, 1.5)), ("CAF", (0.25, 1.0))]:
        r = sub_a[sub_a["iso3"] == iso]
        if not r.empty:
            row = r.iloc[0]
            ax.annotate(
                iso,
                xy=(row["log10_health_exp_per_capita"], row["life_expectancy"]),
                xytext=(row["log10_health_exp_per_capita"] + offset[0], row["life_expectancy"] + offset[1]),
                fontsize=8.5, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
            )

    # (c) Log10 Health Exp vs Under-5 Mortality
    sub_c = cs[["iso3", "income_group", "health_exp_per_capita_usd", "log10_health_exp_per_capita", "under5_mortality_per_1000"]].dropna()
    ax = axes[1, 0]
    scatter_by_income(ax, sub_c, "health_exp_per_capita_usd", "under5_mortality_per_1000")
    rho_c, _ = stats.spearmanr(sub_c["health_exp_per_capita_usd"], sub_c["under5_mortality_per_1000"])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_title(f"(c) Health Spending (log scale) vs. Under-5 Mortality (n={len(sub_c)})", fontsize=11)
    ax.set_xlabel("Health Expenditure per Capita (current US$, log scale)")
    ax.set_ylabel("Under-5 Mortality (per 1,000 Live Births)")
    ax.text(
        0.96, 0.93,
        f"Spearman ρ = {rho_c:.3f} (p < 0.001)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#CCCCCC", "alpha": 0.9},
    )
    for iso, (xm, dy) in [("CAF", (2.2, -12)), ("TCD", (2.2, 12)), ("USA", (0.45, 22))]:
        r = sub_c[sub_c["iso3"] == iso]
        if not r.empty:
            row = r.iloc[0]
            ax.annotate(
                iso,
                xy=(row["health_exp_per_capita_usd"], row["under5_mortality_per_1000"]),
                xytext=(row["health_exp_per_capita_usd"] * xm, row["under5_mortality_per_1000"] + dy),
                fontsize=8.5, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
            )

    # (d) Physician Density vs Life Expectancy
    sub_d = cs[["iso3", "income_group", "physicians_per_1000", "life_expectancy"]].dropna()
    ax = axes[1, 1]
    scatter_by_income(ax, sub_d, "physicians_per_1000", "life_expectancy")
    rho_d, _ = stats.spearmanr(sub_d["physicians_per_1000"], sub_d["life_expectancy"])
    ax.set_title(f"(d) Physician Density vs. Life Expectancy (n={len(sub_d)})", fontsize=11)
    ax.set_xlabel("Physicians per 1,000 Population")
    ax.set_ylabel("Life Expectancy at Birth (Years)")
    ax.text(
        0.96, 0.06,
        f"Spearman ρ = {rho_d:.3f} (p < 0.001)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#CCCCCC", "alpha": 0.9},
    )
    for iso, offset in [("CUB", (-1.5, -5.0)), ("USA", (2.0, -11.5)), ("CAF", (1.2, 1.5)), ("JPN", (1.1, 1.5))]:
        r = sub_d[sub_d["iso3"] == iso]
        if not r.empty:
            row = r.iloc[0]
            ax.annotate(
                iso,
                xy=(row["physicians_per_1000"], row["life_expectancy"]),
                xytext=(row["physicians_per_1000"] + offset[0], row["life_expectancy"] + offset[1]),
                fontsize=8.5, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
            )

    # Shared legend at top
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=True,
        facecolor="white",
        bbox_to_anchor=(0.5, 0.955),
        fontsize=9.5,
    )
    fig.suptitle(
        "Preston-Style Analysis: Health Financing, Workforce & Diminishing Marginal Returns (2018–2022)",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.92])
    fig.savefig(FIGURES_DIR / "fig03_preston_diminishing_returns.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Longitudinal Trends (2000-2022) by Income Group
# ---------------------------------------------------------------------------
def plot_longitudinal_trends(panel: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    metrics = [
        ("health_exp_per_capita_usd", "(a) Health Exp. / Capita (US$, log)", "Median Current US$ (IQR band)", True),
        ("physicians_per_1000", "(b) Physician Density (/ 1,000)", "Median Physicians / 1,000 (IQR band)", False),
        ("life_expectancy", "(c) Life Expectancy at Birth (Years)", "Median Years (IQR band)", False),
    ]

    for ax, (col, title, ylabel, log_y) in zip(axes, metrics):
        for ig in INCOME_ORDER:
            sub = panel[panel["income_group"] == ig]
            agg = (
                sub.groupby("year")[col]
                .agg(
                    median="median",
                    q25=lambda s: s.quantile(0.25),
                    q75=lambda s: s.quantile(0.75),
                )
                .reset_index()
            )
            ax.plot(
                agg["year"],
                agg["median"],
                label=ig,
                color=INCOME_PALETTE[ig],
                linewidth=2.2,
            )
            ax.fill_between(
                agg["year"],
                agg["q25"],
                agg["q75"],
                color=INCOME_PALETTE[ig],
                alpha=0.14,
            )

        # Highlight COVID-19 period
        ax.axvspan(2020, 2022, color="#888888", alpha=0.14, label="COVID-19 (2020–22)" if col == "health_exp_per_capita_usd" else None)
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:,.0f}"))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_xlim(2000, 2022)
        ax.set_xticks([2000, 2005, 2010, 2015, 2020])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=True,
        facecolor="white",
        bbox_to_anchor=(0.5, 0.935),
        fontsize=9.2,
    )
    fig.suptitle(
        "Longitudinal Trajectories of Health Financing, Workforce & Longevity by Income Group (2000–2022)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.005,
        "Note: Lines show annual group medians with shaded IQRs. Income groups reflect World Bank's latest static tier; USD figures are nominal (current US$).",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.87])
    fig.savefig(FIGURES_DIR / "fig04_longitudinal_trends_2000_2022.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: COVID-19 Structural Shift Overview (EDA for RQ3)
# ---------------------------------------------------------------------------
def plot_covid_overview(covid: pd.DataFrame, covid_yearly: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))

    # (a) Distribution of paired differences in health_exp_pct_gdp
    ax = axes[0]
    sub = covid.dropna(subset=["health_exp_pct_gdp_diff"])
    diffs = sub["health_exp_pct_gdp_diff"]
    sns.histplot(diffs, bins=28, kde=True, color="#0072B2", alpha=0.55, edgecolor="white", ax=ax)
    ax.axvline(0, color="#222222", ls="--", lw=1.3, label="Zero change")
    ax.axvline(diffs.median(), color="#D55E00", ls="-", lw=2.0, label=f"Median ({diffs.median():+.2f} pp)")
    ax.set_title(f"(a) Distribution of COVID Shift in Health Exp.\n(% of GDP, 2020–22 minus 2017–19; n={len(sub)})", fontsize=10.5)
    ax.set_xlabel("Change in Health Expenditure (% of GDP, pp)")
    ax.set_ylabel("Number of Countries")
    ax.legend(fontsize=8.8, loc="upper left")
    ax.annotate("SSD (-6.88)", xy=(-6.88, 1.2), xytext=(-6.7, 12), fontsize=8.2, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9})
    ax.annotate("AFG (+6.16)", xy=(6.16, 1.2), xytext=(3.1, 12), fontsize=8.2, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9})

    # (b) Indexed year-by-year medians (2017 = 100) showing 2020 GDP denominator effect
    ax = axes[1]
    base_pct = covid_yearly.loc[covid_yearly["year"] == 2017, "median_health_exp_pct_gdp"].iloc[0]
    base_pc = covid_yearly.loc[covid_yearly["year"] == 2017, "median_health_exp_per_capita_usd"].iloc[0]
    base_gdp = covid_yearly.loc[covid_yearly["year"] == 2017, "median_gdp_per_capita_usd"].iloc[0]

    ax.plot(covid_yearly["year"], covid_yearly["median_health_exp_pct_gdp"] / base_pct * 100,
            marker="o", lw=2.3, color="#D55E00", label="Health Exp. (% GDP)")
    ax.plot(covid_yearly["year"], covid_yearly["median_health_exp_per_capita_usd"] / base_pc * 100,
            marker="s", lw=2.3, color="#0072B2", label="Health Exp. / Capita (US$)")
    ax.plot(covid_yearly["year"], covid_yearly["median_gdp_per_capita_usd"] / base_gdp * 100,
            marker="^", lw=2.1, ls="--", color="#009E73", label="GDP / Capita (US$)")
    ax.axvspan(2019.5, 2022.2, color="#888888", alpha=0.13)
    ax.axhline(100, color="#666666", ls=":", lw=1.0)
    ax.set_title("(b) Indexed Medians (2017 = 100, n=191)\n2020 GDP Drop vs. 2021–22 Spending Surge", fontsize=10.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Index (2017 Median = 100)")
    ax.set_xticks([2017, 2018, 2019, 2020, 2021, 2022])
    ax.legend(fontsize=8.5, loc="upper left")
    ax.annotate(
        "2020 GDP dip\n(-10% vs 2019)",
        xy=(2020, covid_yearly.loc[covid_yearly["year"] == 2020, "median_gdp_per_capita_usd"].iloc[0] / base_gdp * 100),
        xytext=(2019.6, 86.5),
        fontsize=8.2, fontweight="bold", color="#00664B",
        arrowprops={"arrowstyle": "->", "color": "#00664B", "lw": 0.9},
    )
    ax.set_ylim(82, 138)

    # (c) Change in Health Exp (% GDP) by Income Group
    ax = axes[2]
    palette_list = [INCOME_PALETTE[ig] for ig in INCOME_ORDER]
    sns.boxplot(
        data=sub,
        x="income_group",
        y="health_exp_pct_gdp_diff",
        hue="income_group",
        order=INCOME_ORDER,
        palette=palette_list,
        width=0.52,
        showfliers=False,
        boxprops={"alpha": 0.45, "linewidth": 1.3},
        medianprops={"color": "#111111", "linewidth": 2.0},
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=sub,
        x="income_group",
        y="health_exp_pct_gdp_diff",
        hue="income_group",
        order=INCOME_ORDER,
        palette=palette_list,
        size=3.8,
        alpha=0.72,
        jitter=0.18,
        legend=False,
        ax=ax,
    )
    ax.axhline(0, color="#222222", ls="--", lw=1.2)
    counts = [int((sub["income_group"] == ig).sum()) for ig in INCOME_ORDER]
    ax.set_xticks(range(len(INCOME_ORDER)))
    ax.set_xticklabels(
        [f"{INCOME_SHORT[ig]}\n(n={n})" for ig, n in zip(INCOME_ORDER, counts)],
        fontsize=9,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Change in Health Exp. (% GDP, pp)")
    ax.set_title("(c) COVID Shift in Health Exp. (% GDP)\nAcross Income Tiers", fontsize=10.5)
    ax.set_ylim(-4.0, 5.0)
    ax.text(
        0.98, 0.03,
        "y-axis zoomed [-4, +5] pp\n(excludes SSD -6.9, AFG +6.2)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8, color="#555555",
    )

    fig.suptitle(
        "Exploratory Overview of COVID-19 Structural Shift in National Health Financing",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(FIGURES_DIR / "fig05_covid_shift_overview.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    panel, cs, covid = load_datasets()
    overall_df, by_income_df, rho_mat, pval_mat, n_mat, covid_yearly = compute_summary_tables(
        cs, panel, covid
    )

    plot_spearman_heatmap(rho_mat, pval_mat, n_mat)
    plot_income_distributions(cs)
    plot_preston_curves(cs)
    plot_longitudinal_trends(panel)
    plot_covid_overview(covid, covid_yearly)

    # Write consolidated plain-text EDA report
    lines = []
    lines.append("=" * 80)
    lines.append("KIE4033 ASSIGNMENT 1 — EXPLORATORY DATA ANALYSIS SUMMARY")
    lines.append("=" * 80)
    lines.append("\n1. OVERALL SUMMARY STATISTICS (2018-2022 Cross-Section, 215 Classified Countries)")
    lines.append(overall_df.to_string(index=False))

    lines.append("\n\n2. SUMMARY STATISTICS BY WORLD BANK INCOME GROUP (Selected Core Variables)")
    core_sub = by_income_df[
        by_income_df["variable"].isin([
            "health_exp_per_capita_usd",
            "physicians_per_1000",
            "life_expectancy",
            "under5_mortality_per_1000",
        ])
    ]
    lines.append(core_sub.to_string(index=False))

    lines.append("\n\n3. SPEARMAN RANK CORRELATION MATRIX (rho)")
    lines.append(rho_mat.round(3).to_string())

    lines.append("\n\n4. YEAR-BY-YEAR MEDIANS ACROSS 2017-2022 (Complete Financing Countries, n=191)")
    lines.append(covid_yearly.to_string(index=False))

    lines.append("\n\n5. GENERATED EDA FIGURES (saved in figures/)")
    for fname in [
        "fig01_spearman_correlation_heatmap.png",
        "fig02_income_tier_distributions.png",
        "fig03_preston_diminishing_returns.png",
        "fig04_longitudinal_trends_2000_2022.png",
        "fig05_covid_shift_overview.png",
    ]:
        lines.append(f"  - figures/{fname}")

    report_text = "\n".join(lines)
    (RESULTS_DIR / "eda_summary_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nSaved all EDA tables to {RESULTS_DIR} and figures to {FIGURES_DIR}.")


if __name__ == "__main__":
    main()
