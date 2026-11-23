"""
03_statistical_hypothesis_testing.py
------------------------------------
Statistical Hypothesis Testing & Regression Diagnostics for KIE4033 Assignment 1:
Global Health Financing and Workforce Capacity (UN SDG 3.8 & 3.c).

Research Questions Tested:
    RQ1: Disparity across World Bank income tiers (Shapiro-Wilk, Levene,
         Kruskal-Wallis H-test, epsilon-squared effect size, Dunn's Holm post-hoc).
    RQ2: Associations & diminishing returns vs. health outcomes (Spearman rho with
         bootstrap 95% CI, Linear vs. Log-linear comparison, OLS with HC3 robust SEs,
         VIF, Breusch-Pagan, residual normality, Cook's distance, CAF sensitivity check).
    RQ3: COVID-19 structural shift 2017-2019 vs. 2020-2022 (Shapiro-Wilk on paired
         differences, Wilcoxon signed-rank on both % GDP and per-capita USD + life
         expectancy, rank-biserial effect size, income-tier stratification).

Inputs:
    data/processed/cleaned_sdg3_cross_section_2018_2022.csv
    data/processed/cleaned_sdg3_covid_paired.csv

Outputs:
    figures/fig06_rq1_dunn_posthoc_heatmaps.png
    figures/fig07_rq2_ols_diagnostics.png
    figures/fig08_rq3_covid_paired_shift.png
    results/stats/*.csv & results/stats/statistical_analysis_report.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ---------------------------------------------------------------------------
# Configuration & Styling
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results" / "stats"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

INCOME_ORDER = [
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "High income",
]
INCOME_SHORT = ["Low", "Lower Mid", "Upper Mid", "High"]
INCOME_PALETTE = {
    "Low income": "#D55E00",
    "Lower middle income": "#E69F00",
    "Upper middle income": "#009E73",
    "High income": "#0072B2",
}

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.03)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titleweight": "bold",
    "axes.labelweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def bootstrap_ci(data: np.ndarray, stat_fn, n_boot: int = 3000, alpha: float = 0.05):
    n = len(data)
    idx = RNG.integers(0, n, size=(n_boot, n))
    boots = np.array([stat_fn(data[i]) for i in idx])
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, alpha: float = 0.05):
    n = len(x)
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        rhos[b] = stats.spearmanr(x[idx], y[idx]).statistic
    return float(np.percentile(rhos, 100 * alpha / 2)), float(np.percentile(rhos, 100 * (1 - alpha / 2)))


# ---------------------------------------------------------------------------
# RQ1: Disparity Across World Bank Income Tiers
# ---------------------------------------------------------------------------
def run_rq1(cs: pd.DataFrame):
    rq1_vars = [
        "health_exp_per_capita_usd",
        "physicians_per_1000",
        "hospital_beds_per_1000",
        "life_expectancy",
        "under5_mortality_per_1000",
    ]

    shapiro_rows = []
    omnibus_rows = []
    dunn_tables = {}

    for col in rq1_vars:
        sub = cs[["income_group", col]].dropna()
        groups = [sub.loc[sub["income_group"] == ig, col].values for ig in INCOME_ORDER]

        # 1. Shapiro-Wilk normality check per group
        any_non_normal = False
        for ig, g in zip(INCOME_ORDER, groups):
            w_stat, p_sw = stats.shapiro(g)
            if p_sw < 0.05:
                any_non_normal = True
            shapiro_rows.append({
                "variable": col,
                "income_group": ig,
                "n": len(g),
                "shapiro_W": round(w_stat, 4),
                "shapiro_p": p_sw,
                "normal_at_0.05": p_sw >= 0.05,
            })

        # 2. Levene's homogeneity of variance test (Brown-Forsythe median-centered)
        lev_stat, lev_p = stats.levene(*groups, center="median")

        # 3. Kruskal-Wallis H-test & epsilon-squared effect size
        h_stat, kw_p = stats.kruskal(*groups)
        n_total = len(sub)
        k = len(groups)
        eps_sq = (h_stat - k + 1) / (n_total - k)

        omnibus_rows.append({
            "variable": col,
            "n_total": n_total,
            "any_group_non_normal": any_non_normal,
            "levene_stat": round(lev_stat, 4),
            "levene_p": lev_p,
            "equal_var_at_0.05": lev_p >= 0.05,
            "kruskal_H": round(h_stat, 4),
            "kruskal_p": kw_p,
            "epsilon_sq": round(eps_sq, 4),
        })

        # 4. Dunn's post-hoc pairwise comparisons with Holm adjustment
        dunn = sp.posthoc_dunn(sub, val_col=col, group_col="income_group", p_adjust="holm")
        dunn = dunn.reindex(index=INCOME_ORDER, columns=INCOME_ORDER)
        dunn_tables[col] = dunn

    shapiro_df = pd.DataFrame(shapiro_rows)
    omnibus_df = pd.DataFrame(omnibus_rows)

    shapiro_df.to_csv(RESULTS_DIR / "rq1_shapiro_normality.csv", index=False)
    omnibus_df.to_csv(RESULTS_DIR / "rq1_levene_and_kruskal.csv", index=False)
    dunn_tables["health_exp_per_capita_usd"].to_csv(RESULTS_DIR / "rq1_dunn_health_exp.csv")
    dunn_tables["physicians_per_1000"].to_csv(RESULTS_DIR / "rq1_dunn_physicians.csv")

    # Plot Figure 6: Dunn's post-hoc heatmaps for the 2 primary RQ1 variables
    plot_rq1_dunn(
        dunn_tables["health_exp_per_capita_usd"],
        dunn_tables["physicians_per_1000"],
        omnibus_df,
    )
    return shapiro_df, omnibus_df, dunn_tables


def plot_rq1_dunn(dunn_he: pd.DataFrame, dunn_ph: pd.DataFrame, omnibus_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    configs = [
        (axes[0], dunn_he, "health_exp_per_capita_usd", "(a) Per-Capita Health Expenditure"),
        (axes[1], dunn_ph, "physicians_per_1000", "(b) Physician Density (/ 1,000)"),
    ]

    for ax, mat, col, title in configs:
        row = omnibus_df[omnibus_df["variable"] == col].iloc[0]
        # Transform to -log10(p) for color intensity while annotating exact adjusted p-values
        clipped = mat.clip(lower=1e-20)
        color_vals = -np.log10(clipped)
        mask = np.triu(np.ones_like(mat, dtype=bool), k=0)

        annot = pd.DataFrame("", index=INCOME_ORDER, columns=INCOME_ORDER)
        for i in range(len(INCOME_ORDER)):
            for j in range(len(INCOME_ORDER)):
                if i > j:
                    p = mat.iloc[i, j]
                    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                    p_str = "< 0.0001" if p < 1e-4 else f"{p:.4f}"
                    annot.iloc[i, j] = f"{p_str}\n({sig})"

        sns.heatmap(
            color_vals,
            mask=mask,
            annot=annot,
            fmt="",
            cmap="YlGnBu",
            vmin=0,
            vmax=15,
            square=True,
            linewidths=0.8,
            linecolor="white",
            cbar_kws={"label": r"$-\log_{10}(p_{\mathrm{adj}})$", "shrink": 0.8},
            xticklabels=INCOME_SHORT,
            yticklabels=INCOME_SHORT,
            annot_kws={"size": 9.5, "weight": "semibold"},
            ax=ax,
        )
        ax.set_title(
            f"{title}\nKruskal-Wallis H = {row['kruskal_H']:.2f}, "
            f"p = {fmt_p(row['kruskal_p'])}, ε² = {row['epsilon_sq']:.3f} (n={int(row['n_total'])})",
            fontsize=10.5,
            pad=10,
        )
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "RQ1: Dunn's Post-Hoc Pairwise Comparisons (Holm-Adjusted p-Values)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.93])
    fig.savefig(FIGURES_DIR / "fig06_rq1_dunn_posthoc_heatmaps.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# RQ2: Associations, Diminishing Returns & OLS Multiple Regression
# ---------------------------------------------------------------------------
def fit_ols_model(df: pd.DataFrame, y_col: str, x_cols: list, label: str):
    sub = df[["iso3", "country", y_col] + x_cols].dropna().copy()
    X = sm.add_constant(sub[x_cols])
    y = sub[y_col]
    ols = sm.OLS(y, X).fit()
    ols_hc3 = ols.get_robustcov_results(cov_type="HC3")

    # Diagnostics
    resid = ols.resid
    sw_w, sw_p = stats.shapiro(resid)
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, X)
    cooks_d = ols.get_influence().cooks_distance[0]
    n_high_cook = int((cooks_d > 4 / len(sub)).sum())

    # VIFs (exclude constant)
    vifs = {}
    if len(x_cols) > 1:
        for i, col in enumerate(x_cols, start=1):
            vifs[col] = round(float(variance_inflation_factor(X.values, i)), 3)
    else:
        vifs[x_cols[0]] = 1.0

    coef_rows = []
    for i, name in enumerate(["const"] + x_cols):
        coef_rows.append({
            "model": label,
            "outcome": y_col,
            "n": len(sub),
            "r2": round(ols.rsquared, 4),
            "adj_r2": round(ols.rsquared_adj, 4),
            "aic": round(ols.aic, 2),
            "bic": round(ols.bic, 2),
            "term": name,
            "coef": round(float(ols.params.iloc[i]), 4),
            "se_ols": round(float(ols.bse.iloc[i]), 4),
            "se_hc3": round(float(ols_hc3.bse[i]), 4),
            "t_hc3": round(float(ols_hc3.tvalues[i]), 3),
            "p_hc3": float(ols_hc3.pvalues[i]),
            "vif": vifs.get(name, np.nan),
        })

    diag = {
        "model": label,
        "outcome": y_col,
        "n": len(sub),
        "r2": round(ols.rsquared, 4),
        "adj_r2": round(ols.rsquared_adj, 4),
        "aic": round(ols.aic, 2),
        "bic": round(ols.bic, 2),
        "resid_shapiro_W": round(sw_w, 4),
        "resid_shapiro_p": sw_p,
        "bp_lm_stat": round(bp_lm, 4),
        "bp_p": bp_p,
        "max_vif": max(vifs.values()),
        "n_cooks_gt_4_over_n": n_high_cook,
        "max_cooks_d": round(float(np.max(cooks_d)), 4),
    }
    return ols, ols_hc3, sub, cooks_d, coef_rows, diag


def run_rq2(cs: pd.DataFrame):
    # 1. Spearman correlations with bootstrap 95% CIs
    predictors = [
        "health_exp_per_capita_usd",
        "health_exp_pct_gdp",
        "physicians_per_1000",
        "hospital_beds_per_1000",
        "gdp_per_capita_usd",
    ]
    outcomes = ["life_expectancy", "under5_mortality_per_1000"]
    sp_rows = []
    for y_col in outcomes:
        for x_col in predictors:
            sub = cs[[x_col, y_col]].dropna()
            rho, pval = stats.spearmanr(sub[x_col], sub[y_col])
            ci_lo, ci_hi = bootstrap_spearman_ci(sub[x_col].values, sub[y_col].values)
            sp_rows.append({
                "outcome": y_col,
                "predictor": x_col,
                "n": len(sub),
                "spearman_rho": round(rho, 4),
                "ci95_low": round(ci_lo, 4),
                "ci95_high": round(ci_hi, 4),
                "p_value": pval,
            })
    sp_df = pd.DataFrame(sp_rows)
    sp_df.to_csv(RESULTS_DIR / "rq2_spearman_outcomes.csv", index=False)

    # Prepare log10(under5_mortality) for multiplicative mortality model
    cs_mod = cs.copy()
    cs_mod["log10_under5_mortality"] = np.log10(cs_mod["under5_mortality_per_1000"])

    # 2. Fit regression suite
    all_coefs = []
    all_diags = []

    # Model 0a vs 0b: Diminishing returns comparison (linear vs log10 single predictor)
    _, _, _, _, c0a, d0a = fit_ols_model(
        cs_mod, "life_expectancy", ["health_exp_per_capita_usd"], "M0a_Linear_Spend_Only"
    )
    _, _, _, _, c0b, d0b = fit_ols_model(
        cs_mod, "life_expectancy", ["log10_health_exp_per_capita"], "M0b_Log10_Spend_Only"
    )

    # Model 1 (Primary): life_expectancy ~ log10_health_exp + physicians
    m1_ols, _, m1_sub, m1_cooks, c1, d1 = fit_ols_model(
        cs_mod,
        "life_expectancy",
        ["log10_health_exp_per_capita", "physicians_per_1000"],
        "M1_Primary_Spend_Physicians",
    )

    # Model 2 (GDP Control / Collinearity check)
    _, _, _, _, c2, d2 = fit_ols_model(
        cs_mod,
        "life_expectancy",
        ["log10_health_exp_per_capita", "physicians_per_1000", "log10_gdp_per_capita"],
        "M2_GDP_Control",
    )

    # Model 3 (Infrastructure Sensitivity: + hospital beds)
    _, _, _, _, c3, d3 = fit_ols_model(
        cs_mod,
        "life_expectancy",
        ["log10_health_exp_per_capita", "physicians_per_1000", "hospital_beds_per_1000"],
        "M3_Plus_Hospital_Beds",
    )

    # Model 4 (Outlier Sensitivity: M1 excluding CAF)
    cs_no_caf = cs_mod[cs_mod["iso3"] != "CAF"].copy()
    _, _, _, _, c4, d4 = fit_ols_model(
        cs_no_caf,
        "life_expectancy",
        ["log10_health_exp_per_capita", "physicians_per_1000"],
        "M4_Sensitivity_Excl_CAF",
    )

    # Model 5 (Under-5 Mortality outcome, log10-transformed)
    _, _, _, _, c5, d5 = fit_ols_model(
        cs_mod,
        "log10_under5_mortality",
        ["log10_health_exp_per_capita", "physicians_per_1000"],
        "M5_Log10_Under5_Mortality",
    )

    for clist in (c0a, c0b, c1, c2, c3, c4, c5):
        all_coefs.extend(clist)
    for d in (d0a, d0b, d1, d2, d3, d4, d5):
        all_diags.append(d)

    coefs_df = pd.DataFrame(all_coefs)
    diags_df = pd.DataFrame(all_diags)
    coefs_df.to_csv(RESULTS_DIR / "rq2_ols_models_summary.csv", index=False)
    diags_df.to_csv(RESULTS_DIR / "rq2_ols_diagnostics.csv", index=False)

    # Top 10 influential observations in Model 1 (by Cook's distance)
    m1_infl = m1_sub[["iso3", "country", "life_expectancy", "log10_health_exp_per_capita", "physicians_per_1000"]].copy()
    m1_infl["fitted"] = m1_ols.fittedvalues.values
    m1_infl["residual"] = m1_ols.resid.values
    m1_infl["cooks_d"] = m1_cooks
    m1_infl = m1_infl.sort_values("cooks_d", ascending=False)
    m1_infl.head(10).round(4).to_csv(RESULTS_DIR / "rq2_cooks_distance_top10.csv", index=False)

    # Plot Figure 7: OLS Diagnostics for Model 1
    plot_rq2_diagnostics(m1_ols, m1_sub, m1_cooks, d1)
    return sp_df, coefs_df, diags_df, m1_infl.head(10)


def plot_rq2_diagnostics(ols, sub: pd.DataFrame, cooks_d: np.ndarray, diag: dict):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.8))
    fitted = ols.fittedvalues.values
    resid = ols.resid.values
    std_resid = ols.get_influence().resid_studentized_internal

    # (a) Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(fitted, resid, color="#0072B2", alpha=0.72, s=38, edgecolors="white", lw=0.4)
    ax.axhline(0, color="#222222", ls="--", lw=1.2)
    # Highlight top 3 largest absolute residuals with non-overlapping offsets
    top_res_idx = np.argsort(np.abs(resid))[-3:]
    res_offsets = {"CAF": (1.6, -0.5), "LSO": (2.2, -2.5), "NGA": (-4.8, -3.2)}
    for idx in top_res_idx:
        iso = sub["iso3"].iloc[idx]
        dx, dy = res_offsets.get(iso, (1.2, -2.0))
        ax.annotate(
            f"{iso} ({resid[idx]:+.1f})",
            xy=(fitted[idx], resid[idx]),
            xytext=(fitted[idx] + dx, resid[idx] + dy),
            fontsize=8.5, fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
        )
    ax.set_title(f"(a) Residuals vs. Fitted (Breusch-Pagan p = {fmt_p(diag['bp_p'])})", fontsize=10.8)
    ax.set_xlabel("Fitted Life Expectancy (Years)")
    ax.set_ylabel("Residuals (Years)")

    # (b) Normal Q-Q Plot of Standardized Residuals
    ax = axes[0, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(std_resid, dist="norm")
    ax.scatter(osm, osr, color="#009E73", alpha=0.75, s=36, edgecolors="white", lw=0.4)
    ax.plot(osm, slope * osm + intercept, color="#D55E00", lw=1.8, ls="--")
    ax.set_title(f"(b) Normal Q-Q Plot (Shapiro-Wilk p = {fmt_p(diag['resid_shapiro_p'])})", fontsize=10.8)
    ax.set_xlabel("Theoretical Normal Quantiles")
    ax.set_ylabel("Standardized Residuals")

    # (c) Cook's Distance Influence Plot
    ax = axes[1, 0]
    thresh = 4 / len(sub)
    x_idx = np.arange(len(sub))
    ax.vlines(x_idx, 0, cooks_d, color="#0072B2", alpha=0.7, lw=1.2)
    ax.scatter(x_idx, cooks_d, color="#0072B2", s=20, alpha=0.8)
    ax.axhline(thresh, color="#D55E00", ls="--", lw=1.4, label=f"Threshold 4/n = {thresh:.3f}")
    ax.set_ylim(0, float(np.max(cooks_d)) * 1.18)
    top_cook = np.argsort(cooks_d)[-4:]
    cook_offsets = {"CAF": (14, -0.005), "CUB": (12, 0.012), "USA": (-38, 0.025), "NGA": (-14, 0.024)}
    for idx in top_cook:
        iso = sub["iso3"].iloc[idx]
        dx, dy = cook_offsets.get(iso, (8, 0.015))
        ax.annotate(
            f"{iso} ({cooks_d[idx]:.3f})",
            xy=(idx, cooks_d[idx]),
            xytext=(idx + dx, cooks_d[idx] + dy),
            fontsize=8.5, fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
        )
    ax.set_title(f"(c) Cook's Distance Influence Check ({diag['n_cooks_gt_4_over_n']} pts > 4/n)", fontsize=10.8)
    ax.set_xlabel("Country Observation Index (Sorted Alphabetically)")
    ax.set_ylabel("Cook's Distance")
    ax.legend(fontsize=8.8, loc="upper right")

    # (d) Actual vs Predicted Life Expectancy
    ax = axes[1, 1]
    actual = sub["life_expectancy"].values
    ax.scatter(fitted, actual, color="#0072B2", alpha=0.72, s=38, edgecolors="white", lw=0.4)
    lims = [min(fitted.min(), actual.min()) - 2, max(fitted.max(), actual.max()) + 2]
    ax.plot(lims, lims, color="#D55E00", ls="--", lw=1.6, label="45° Perfect Fit")
    for iso, (dx, dy) in [("CAF", (3.0, -1.0)), ("USA", (-2.5, -10.5)), ("JPN", (-7.0, 2.8))]:
        m = sub["iso3"] == iso
        if m.any():
            i = np.where(m.values)[0][0]
            ax.annotate(
                iso,
                xy=(fitted[i], actual[i]),
                xytext=(fitted[i] + dx, actual[i] + dy),
                fontsize=8.5, fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.9},
            )
    ax.set_title(f"(d) Observed vs. Predicted Life Expectancy (Adj. $R^2$ = {diag['adj_r2']:.3f})", fontsize=10.8)
    ax.set_xlabel("Predicted Life Expectancy (Years)")
    ax.set_ylabel("Observed Life Expectancy (Years)")
    ax.legend(fontsize=8.8, loc="upper left")

    fig.suptitle(
        f"RQ2: OLS Regression Diagnostics — Life Expectancy ~ log10(Health Spend) + Physicians (n={diag['n']})",
        fontsize=12.8,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    fig.savefig(FIGURES_DIR / "fig07_rq2_ols_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# RQ3: COVID-19 Structural Shift (Paired Wilcoxon + Income Stratification)
# ---------------------------------------------------------------------------
def run_rq3(covid: pd.DataFrame):
    tests = [
        ("health_exp_pct_gdp", "health_exp_pct_gdp_pre", "health_exp_pct_gdp_post", "health_exp_pct_gdp_diff", "greater"),
        ("health_exp_per_capita_usd", "health_exp_per_capita_usd_pre", "health_exp_per_capita_usd_post", "health_exp_per_capita_usd_diff", "greater"),
        ("life_expectancy", "life_expectancy_pre", "life_expectancy_post", "life_expectancy_diff", "less"),
    ]

    wilcoxon_rows = []
    for label, pre_c, post_c, diff_c, alt in tests:
        sub = covid[[pre_c, post_c, diff_c]].dropna()
        diffs = sub[diff_c].values
        n = len(diffs)

        # Shapiro-Wilk on paired differences
        sw_w, sw_p = stats.shapiro(diffs)

        # Two-sided and directional Wilcoxon signed-rank tests
        res_two = stats.wilcoxon(sub[post_c], sub[pre_c], alternative="two-sided")
        res_dir = stats.wilcoxon(sub[post_c], sub[pre_c], alternative=alt)

        # Exact matched-pairs rank-biserial correlation r_rb = (W+ - W-) / (W+ + W-)
        nz = diffs[diffs != 0]
        ranks = stats.rankdata(np.abs(nz))
        w_plus = float(ranks[nz > 0].sum())
        w_minus = float(ranks[nz < 0].sum())
        r_rb = (w_plus - w_minus) / (w_plus + w_minus)

        # Bootstrap 95% CI for median difference
        ci_lo, ci_hi = bootstrap_ci(diffs, np.median)

        wilcoxon_rows.append({
            "metric": label,
            "n_pairs": n,
            "n_increased": int((diffs > 0).sum()),
            "n_decreased": int((diffs < 0).sum()),
            "median_pre": round(float(sub[pre_c].median()), 3),
            "median_post": round(float(sub[post_c].median()), 3),
            "median_diff": round(float(np.median(diffs)), 3),
            "median_diff_ci95_low": round(ci_lo, 3),
            "median_diff_ci95_high": round(ci_hi, 3),
            "mean_diff": round(float(np.mean(diffs)), 3),
            "diff_shapiro_W": round(sw_w, 4),
            "diff_shapiro_p": sw_p,
            "wilcoxon_stat": float(res_two.statistic),
            "wilcoxon_p_two_sided": float(res_two.pvalue),
            "wilcoxon_alt_dir": alt,
            "wilcoxon_p_one_sided": float(res_dir.pvalue),
            "rank_biserial_r": round(r_rb, 4),
        })

    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    wilcoxon_df.to_csv(RESULTS_DIR / "rq3_wilcoxon_results.csv", index=False)

    # Income-stratified COVID response comparison
    strat_rows = []
    for metric_col in ["health_exp_pct_gdp_diff", "health_exp_per_capita_usd_pct_change", "life_expectancy_diff"]:
        sub = covid[["income_group", metric_col]].dropna()
        groups = [sub.loc[sub["income_group"] == ig, metric_col].values for ig in INCOME_ORDER]
        h_stat, kw_p = stats.kruskal(*groups)
        eps_sq = (h_stat - len(groups) + 1) / (len(sub) - len(groups))
        for ig, g in zip(INCOME_ORDER, groups):
            # Within-group Wilcoxon against 0
            _, w_p = stats.wilcoxon(g)
            strat_rows.append({
                "metric": metric_col,
                "income_group": ig,
                "n": len(g),
                "median": round(float(np.median(g)), 3),
                "iqr": round(float(np.percentile(g, 75) - np.percentile(g, 25)), 3),
                "pct_positive": round(float((g > 0).mean() * 100), 1),
                "within_group_wilcoxon_p": w_p,
                "between_groups_kw_H": round(h_stat, 3),
                "between_groups_kw_p": kw_p,
                "between_groups_eps_sq": round(eps_sq, 4),
            })

    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(RESULTS_DIR / "rq3_income_stratified_covid.csv", index=False)

    # Plot Figure 8: RQ3 Paired Shift & Spending vs GDP Growth Quadrant
    plot_rq3_results(covid, wilcoxon_df)
    return wilcoxon_df, strat_df


def plot_rq3_results(covid: pd.DataFrame, wilcoxon_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))

    # (a) Paired slopegraph of income-group medians: Pre (2017-19) vs COVID (2020-22)
    ax = axes[0]
    sub = covid.dropna(subset=["health_exp_pct_gdp_diff"])
    for ig in INCOME_ORDER:
        g = sub[sub["income_group"] == ig]
        y_pre = g["health_exp_pct_gdp_pre"].median()
        y_post = g["health_exp_pct_gdp_post"].median()
        ax.plot(
            [0, 1],
            [y_pre, y_post],
            marker="o",
            markersize=8,
            lw=2.6,
            color=INCOME_PALETTE[ig],
            label=f"{ig} ({y_pre:.2f}% → {y_post:.2f}%, Δ={y_post - y_pre:+.2f} pp)",
        )
        ax.text(-0.04, y_pre, f"{y_pre:.2f}%", ha="right", va="center", fontsize=8.8, fontweight="bold", color=INCOME_PALETTE[ig])
        ax.text(1.04, y_post, f"{y_post:.2f}%", ha="left", va="center", fontsize=8.8, fontweight="bold", color=INCOME_PALETTE[ig])

    row_gdp = wilcoxon_df[wilcoxon_df["metric"] == "health_exp_pct_gdp"].iloc[0]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre-COVID\n(2017–2019)", "COVID Period\n(2020–2022)"], fontsize=10)
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(4.1, 10.2)
    ax.set_ylabel("Median Health Expenditure (% of GDP)")
    ax.set_title(
        f"(a) Pre- vs. COVID Median Health Spending Share\n"
        f"Overall Wilcoxon p = {fmt_p(row_gdp['wilcoxon_p_two_sided'])}, r_rb = {row_gdp['rank_biserial_r']:+.3f} (n={int(row_gdp['n_pairs'])})",
        fontsize=10.5,
    )
    ax.legend(loc="upper left", fontsize=8.1, frameon=True, facecolor="white")

    # (b) Per-Capita Health Spending % Change vs GDP per Capita % Change (Disentangling Denominator Effect)
    ax = axes[1]
    sub_b = covid.dropna(subset=["health_exp_per_capita_usd_pct_change", "gdp_per_capita_usd_pct_change"])
    for ig in INCOME_ORDER:
        g = sub_b[sub_b["income_group"] == ig]
        ax.scatter(
            g["gdp_per_capita_usd_pct_change"],
            g["health_exp_per_capita_usd_pct_change"],
            color=INCOME_PALETTE[ig],
            alpha=0.78,
            s=42,
            edgecolors="white",
            lw=0.4,
            label=ig,
        )
    ax.axhline(0, color="#444444", ls=":", lw=1.0)
    ax.axvline(0, color="#444444", ls=":", lw=1.0)
    lim_lo, lim_hi = -55, 85
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#222222", ls="--", lw=1.4, label="45° Equal Growth Line")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(-55, 95)

    above = int((sub_b["health_exp_per_capita_usd_pct_change"] > sub_b["gdp_per_capita_usd_pct_change"]).sum())
    q1_cnt = int(((sub_b["gdp_per_capita_usd_pct_change"] > 0) & (sub_b["health_exp_per_capita_usd_pct_change"] > 0)).sum())
    q2_cnt = int(((sub_b["gdp_per_capita_usd_pct_change"] <= 0) & (sub_b["health_exp_per_capita_usd_pct_change"] > 0)).sum())

    ax.text(
        0.03, 0.96,
        f"Health spend outpaced GDP: {above}/{len(sub_b)} ({above/len(sub_b)*100:.1f}%)\n"
        f"Both rose (Q1): {q1_cnt} | GDP fell & Health rose (Q2): {q2_cnt}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#CCCCCC", "alpha": 0.92},
    )
    row_pc = wilcoxon_df[wilcoxon_df["metric"] == "health_exp_per_capita_usd"].iloc[0]
    ax.set_xlabel("GDP per Capita Change: 2020–22 vs. 2017–19 (%)")
    ax.set_ylabel("Health Exp. per Capita Change: 2020–22 vs. 2017–19 (%)")
    ax.set_title(
        f"(b) Real Spending Growth vs. GDP Denominator Effect\n"
        f"Per-Capita Wilcoxon p = {fmt_p(row_pc['wilcoxon_p_two_sided'])}, r_rb = {row_pc['rank_biserial_r']:+.3f} (n={int(row_pc['n_pairs'])}; w/ GDP n={len(sub_b)})",
        fontsize=10.2,
    )
    ax.legend(loc="lower right", fontsize=8.2, frameon=True, facecolor="white")

    fig.suptitle(
        "RQ3: Paired Analysis of COVID-19 Health Financing Shift (2017–2019 vs. 2020–2022)",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(FIGURES_DIR / "fig08_rq3_covid_paired_shift.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Execution & Consolidated Report
# ---------------------------------------------------------------------------
def main():
    cs = pd.read_csv(PROCESSED_DIR / "cleaned_sdg3_cross_section_2018_2022.csv")
    covid = pd.read_csv(PROCESSED_DIR / "cleaned_sdg3_covid_paired.csv")
    for df in (cs, covid):
        df["income_group"] = pd.Categorical(df["income_group"], categories=INCOME_ORDER, ordered=True)

    shapiro_df, omnibus_df, dunn_tables = run_rq1(cs)
    sp_df, coefs_df, diags_df, top_cooks = run_rq2(cs)
    wilcoxon_df, strat_df = run_rq3(covid)

    lines = []
    lines.append("=" * 88)
    lines.append("KIE4033 ASSIGNMENT 1 — STATISTICAL HYPOTHESIS TESTING & DIAGNOSTICS REPORT")
    lines.append("=" * 88)

    lines.append("\n--- RQ1: ASSUMPTION CHECKS (SHAPIRO-WILK NORMALITY PER INCOME GROUP) ---")
    lines.append(shapiro_df[shapiro_df["variable"].isin(["health_exp_per_capita_usd", "physicians_per_1000"])].to_string(index=False))

    lines.append("\n--- RQ1: LEVENE HOMOGENEITY & KRUSKAL-WALLIS OMNIBUS RESULTS ---")
    lines.append(omnibus_df.to_string(index=False))

    lines.append("\n--- RQ1: DUNN'S POST-HOC (HOLM-ADJUSTED P-VALUES) — HEALTH EXP. PER CAPITA ---")
    lines.append(dunn_tables["health_exp_per_capita_usd"].map(fmt_p).to_string())

    lines.append("\n--- RQ1: DUNN'S POST-HOC (HOLM-ADJUSTED P-VALUES) — PHYSICIANS PER 1,000 ---")
    lines.append(dunn_tables["physicians_per_1000"].map(fmt_p).to_string())

    lines.append("\n\n--- RQ2: SPEARMAN RANK CORRELATIONS WITH 95% BOOTSTRAP CI ---")
    lines.append(sp_df.to_string(index=False))

    lines.append("\n--- RQ2: OLS REGRESSION MODEL COMPARISON & DIAGNOSTICS ---")
    lines.append(diags_df.to_string(index=False))

    lines.append("\n--- RQ2: OLS COEFFICIENTS (WITH HC3 ROBUST STANDARD ERRORS & VIF) ---")
    lines.append(coefs_df[["model", "term", "coef", "se_ols", "se_hc3", "t_hc3", "p_hc3", "vif"]].to_string(index=False))

    lines.append("\n--- RQ2: TOP 10 INFLUENTIAL COUNTRIES IN PRIMARY MODEL M1 (COOK'S DISTANCE) ---")
    lines.append(top_cooks.to_string(index=False))

    lines.append("\n\n--- RQ3: PAIRED WILCOXON SIGNED-RANK TESTS (2017-2019 vs. 2020-2022) ---")
    lines.append(wilcoxon_df.to_string(index=False))

    lines.append("\n--- RQ3: INCOME-STRATIFIED COVID FINANCING & LONGEVITY SHIFTS ---")
    lines.append(strat_df.to_string(index=False))

    report_text = "\n".join(lines)
    (RESULTS_DIR / "statistical_analysis_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nSaved all statistical tables to {RESULTS_DIR} and figures to {FIGURES_DIR}.")


if __name__ == "__main__":
    main()
