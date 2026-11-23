"""
01_data_cleaning.py
-------------------
Data cleaning and preparation pipeline for KIE4033 Assignment 1:
Global Health Financing and Workforce Capacity (UN SDG 3.8 & 3.c).

Input:
    data/raw/global_health_and_life_expectancy.csv

Outputs:
    data/processed/cleaned_sdg3_panel_2000_2022.csv
    data/processed/cleaned_sdg3_cross_section_2018_2022.csv
    data/processed/cleaned_sdg3_covid_paired.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "global_health_and_life_expectancy.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def interpolate_short_gaps(series: pd.Series, max_gap: int = 3) -> pd.Series:
    """Linearly interpolate ONLY strictly interior NaN runs of length <= max_gap.

    Note: pandas' built-in ``series.interpolate(limit=max_gap, limit_area='inside')``
    partially fills the first ``max_gap`` values of arbitrarily long gaps (e.g.,
    10-to-15-year gaps) toward a distant future endpoint. Masking runs longer than
    ``max_gap`` ensures only genuine 1-to-3-year intermittent survey gaps are filled.
    """
    is_na = series.isna()
    if not is_na.any() or is_na.all():
        return series

    run_id = is_na.ne(is_na.shift()).cumsum()
    run_lengths = is_na.groupby(run_id).transform("sum")

    interpolated = series.interpolate(method="linear", limit_area="inside")
    long_gap_mask = is_na & (run_lengths > max_gap)
    return interpolated.mask(long_gap_mask)


def main() -> None:
    # 1. Load raw dataset and verify duplicates
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw dataset shape: {df_raw.shape}")
    print(f"Exact duplicate rows: {df_raw.duplicated().sum()}")
    print(f"Duplicate country-year pairs: {df_raw.duplicated(subset=['iso3', 'year']).sum()}")

    # 2. Clean leading/trailing whitespace in categorical/identifier string columns
    # (Fixes trailing spaces in 'Sub-Saharan Africa ' and 'Latin America & Caribbean ')
    str_cols = ["country", "iso3", "region", "income_group"]
    for col in str_cols:
        df_raw[col] = df_raw[col].astype(str).str.strip()

    # 3. Sort chronologically within each country (required for time-series interpolation)
    df = df_raw.sort_values(["iso3", "year"]).reset_index(drop=True)

    # 4. Select variables relevant to SDG 3.8 & 3.c (Financing, Workforce & Health Outcomes)
    selected_cols = [
        "country", "iso3", "year", "region", "income_group", "population",
        "gdp_per_capita_usd", "health_exp_per_capita_usd", "health_exp_pct_gdp",
        "physicians_per_1000", "hospital_beds_per_1000",
        "life_expectancy", "infant_mortality_per_1000", "under5_mortality_per_1000",
    ]
    df = df[selected_cols].copy()

    # 5. Remove 'Not classified' income group (Ethiopia & Venezuela, RB — 68 rows)
    # and set ordered World Bank income tiers
    unclassified = df[df["income_group"] == "Not classified"][["iso3", "country"]].drop_duplicates()
    print(
        f"\nRows removed with 'Not classified' income_group: "
        f"{(df['income_group'] == 'Not classified').sum()} "
        f"({', '.join(f'{r.country} [{r.iso3}]' for r in unclassified.itertuples())})"
    )
    df = df[df["income_group"] != "Not classified"].copy()

    income_order = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    df["income_group"] = pd.Categorical(df["income_group"], categories=income_order, ordered=True)

    # 6. Restrict time window to 2000-2022
    # (1990-1999 has 100% missing health expenditure; 2023 has 73.3% missing physicians_per_1000)
    df_panel = df[df["year"].between(2000, 2022)].copy()
    assert df_panel.groupby("iso3")["year"].nunique().eq(23).all(), (
        "Expected complete 23-year panel (2000-2022) per country before interpolation."
    )
    print(f"Filtered panel shape (2000-2022, {df_panel['iso3'].nunique()} classified countries): {df_panel.shape}")

    # Record missingness BEFORE interpolation
    numeric_cols = [
        "gdp_per_capita_usd", "health_exp_per_capita_usd", "health_exp_pct_gdp",
        "physicians_per_1000", "hospital_beds_per_1000",
        "life_expectancy", "infant_mortality_per_1000", "under5_mortality_per_1000",
    ]
    missing_before = (df_panel[numeric_cols].isnull().mean() * 100).round(2)

    # 7. Strictly bounded within-country linear interpolation (gap length <= 3 years only)
    for col in numeric_cols:
        df_panel[col] = df_panel.groupby("iso3")[col].transform(
            lambda s: interpolate_short_gaps(s, max_gap=3)
        )

    # Print Missing-Value Audit Table for the report
    missing_after = (df_panel[numeric_cols].isnull().mean() * 100).round(2)
    cells_filled = (
        df[df["year"].between(2000, 2022)][numeric_cols].isnull().sum()
        - df_panel[numeric_cols].isnull().sum()
    )
    audit_df = pd.DataFrame({
        "Missing_Before_%": missing_before,
        "Missing_After_%": missing_after,
        "Reduction_%": (missing_before - missing_after).round(2),
        "Cells_Interpolated": cells_filled,
    })
    print("\n--- Missing Value Audit (2000-2022 Panel, Strictly <=3-Year Interior Gaps) ---")
    print(audit_df.to_string())

    # 8. Feature Engineering: Log10 transforms for right-skewed monetary metrics (panel level)
    assert (df_panel["health_exp_per_capita_usd"].dropna() > 0).all()
    assert (df_panel["gdp_per_capita_usd"].dropna() > 0).all()
    df_panel["log10_health_exp_per_capita"] = np.log10(df_panel["health_exp_per_capita_usd"])
    df_panel["log10_gdp_per_capita"] = np.log10(df_panel["gdp_per_capita_usd"])

    # 9. Build Analysis-Ready Dataset A: Cross-Sectional Country Summary (2018-2022 mean)
    # Used for RQ1 (Kruskal-Wallis across income tiers) and RQ2 (Spearman & Regression).
    # Note: log10 variables are computed AFTER aggregating to avoid Jensen's inequality
    # discrepancy between mean(log10(x)) and log10(mean(x)).
    df_cross_section = (
        df_panel[df_panel["year"].between(2018, 2022)]
        .groupby(["country", "iso3", "region", "income_group"], observed=True, as_index=False)[
            numeric_cols
        ]
        .mean()
    )
    df_cross_section["log10_health_exp_per_capita"] = np.log10(
        df_cross_section["health_exp_per_capita_usd"]
    )
    df_cross_section["log10_gdp_per_capita"] = np.log10(
        df_cross_section["gdp_per_capita_usd"]
    )
    print(f"\nCross-sectional dataset shape (2018-2022 averages): {df_cross_section.shape}")
    print(f"Valid country counts per variable (out of {len(df_cross_section)}):")
    print(df_cross_section[numeric_cols].notnull().sum())

    # 10. Build Analysis-Ready Dataset B: Pre-COVID (2017-2019) vs. COVID (2020-2022) Paired Data
    # Used for RQ3 (Wilcoxon Signed-Rank Test). Includes per-capita spending and GDP per capita
    # alongside % of GDP so downstream analysis can disentangle the 2020 GDP denominator effect.
    covid_vars = [
        "health_exp_per_capita_usd",
        "health_exp_pct_gdp",
        "gdp_per_capita_usd",
        "life_expectancy",
    ]
    pre_covid = (
        df_panel[df_panel["year"].between(2017, 2019)]
        .groupby(["country", "iso3", "region", "income_group"], observed=True, as_index=False)[
            covid_vars
        ]
        .mean()
    )
    post_covid = (
        df_panel[df_panel["year"].between(2020, 2022)]
        .groupby(["country", "iso3", "region", "income_group"], observed=True, as_index=False)[
            covid_vars
        ]
        .mean()
    )
    df_covid_paired = pd.merge(
        pre_covid,
        post_covid,
        on=["country", "iso3", "region", "income_group"],
        suffixes=("_pre", "_post"),
    )
    df_covid_paired["health_exp_pct_gdp_diff"] = (
        df_covid_paired["health_exp_pct_gdp_post"] - df_covid_paired["health_exp_pct_gdp_pre"]
    )
    df_covid_paired["health_exp_per_capita_usd_diff"] = (
        df_covid_paired["health_exp_per_capita_usd_post"]
        - df_covid_paired["health_exp_per_capita_usd_pre"]
    )
    df_covid_paired["health_exp_per_capita_usd_pct_change"] = (
        df_covid_paired["health_exp_per_capita_usd_diff"]
        / df_covid_paired["health_exp_per_capita_usd_pre"]
        * 100
    )
    df_covid_paired["gdp_per_capita_usd_pct_change"] = (
        (df_covid_paired["gdp_per_capita_usd_post"] - df_covid_paired["gdp_per_capita_usd_pre"])
        / df_covid_paired["gdp_per_capita_usd_pre"]
        * 100
    )
    df_covid_paired["life_expectancy_diff"] = (
        df_covid_paired["life_expectancy_post"] - df_covid_paired["life_expectancy_pre"]
    )

    print(f"\nPaired COVID dataset shape: {df_covid_paired.shape}")
    print(
        "Complete country pairs for health_exp_pct_gdp:",
        df_covid_paired["health_exp_pct_gdp_diff"].notnull().sum(),
    )
    print(
        "Complete country pairs for health_exp_per_capita_usd:",
        df_covid_paired["health_exp_per_capita_usd_diff"].notnull().sum(),
    )

    # 11. Export cleaned datasets for downstream EDA & Statistical Analysis scripts
    panel_out = PROCESSED_DIR / "cleaned_sdg3_panel_2000_2022.csv"
    cs_out = PROCESSED_DIR / "cleaned_sdg3_cross_section_2018_2022.csv"
    covid_out = PROCESSED_DIR / "cleaned_sdg3_covid_paired.csv"

    df_panel.to_csv(panel_out, index=False)
    df_cross_section.to_csv(cs_out, index=False)
    df_covid_paired.to_csv(covid_out, index=False)
    print(f"\nExported all 3 cleaned CSV files successfully to {PROCESSED_DIR}.")


if __name__ == "__main__":
    main()
