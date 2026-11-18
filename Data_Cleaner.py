import pandas as pd
import numpy as np

# 1. Load raw dataset and verify duplicates
df_raw = pd.read_csv('global_health_and_life_expectancy.csv')
print(f"Raw dataset shape: {df_raw.shape}")
print(f"Exact duplicate rows: {df_raw.duplicated().sum()}")
print(f"Duplicate country-year pairs: {df_raw.duplicated(subset=['iso3', 'year']).sum()}")

# 2. Sort chronologically within each country (required for time-series interpolation)
df = df_raw.sort_values(['iso3', 'year']).reset_index(drop=True)

# 3. Select variables relevant to SDG 3.8 & 3.c (Financing, Workforce & Health Outcomes)
selected_cols = [
    'country', 'iso3', 'year', 'region', 'income_group', 'population',
    'gdp_per_capita_usd', 'health_exp_per_capita_usd', 'health_exp_pct_gdp',
    'physicians_per_1000', 'hospital_beds_per_1000',
    'life_expectancy', 'infant_mortality_per_1000', 'under5_mortality_per_1000'
]
df = df[selected_cols].copy()

# 4. Remove 'Not classified' income group (only 2 countries / 68 rows) & order income tiers
print("\nRows removed with 'Not classified' income_group:", (df['income_group'] == 'Not classified').sum())
df = df[df['income_group'] != 'Not classified'].copy()

income_order = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
df['income_group'] = pd.Categorical(df['income_group'], categories=income_order, ordered=True)

# 5. Restrict time window to 2000-2022
# (1990-1999 has 100% missing health expenditure; 2023 has ~73% missing workforce reporting lag)
df_panel = df[df['year'].between(2000, 2022)].copy()
print(f"Filtered panel shape (2000-2022, 215 classified countries): {df_panel.shape}")

# Record missingness BEFORE interpolation
numeric_cols = [
    'gdp_per_capita_usd', 'health_exp_per_capita_usd', 'health_exp_pct_gdp',
    'physicians_per_1000', 'hospital_beds_per_1000',
    'life_expectancy', 'infant_mortality_per_1000', 'under5_mortality_per_1000'
]
missing_before = (df_panel[numeric_cols].isnull().mean() * 100).round(2)

# 6. Bounded within-country linear interpolation
# Fills intermittent 1-to-3-year survey gaps strictly between two known years
df_panel[numeric_cols] = (
    df_panel.groupby('iso3')[numeric_cols]
    .transform(lambda s: s.interpolate(method='linear', limit=3, limit_area='inside'))
)

# Print Missing-Value Audit Table for your report
missing_after = (df_panel[numeric_cols].isnull().mean() * 100).round(2)
audit_df = pd.DataFrame({
    'Missing_Before_%': missing_before,
    'Missing_After_%': missing_after,
    'Reduction_%': (missing_before - missing_after).round(2)
})
print("\n--- Missing Value Audit (2000-2022 Panel) ---")
print(audit_df)

# 7. Feature Engineering: Log10 transforms for right-skewed monetary metrics
df_panel['log10_health_exp_per_capita'] = np.log10(df_panel['health_exp_per_capita_usd'])
df_panel['log10_gdp_per_capita'] = np.log10(df_panel['gdp_per_capita_usd'])

# 8. Build Analysis-Ready Dataset A: Cross-Sectional Country Summary (2018-2022 mean)
# Used for RQ1 (Kruskal-Wallis across income tiers) and RQ2 (Spearman & Regression)
df_cross_section = (
    df_panel[df_panel['year'].between(2018, 2022)]
    .groupby(['country', 'iso3', 'region', 'income_group'], observed=True, as_index=False)[
        numeric_cols + ['log10_health_exp_per_capita', 'log10_gdp_per_capita']
    ]
    .mean()
)
print(f"\nCross-sectional dataset shape (2018-2022 averages): {df_cross_section.shape}")
print("Valid country counts per variable (out of 215):")
print(df_cross_section[numeric_cols].notnull().sum())

# 9. Build Analysis-Ready Dataset B: Pre-COVID (2017-2019) vs. COVID (2020-2022) Paired Data
# Used for RQ3 (Wilcoxon Signed-Rank Test)
pre_covid = (
    df_panel[df_panel['year'].between(2017, 2019)]
    .groupby(['country', 'iso3', 'region', 'income_group'], observed=True, as_index=False)[
        ['health_exp_per_capita_usd', 'health_exp_pct_gdp', 'life_expectancy']
    ]
    .mean()
)
post_covid = (
    df_panel[df_panel['year'].between(2020, 2022)]
    .groupby(['country', 'iso3', 'region', 'income_group'], observed=True, as_index=False)[
        ['health_exp_per_capita_usd', 'health_exp_pct_gdp', 'life_expectancy']
    ]
    .mean()
)
df_covid_paired = pd.merge(
    pre_covid, post_covid,
    on=['country', 'iso3', 'region', 'income_group'],
    suffixes=('_pre', '_post')
)
df_covid_paired['health_exp_pct_gdp_diff'] = (
    df_covid_paired['health_exp_pct_gdp_post'] - df_covid_paired['health_exp_pct_gdp_pre']
)
print(f"\nPaired COVID dataset shape: {df_covid_paired.shape}")
print("Complete country pairs for health_exp_pct_gdp:", df_covid_paired['health_exp_pct_gdp_diff'].notnull().sum())

# 10. Export cleaned datasets for downstream EDA & Statistical Analysis scripts
df_panel.to_csv('cleaned_sdg3_panel_2000_2022.csv', index=False)
df_cross_section.to_csv('cleaned_sdg3_cross_section_2018_2022.csv', index=False)
df_covid_paired.to_csv('cleaned_sdg3_covid_paired.csv', index=False)
print("\nExported all 3 cleaned CSV files successfully.")