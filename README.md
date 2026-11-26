# KIE4033-Data-Analytics-Assignment-1

# Exploratory Data Analysis & Statistical Hypothesis Testing: Global Health Financing and Workforce Capacity (SDG 3.8 & 3.c)

**Course:** KIE4033 – Exploratory Data Analytics  
**Assignment:** Assignment 1 (2026/2027 Semester 1)  
**Theme:** UN Sustainable Development Goal 3 – Good Health and Well-Being (Targets 3.8 & 3.c)  
**Group:** Group 2 | 22108446/1 & 22117429/2  

---

## 1. Project Overview

Achieving **UN SDG 3.8** (Universal Health Coverage and financial risk protection) and **SDG 3.c** (substantially increasing health financing and the recruitment, development, training, and retention of the health workforce) requires equitable investment across nations. However, stark structural disparities persist between high-income and low-income countries.

This project applies **Exploratory Data Analysis (EDA)** and **non-parametric & parametric statistical hypothesis testing** in Python to evaluate how national health financing and healthcare workforce density vary across World Bank income tiers, how strongly they correlate with core population health outcomes (life expectancy and under-5 mortality), and whether the COVID-19 pandemic significantly shifted national health expenditure shares and per-capita spending.

### Research Questions
1. **RQ1 (Disparity Across Income Tiers):** Is there a statistically significant difference in per-capita health expenditure (`health_exp_per_capita_usd`) and physician density (`physicians_per_1000`) across World Bank income groups?
2. **RQ2 (Financing & Workforce vs. Health Outcomes):** How strongly do health financing and healthcare workforce/infrastructure capacity correlate with `life_expectancy` and `under5_mortality_per_1000`, and do diminishing marginal returns occur at higher spending levels?
3. **RQ3 (Pandemic Structural Shift):** Did national health expenditure as a share of GDP (`health_exp_pct_gdp`)—and per-capita health spending (`health_exp_per_capita_usd`)—increase significantly during the COVID-19 period (`2020–2022`) compared to the pre-pandemic baseline (`2017–2019`)?

---

## 2. Dataset Description

* **Source:** World Bank Open Data (`data/raw/global_health_and_life_expectancy.csv`)
* **Raw Coverage:** 7,378 country-year records spanning 217 countries/territories from 1990 to 2023 (18 variables; 0 duplicate country-year pairs).
* **Cleaned Analysis Scope:** 4,945 country-year records across **215 classified countries/territories** from **2000 to 2022** (16 columns including engineered $\log_{10}$ features).

### Key Variables Used
| Variable | Description | SDG Target Alignment |
| :--- | :--- | :--- |
| `country`, `iso3`, `year` | Country/territory name, ISO-3 code, and observation year | Identifier |
| `region`, `income_group` | World Bank geographic region and income classification (latest static tier) | Grouping Variable |
| `population` | Total population | Demographic Context |
| `health_exp_per_capita_usd` | Current health expenditure per capita (current US\$) | SDG 3.8 / 3.c |
| `health_exp_pct_gdp` | Current health expenditure (% of GDP) | SDG 3.8 / 3.c |
| `physicians_per_1000` | Physicians per 1,000 population | SDG 3.c |
| `hospital_beds_per_1000` | Hospital beds per 1,000 population | SDG 3.8 |
| `gdp_per_capita_usd` | Gross Domestic Product per capita (current US\$) | Economic Control |
| `life_expectancy` | Life expectancy at birth, total (years) | SDG 3 Outcome |
| `infant_mortality_per_1000` | Infant mortality rate (per 1,000 live births) | SDG 3.2 Outcome |
| `under5_mortality_per_1000` | Under-5 mortality rate (per 1,000 live births) | SDG 3.2 Outcome |

*(Note: 4 raw columns—`life_expectancy_male`, `life_expectancy_female`, `fertility_rate`, and `measles_immunization_pct`—were excluded during scope selection to keep the focus on health financing, workforce/infrastructure capacity, and headline mortality/longevity outcomes.)*

---

## 3. Repository Structure

```text
├── data/
│   ├── raw/
│   │   └── global_health_and_life_expectancy.csv
│   └── processed/
│       ├── cleaned_sdg3_panel_2000_2022.csv
│       ├── cleaned_sdg3_cross_section_2018_2022.csv
│       └── cleaned_sdg3_covid_paired.csv
├── scripts/
│   ├── 01_data_cleaning.py
│   ├── 02_exploratory_data_analysis.py
│   └── 03_statistical_hypothesis_testing.py
├── figures/
│   └── (Generated EDA and statistical plots: fig01–fig08)
├── results/
│   ├── eda/
│   │   └── (Summary tables, Spearman matrix & eda_summary_report.txt)
│   └── stats/
│       └── (Hypothesis tests, OLS/VIF/Cook's D tables & statistical_analysis_report.txt)
├── requirements.txt
└── README.md
```

---

## 4. Methodology Summary

### Data Cleaning & Preprocessing (`scripts/01_data_cleaning.py`)
1. **String Standardization:** Stripped leading/trailing whitespace across `country`, `iso3`, `region`, and `income_group` (fixing trailing spaces in `'Sub-Saharan Africa'` and `'Latin America & Caribbean'`).
2. **Income Classification Filtering:** Excluded the 2 `'Not classified'` sovereign nations—**Ethiopia (`ETH`)** and **Venezuela, RB (`VEN`)** (68 raw rows)—which are temporarily unclassified in the World Bank's latest income grouping, preserving clean comparisons across the 4 official ordered income tiers (`Low`, `Lower middle`, `Upper middle`, `High`).
3. **Structural Time-Window Filtering (`2000–2022`):** Removed `1990–1999` (where World Bank standardized health expenditure series are 100% missing) and `2023` (due to severe workforce reporting lag: 73.3% missing in `physicians_per_1000` and 66.4% in `hospital_beds_per_1000`).
4. **Strictly Bounded Within-Country Interpolation (`max_gap=3`):** Linearly interpolated **only** strictly interior missing runs of length $\le 3$ years within each `iso3` (leaving gaps longer than 3 years and leading/trailing gaps completely untouched). This bridged 513 short survey gaps in `physicians_per_1000` (missingness reduced from 43.72% to 33.35%), 84 in `hospital_beds_per_1000` (42.12% to 40.42%), and 4 in `gdp_per_capita_usd` (3.09% to 3.01%), without extrapolating across multi-year missing stretches.
5. **Feature Engineering & Analysis-Ready Aggregation:**
   * **Panel (`cleaned_sdg3_panel_2000_2022.csv`):** 4,945 rows × 16 columns, including $\log_{10}$-transformed `health_exp_per_capita_usd` and `gdp_per_capita_usd`.
   * **5-Year Cross-Section (`cleaned_sdg3_cross_section_2018_2022.csv`):** 215 countries × 14 columns averaging `2018–2022` values, with $\log_{10}$ variables computed **after** aggregation so `log10_health_exp_per_capita == log10(health_exp_per_capita_usd)` exactly.
   * **Pre-/Post-COVID Paired (`cleaned_sdg3_covid_paired.csv`):** 215 countries × 17 columns comparing `2017–2019` vs. `2020–2022` averages across `health_exp_pct_gdp`, `health_exp_per_capita_usd`, `gdp_per_capita_usd`, and `life_expectancy` (191 complete financing pairs), enabling downstream separation of real spending growth from the 2020 GDP denominator contraction.

### Exploratory Data Analysis (`scripts/02_exploratory_data_analysis.py`)
* Spearman rank correlation matrix across financing, workforce, and health outcome indicators.
* Income-tier distribution comparisons (boxplots/strip plots) highlighting skewness, efficiency outliers, and crisis-mortality anomalies.
* Preston-style log-linear scatter analysis illustrating diminishing marginal returns on health spending.
* Longitudinal time-series trends (`2000–2022`) tracking health spending and physician density across income groups.

### Statistical Hypothesis Testing (`scripts/03_statistical_hypothesis_testing.py`)
* **Assumption Verification:** Shapiro-Wilk normality tests and Levene's homogeneity of variance tests prior to test selection.
* **RQ1:** Kruskal-Wallis $H$-test across the 4 income groups followed by Dunn's post-hoc pairwise comparisons with Holm adjustment (plus effect sizes $\epsilon^2$).
* **RQ2:** Spearman rank correlation ($\rho$) and Ordinary Least Squares (OLS) multiple linear regression modeling health outcomes against $\log_{10}(\text{health\_exp\_per\_capita\_usd})$ and `physicians_per_1000`, with VIF, residual/Cook's distance diagnostics, and outlier sensitivity checks.
* **RQ3:** Wilcoxon signed-rank tests comparing paired national `health_exp_pct_gdp` and `health_exp_per_capita_usd` before (`2017–2019`) and during (`2020–2022`) the COVID-19 pandemic.

---

## 5. How to Reproduce the Analysis

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/KIE4033-Data-Analytics-Assignment-1.git
   cd KIE4033-Data-Analytics-Assignment-1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Core libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`, `scikit-posthocs`)*

3. **Run the pipeline in order:**
   ```bash
   python scripts/01_data_cleaning.py
   python scripts/02_exploratory_data_analysis.py
   python scripts/03_statistical_hypothesis_testing.py
   ```

---

## 6. References

1. World Bank. (2024). *World Development Indicators: Health Nutrition and Population Statistics*. The World Bank Group. https://databank.worldbank.org/
2. United Nations. (2015). *Transforming our world: the 2030 Agenda for Sustainable Development (Goal 3: Ensure healthy lives and promote well-being for all at all ages)*. https://sdgs.un.org/goals/goal3
3. Preston, S. H. (1975). The changing relation between mortality and level of economic development. *Population Studies*, 29(2), 231–248.
