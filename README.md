# KIE4033-Data-Analytics-Assignment-1

# Exploratory Data Analysis & Statistical Hypothesis Testing: Global Health Financing and Workforce Capacity (SDG 3.8 & 3.c)

**Course:** KIE4033 – Exploratory Data Analytics  
**Assignment:** Assignment 1 (2026/2027 Semester 1)  
**Theme:** UN Sustainable Development Goal 3 – Good Health and Well-Being (Targets 3.8 & 3.c)  
**Group:** Group XY *(Update with your group number and member names)*  

---

## 1. Project Overview

Achieving **UN SDG 3.8** (Universal Health Coverage and financial risk protection) and **SDG 3.c** (substantially increasing health financing and the recruitment, development, training, and retention of the health workforce) requires equitable investment across nations. However, stark structural disparities persist between high-income and low-income countries.

This project applies **Exploratory Data Analysis (EDA)** and **non-parametric & parametric statistical hypothesis testing** in Python to evaluate how national health financing and healthcare workforce density vary across World Bank income tiers, how strongly they correlate with core population health outcomes (life expectancy and under-5 mortality), and whether the COVID-19 pandemic significantly shifted national health expenditure shares.

### Research Questions
1. **RQ1 (Disparity Across Income Tiers):** Is there a statistically significant difference in per-capita health expenditure (`health_exp_per_capita_usd`) and physician density (`physicians_per_1000`) across World Bank income groups?
2. **RQ2 (Financing & Workforce vs. Health Outcomes):** How strongly do health financing and healthcare workforce/infrastructure capacity correlate with `life_expectancy` and `under5_mortality_per_1000`, and do diminishing marginal returns occur at higher spending levels?
3. **RQ3 (Pandemic Structural Shift):** Did national health expenditure as a share of GDP (`health_exp_pct_gdp`) increase significantly during the COVID-19 period (`2020–2022`) compared to the pre-pandemic baseline (`2017–2019`)?

---

## 2. Dataset Description

* **Source:** World Bank Open Data (`global_health_and_life_expectancy.csv`)
* **Raw Coverage:** 7,378 country-year records spanning 217 countries/territories from 1990 to 2023 (18 variables).
* **Cleaned Analysis Scope:** 4,945 country-year records across **215 classified countries** from **2000 to 2022**.

### Key Variables Used
| Variable | Description | SDG Target Alignment |
| :--- | :--- | :--- |
| `country`, `iso3`, `year` | Country name, ISO-3 code, and observation year | Identifier |
| `region`, `income_group` | World Bank geographic region and income classification | Grouping Variable |
| `health_exp_per_capita_usd` | Current health expenditure per capita (current US$) | SDG 3.8 / 3.c |
| `health_exp_pct_gdp` | Current health expenditure (% of GDP) | SDG 3.8 / 3.c |
| `physicians_per_1000` | Physicians per 1,000 population | SDG 3.c |
| `hospital_beds_per_1000` | Hospital beds per 1,000 population | SDG 3.8 |
| `gdp_per_capita_usd` | Gross Domestic Product per capita (current US$) | Economic Control |
| `life_expectancy` | Life expectancy at birth, total (years) | SDG 3 Outcome |
| `under5_mortality_per_1000` | Under-5 mortality rate (per 1,000 live births) | SDG 3.2 Outcome |

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
├── scripts/  (or notebooks/)
│   ├── 01_data_cleaning.py
│   ├── 02_exploratory_data_analysis.py
│   └── 03_statistical_hypothesis_testing.py
├── figures/
│   └── (Generated EDA and statistical plots)
├── KIE4033_Asgn1_GroupXY.pdf
├── requirements.txt
└── README.md
```

---

## 4. Methodology Summary

### Data Cleaning & Preprocessing (`01_data_cleaning.py`)
1. **Structural Time-Window Filtering (`2000–2022`):** Removed `1990–1999` (where World Bank standardized health expenditure series are 100% missing) and `2023` (due to ~73% reporting lag in workforce indicators).
2. **Income Classification Filtering:** Excluded the 2 `'Not classified'` territories (68 rows) to maintain clean statistical group comparisons across the 4 official World Bank income tiers.
3. **Bounded Within-Country Interpolation:** Applied linear interpolation grouped by `iso3` (`limit=3, limit_area='inside'`) to bridge short 1–3 year survey gaps without fabricating multi-decade missing series.
4. **Feature Engineering & Aggregation:** Created $\log_{10}$-transformed expenditure/GDP variables to account for extreme right-skewness, plus a 5-year cross-sectional country average (`2018–2022`) and a pre/post-COVID paired dataset (`2017–2019` vs. `2020–2022`).

### Exploratory Data Analysis (`02_exploratory_data_analysis.py`)
* Spearman rank correlation matrix across financing, workforce, and health outcome indicators.
* Income-tier distribution comparisons (boxplots/violin plots) highlighting skewness and efficiency outliers.
* Preston-style log-linear scatter analysis illustrating diminishing marginal returns on health spending.
* Longitudinal time-series trends (`2000–2022`) tracking health spending and physician density across income groups.

### Statistical Hypothesis Testing (`03_statistical_hypothesis_testing.py`)
* **Assumption Verification:** Shapiro-Wilk normality tests and Levene's homogeneity of variance tests prior to test selection.
* **RQ1:** Kruskal-Wallis $H$-test across the 4 income groups followed by Dunn's post-hoc pairwise comparisons with Bonferroni/Holm adjustment.
* **RQ2:** Spearman rank correlation ($\rho$) and Ordinary Least Squares (OLS) multiple linear regression modeling life expectancy against $\log_{10}(\text{health\_exp\_per\_capita\_usd})$ and `physicians_per_1000`.
* **RQ3:** Wilcoxon signed-rank test comparing paired national `health_exp_pct_gdp` before (`2017–2019`) and during (`2020–2022`) the COVID-19 pandemic.

---

## 5. How to Reproduce the Analysis

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/KIE4033-SDG3-Health-Analytics.git
   cd KIE4033-SDG3-Health-Analytics
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
