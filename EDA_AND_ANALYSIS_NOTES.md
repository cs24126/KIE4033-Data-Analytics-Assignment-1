# EDA & Statistical Analysis Reference Notes

This reference guide captures all verified data characteristics, anomalies, missingness patterns, and methodological caveats discovered during the pre-EDA data audit. Use it as a checklist when writing `scripts/02_exploratory_data_analysis.py`, `scripts/03_statistical_hypothesis_testing.py`, and the final report (`KIE4033_Asgn1_GroupXY.pdf`).

---

## 1. Verified Dataset Snapshot (Post-Cleaning)

| Dataset | Path | Shape | Scope / Purpose |
| :--- | :--- | :--- | :--- |
| **Panel** | `data/processed/cleaned_sdg3_panel_2000_2022.csv` | 4,945 × 16 | 215 classified countries × 23 years (`2000–2022`). Longitudinal trends & missingness audit. |
| **Cross-Section** | `data/processed/cleaned_sdg3_cross_section_2018_2022.csv` | 215 × 14 | 5-year country means (`2018–2022`). Primary input for **RQ1** and **RQ2**. |
| **COVID Paired** | `data/processed/cleaned_sdg3_covid_paired.csv` | 215 × 17 | Paired pre-COVID (`2017–2019`) vs. COVID (`2020–2022`) averages. Primary input for **RQ3**. |

### Panel Missingness Audit (`2000–2022`, Strictly $\le 3$-Year Interior Gaps Interpolated)
| Variable | Missing Before (%) | Missing After (%) | Reduction (pp) | Cells Interpolated |
| :--- | :---: | :---: | :---: | :---: |
| `gdp_per_capita_usd` | 3.09% | 3.01% | 0.08 | 4 |
| `health_exp_per_capita_usd` | 12.68% | 12.68% | 0.00 | 0 |
| `health_exp_pct_gdp` | 12.64% | 12.64% | 0.00 | 0 |
| `physicians_per_1000` | 43.72% | 33.35% | 10.37 | 513 |
| `hospital_beds_per_1000` | 42.12% | 40.42% | 1.70 | 84 |
| `life_expectancy` | 0.00% | 0.00% | 0.00 | 0 |
| `infant_mortality_per_1000` | 9.77% | 9.77% | 0.00 | 0 |
| `under5_mortality_per_1000` | 9.77% | 9.77% | 0.00 | 0 |

---

## 2. Critical Data Anomalies & Outliers to Watch During EDA

1. **Extreme Crisis-Mortality Outliers (`life_expectancy` & `under5_mortality_per_1000`):**
   * **Central African Republic (`CAF`)** exhibits severe mortality spikes in the UN WPP 2024 / World Bank WDI series: `life_expectancy = 14.67` (2009), `31.53` (2019), and `18.82` (2022; `U5M = 424.6`). Because `2019` and `2022` fall inside `2018–2022`, `CAF`'s cross-sectional mean life expectancy is pulled far below other Low-Income countries.
   * Other historical conflict/famine dips in the `2000–2022` panel include **Somalia (`SOM`, 2011: LE = 32.45)** and **South Sudan (`SSD`, 2015–2017: LE = 35.35–39.76)**.
   * **Action for EDA & RQ2:** Annotate `CAF` explicitly on scatter plots and boxplots; run Cook's distance / studentized residual checks on OLS regressions and perform a **sensitivity analysis (with vs. without `CAF`)** to confirm regression slopes and $R^2$ are not driven by one conflict outlier.

2. **Efficiency & Microstate Outliers:**
   * **United States (`USA`)**: Highest per-capita health spending (`\$12,586` in 2022, `16.5%` of GDP) despite lower life expectancy (`77.4` years) than peers spending half as much (`JPN`, `SGP`, `NOR`). Classic Preston-curve efficiency outlier to annotate.
   * **Cuba (`CUB`)**: World's highest physician density (`9.54` per 1,000 in 2021), with missing `health_exp_per_capita_usd` and `gdp_per_capita_usd` in `2022` (4/5 years available in `2018–2022`).
   * **Monaco (`MCO`) & Tuvalu (`TUV`)**: `MCO` has extreme `gdp_per_capita_usd` (`\$226,052` in 2022) and `hospital_beds_per_1000` (`22.27` in 2013); `TUV` has the highest `health_exp_pct_gdp` (`24.46%` in 2017, largely donor/microstate-driven).
   * **WHO/OECD Definition Break in `physicians_per_1000`:** Note that some series jump abruptly due to reporting-definition shifts (e.g., `USA` jumps from `2.548` in 2018 to `3.482` in 2019). Keep this in mind if inspecting individual country trajectories.

3. **Minor WHO GHED vs. World Bank WDI Accounting Discrepancies:**
   * In 53 panel rows (e.g., Turkmenistan `2003–2006`, Yemen `2018`), `(health_exp_per_capita_usd / gdp_per_capita_usd) * 100` differs from `health_exp_pct_gdp` by >2 percentage points due to exchange-rate or GDP-vintage differences between WHO Global Health Expenditure Database (GHED) and World Bank National Accounts. Do not attempt to "recompute" one from the other; use the official WDI columns directly.

---

## 3. Missingness Structure & Sample-Size Tracking

Always report exact sample sizes (`n`) on every plot subtitle and statistical test table, as complete-case counts vary by indicator:

### Cross-Sectional Coverage (`2018–2022`, out of 215 classified countries)
| Income Group | Total | `health_exp_per_capita_usd` | `physicians_per_1000` | `hospital_beds_per_1000` | `life_expectancy` | `under5_mortality_per_1000` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Low income | 25 | 24 | 22 | 16 | 25 | 25 |
| Lower middle income | 50 | 50 | 48 | 32 | 50 | 50 |
| Upper middle income | 54 | 53 | 48 | 44 | 54 | 54 |
| High income | 86 | 64 | 59 | 57 | 86 | 65 |
| **Total** | **215** | **191** | **177** | **149** | **215** | **194** |

* **Non-Random Missingness in High Income:** 22 of the 24 entities missing `health_exp_per_capita_usd` (and all 21 missing `under5_mortality_per_1000`) are small non-sovereign territories (e.g., Bermuda, Cayman Islands, Hong Kong SAR, Macao SAR, Puerto Rico, Greenland, Gibraltar, Guam, etc.), plus DPRK (`PRK`) and Kosovo (`XKX`). Highlighting this in *Data Understanding* shows strong critical awareness—the missingness is concentrated in non-sovereign territories rather than Low-Income nations.
* **Complete Cases for RQ2 Regression:**
  * `life_expectancy ~ log10_health_exp_per_capita + physicians_per_1000`: **`n = 177`** (Low: 22, Lower-middle: 48, Upper-middle: 48, High: 59).
  * Adding `hospital_beds_per_1000`: drops to **`n = 140`**. Consider keeping `hospital_beds_per_1000` as a secondary/sensitivity predictor rather than shrinking the main model by 37 countries.
* **Unequal Underlying Year Counts in `2018–2022` Averages:**
  * `health_exp_per_capita_usd`: 189 countries have all 5 years, 1 has 4 years, 1 has 3 years, 24 have 0 years.
  * `physicians_per_1000`: 119 have 5 yrs, 23 have 4 yrs, 16 have 3 yrs, 6 have 2 yrs, **13 have only 1 year**, 38 have 0 yrs.
  * `hospital_beds_per_1000`: 110 have 5 yrs, 15 have 4 yrs, 9 have 3 yrs, 5 have 2 yrs, **10 have only 1 year**, 66 have 0 yrs.
  * Mention this in *Data Preparation* as a justification for using a 5-year average (stabilizes sparse workforce reporting) while acknowledging unequal observation density.

---

## 4. Research-Question-Specific Statistical Checklist

### RQ1: Disparity Across Income Tiers (`health_exp_per_capita_usd`, `physicians_per_1000`)
* **Assumption checks (required by rubric):** Run **Shapiro-Wilk** per income group (expect strong right-skewness, $p < 0.05$) and **Levene's test** across groups (expect heteroscedasticity, $p < 0.05$) to formally justify choosing non-parametric tests over one-way ANOVA.
* **Primary test:** **Kruskal-Wallis $H$-test** across the 4 ordered income tiers.
* **Post-hoc:** **Dunn's pairwise test** with **Holm** (or Bonferroni) correction, reporting all 6 pairwise adjusted $p$-values.
* **Effect size & summary stats:** Report **medians and IQRs** alongside means/SDs, plus Kruskal-Wallis effect size ($\epsilon^2 = (H - k + 1) / (n - k)$ or $\eta^2_H$) so you quantify *magnitude* of disparity, not just statistical significance.
* **Visualizations:** Boxplots or violin+strip plots ordered explicitly `Low → Lower middle → Upper middle → High`. Use a **$\log_{10}$ y-axis** (or dual raw + log panel) for `health_exp_per_capita_usd` since High Income reaches `\$12,000+` while Low Income clusters under `\$100`.

### RQ2: Financing & Workforce vs. Health Outcomes (`life_expectancy`, `under5_mortality_per_1000`)
* **Non-linearity & Diminishing Returns:**
  * Plot **Preston-style scatter plots**: raw `health_exp_per_capita_usd` vs. `life_expectancy` (clearly concave/curved) side-by-side with `log10_health_exp_per_capita` vs. `life_expectancy` (linearized), colored by `income_group`.
  * Compare linear vs. log-linear $R^2$ (or fit a curve / LOESS smoother) to statistically demonstrate **diminishing marginal returns** at higher spending tiers.
  * Repeat or pair with `under5_mortality_per_1000` (consider $\log_{10}(\text{U5M})$ as well since U5M is strongly right-skewed).
* **Correlation:** Use **Spearman rank correlation ($\rho$)** with $p$-values (robust to non-linearity and extreme outliers like `CAF` and `USA`), presented as a clean annotated correlation heatmap.
* **OLS Multiple Regression & Diagnostics:**
  * Check **multicollinearity (VIF)** among `log10_health_exp_per_capita`, `physicians_per_1000`, and `log10_gdp_per_capita`. Expect high collinearity between `log10_health_exp_per_capita` and `log10_gdp_per_capita`—compare nested models (e.g., spending + physicians vs. controlling for GDP per capita) and discuss VIF explicitly.
  * Check **OLS assumptions**: residual vs. fitted plot (linearity & homoscedasticity), Q-Q plot + Shapiro-Wilk on residuals, Breusch-Pagan test (use **HC3 heteroscedasticity-robust standard errors** if violated!), and **Cook's distance** (flag `CAF`, `USA`, etc.).
  * Run a **sensitivity model excluding extreme conflict outliers (`CAF`)** to verify robustness.

### RQ3: COVID-19 Structural Shift (`2017–2019` vs. `2020–2022`)
* **Paired Sample Structure:** `n = 191` complete country pairs for financing (`190/191` have full 3-vs-3 year coverage; only **Ukraine (`UKR`)** has 3 pre vs. 2 post years due to 2022 war disruption).
* **Assumption check:** Run Shapiro-Wilk on the paired difference distributions (`health_exp_pct_gdp_diff` and `health_exp_per_capita_usd_diff`)—expect non-normality due to extreme shifts (e.g., Afghanistan `+6.16 pp`, South Sudan `-6.88 pp`), justifying the **Wilcoxon signed-rank test**.
* **Disentangling the 2020 GDP Denominator Effect (High-Value Insight!):**
  * Year-by-year medians across the 191 paired countries reveal two distinct mechanisms:
    * **2019 → 2020:** Median `health_exp_pct_gdp` jumped from `5.79%` to `6.73%` largely because median `gdp_per_capita_usd` **contracted ~10%** (`\$6,188` → `\$5,571`) while median `health_exp_per_capita_usd` rose only modestly (`\$380.5` → `\$386.3`).
    * **2021–2022:** Actual per-capita health spending surged (`\$432.6` in 2021, `\$448.2` in 2022), keeping median `health_exp_pct_gdp` elevated (`6.67%` and `6.58%`) even after GDP rebounded (`\$6,061` and `\$6,627`).
  * **Action:** Run Wilcoxon signed-rank tests on **both** `health_exp_pct_gdp` (median diff `+0.62 pp`, 154 increased / 37 decreased) **and** `health_exp_per_capita_usd` (median diff `+\$48.0` / `+16.4%`, 161 increased / 30 decreased), plus stratify the shift by `income_group` (e.g., Kruskal-Wallis on `health_exp_pct_gdp_diff` across income tiers) to see whether High-Income countries mounted larger fiscal responses than Low-Income countries!
  * Also check `life_expectancy_diff` (`n = 215`, median `-0.25` years, 133 decreased vs. 82 increased) as compelling contextual evidence of the pandemic's direct health impact.

---

## 5. Mandatory Limitations to Discuss in the Report (*Discussion — 15%*)

Explicitly addressing these in your Discussion section directly targets the rubric's *"demonstrates critical thinking; acknowledges limitations and context"* criterion:

1. **Nominal Current US\$ vs. Constant PPP:** Both `health_exp_per_capita_usd` and `gdp_per_capita_usd` are measured in *current US\$* (not inflation-adjusted and not PPP-adjusted). Part of the `2000–2022` upward trend reflects USD inflation, and cross-country comparisons do not account for lower local healthcare labor/service prices in developing nations.
2. **Static `income_group` Classification:** World Bank's latest income classification is applied retroactively across `2000–2022`. Longitudinal charts grouped by `income_group` therefore track countries by their *current* tier (e.g., China is grouped as `Upper middle income` throughout `2000–2022` despite being lower-income in 2000).
3. **Exclusion of Ethiopia (`ETH`) and Venezuela (`VEN`):** Both are major sovereign nations temporarily listed as `'Not classified'` in the latest World Bank classification and were excluded to preserve clean 4-tier statistical comparisons.
4. **Mixed Pre-/Post-COVID Cross-Section (`2018–2022`):** Averaging across 5 years maximizes sparse workforce data coverage (`physicians_per_1000`), but blends 2 pre-pandemic years with 3 pandemic shock years.
5. **Observational / Ecological Design (Correlation $\neq$ Causation):** Country-level aggregates cannot establish causal effects and mask large sub-national urban/rural inequalities; broader social determinants (sanitation, nutrition, education, conflict) also shape `life_expectancy` and `under5_mortality_per_1000`.
6. **Excluded Indicators:** Four raw columns (`measles_immunization_pct`, `life_expectancy_male`, `life_expectancy_female`, `fertility_rate`) were intentionally omitted to keep scope focused on SDG 3.8/3.c financing and workforce metrics; `measles_immunization_pct` and sex-disaggregated longevity are strong candidates for future work.
