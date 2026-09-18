# Data and Code: Household Solar Energy Adoption in Botswana (2022 Population and Housing Census)

This repository contains the district-level dataset, analysis code, intermediate results, and figures
underlying two companion manuscripts:

1. **"Spatial Patterns of Household Solar-Cooking Adoption in Botswana: Evidence from the 2022
   Population and Housing Census"**.
2. **"Does Machine Learning Help with Small Administrative Datasets? A Cross-Validated Test on
   Solar-Adoption Forecasting in Botswana"**.

Both papers use the same underlying 28-district dataset described below, but ask different questions:
the first asks *where* solar adoption is concentrated and why (spatial statistics, Moran's I, Getis-Ord
Gi*, binomial regression); the second asks whether machine learning *predicts* adoption better than a
simple linear baseline on a table this small (cross-validation, Random Forest, Gradient Boosting).

## Data provenance and a caveat

The source figures come from Statistics Botswana's 2022 Population and Housing Census report,
*"Utilisation of Green Energy Among Households in Botswana"* (Statistics Botswana, 2024,
<https://statsbots.org.bw>). The district-level percentages, household totals, and urban/rural
classification in `data/botswana_solar_2022.csv` were manually compiled by the authors from that
published report into a single machine-readable table; district centroid coordinates (latitude/longitude)
were added from standard geographic references, as point locations for each district's administrative
centre or largest settlement rather than true polygon centroids (see the Limitations sections of both
papers for what that simplification costs the analysis). This compiled file is *not* an official
Statistics Botswana data product — always cite the original census report for the underlying figures,
and cite this repository for the specific derived table and the analysis built on it.

## Repository structure

```
data/
  botswana_solar_2022.csv              Primary 28-district dataset (see column list below)
  botswana_solar_2022_with_stats.csv   Same data, with Getis-Ord Gi* z-scores/p-values/hotspot class appended
  descriptive_stats.csv                Table 1 of the JESA paper (descriptive statistics)
  predictions.csv                      Out-of-fold Random Forest / linear regression predictions (Energies paper)

code/
  build_dataset.py        Builds botswana_solar_2022.csv from the compiled census figures
  analyze.py              Main analysis: Moran's I, Getis-Ord Gi*, binomial GLM, RF vs. linear 5-fold CV
  analyze_alluses.py      Repeats the spatial analysis for space-heating and water-heating (robustness)
  make_figures.py         Generates the hotspot map and actual-vs-predicted scatter figures
  make_figures2.py        Generates the feature-importance bar chart and error-vs-size figures
  robustness_checks.py    Leave-one-out CV, adoption-rate target, nested-CV hyperparameter tuning
  deeper_checks.py        Naive baseline, Gradient Boosting, paired bootstrap, outlier-removed refit

results/
  glm_binomial_summary.txt      Full statsmodels output for the binomial GLM (JESA Table 3)
  results_summary.txt           Console summary from analyze.py
  alluses_summary.txt           Console summary from analyze_alluses.py
  robustness_results.txt        Console summary from robustness_checks.py (Energies Section 3.2)
  deeper_checks_results.txt     Console summary from deeper_checks.py (Energies Section 3.3)

figures/
  fig1_hotspot_map.png             JESA Figure 1 (Getis-Ord Gi* hotspot map)
  fig2_actual_vs_predicted.png     Energies Figure 1 (actual vs. out-of-fold predicted counts)
  fig3_feature_importance.png      Energies Figure 2 (Random Forest feature importances)
  fig4_error_vs_size.png           Energies Figure 3 (prediction error vs. district size)
```

## Column reference: `botswana_solar_2022.csv`

| Column | Description |
|---|---|
| `District` | Census district or sub-district name (28 rows) |
| `Latitude`, `Longitude` | Administrative-centre/largest-settlement coordinates (decimal degrees) |
| `Solar_Cooking_Pct` | % of households using a solar home system as main energy source for cooking |
| `Solar_HeatingSpace_Pct` | % of households using solar for space heating |
| `Solar_HeatingWater_Pct` | % of households using solar for water heating |
| `Total_Households` | Total households in the district (2022 census) |
| `Urban` | 1 if the district is a formally designated town/city council, else 0 |
| `Solar_Cooking_Count`, `Solar_HeatingSpace_Count`, `Solar_HeatingWater_Count` | Derived counts (percentage × Total_Households) |

## Reproducing the results

Requires Python 3.10+ and the packages in `requirements.txt`. From the repository root:

```bash
pip install -r requirements.txt
cd code

# Rebuild the primary dataset (optional — botswana_solar_2022.csv is already included)
python build_dataset.py

#  paper (spatial statistics, binomial GLM, Random Forest vs. linear baseline)
python analyze.py
python analyze_alluses.py
python make_figures.py

# Energies paper (robustness checks and deeper checks)
python robustness_checks.py
python deeper_checks.py
python make_figures2.py
```

Random seeds are fixed (`numpy.random.seed(42)`, `random_state=42` throughout scikit-learn calls) so
re-running the scripts should reproduce the numbers reported in both papers exactly, aside from Monte
Carlo permutation p-values in the Moran's I / Getis-Ord Gi* tests, which can vary by a small amount
between runs depending on the installed versions of `libpysal`/`esda`.

## License

- **Data** (`data/`): released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see `LICENSE-DATA.txt`.
- **Code** (`code/`): released under the [MIT License](https://opensource.org/licenses/MIT) — see `LICENSE-CODE.txt`.

## How to cite

See `CITATION.cff` for machine-readable citation metadata. Once the companion papers are published,
please cite those directly for the analysis and interpretation, and cite this repository (via its DOI)
for the underlying dataset and code.

## Contact

Mavuna Sebapalo (corresponding author) — mavunas@bac.ac.bw
Botswana Accountancy College (BAC), School of Information Communication Systems, Gaborone, Botswana
