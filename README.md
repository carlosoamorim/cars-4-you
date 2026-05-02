# Cars 4 You — Used Car Price Prediction

**Regression pipeline for predicting used car resale prices, built for a fictional UK car resale company.**

The core challenge was not modelling — it was data quality. Brand and model name fields were filled in by sellers and contained extensive misspellings and inconsistencies (e.g. `"Aud"`, `"udi"`, `"AUD"` all mapping to `"Audi"`). A hybrid fuzzy string matching approach was developed to standardise these before any modelling work began.

---

## Results

| Model | Train MAE | Val MAE | Kaggle MAE |
|---|---|---|---|
| Ridge | £2,472 | £2,467 | £2,507 |
| Gradient Boosting | £1,835 | £1,874 | £1,877 |
| Histogram Gradient Boosting | £1,120 | £1,371 | £1,308 |
| Random Forest | £736 | £1,393 | £1,338 |
| Extra Trees | £526 | £1,310 | £1,293 |
| **Ensemble (0.2 ET + 0.4 RF + 0.4 HGB)** | — | — | **£1,263** |

Final placement: **upper quartile of the Kaggle leaderboard.**

Validation and Kaggle scores are closely aligned across all models, confirming the holdout strategy did not lead to severe under-testing.

---

## Dataset

12 input features from a 2020 UK used car database:

| Feature | Description |
|---|---|
| Brand | Car manufacturer (9 brands) |
| model | Specific car model |
| year | Registration year → converted to **car age** |
| mileage | Total distance travelled (miles) |
| tax | Annual road tax (£) |
| fuelType | Petrol / Diesel / Hybrid / Electric |
| mpg | Average miles per gallon |
| engineSize | Engine displacement (litres) |
| paintQuality% | Mechanic's assessment of paint & hull |
| previousOwners | Number of prior registered owners |
| hasDamage | Boolean damage marker (seller-reported) |
| **price** | Target — purchase price in £ |

---

## Pipeline

### 1. Data Cleaning — Fuzzy String Matching

The most significant preprocessing challenge. Brand and model fields contained hundreds of misspellings, abbreviations, and casing inconsistencies filled in by sellers.

**Approach:** a hybrid scorer combining Levenshtein normalized similarity (70% weight) and token sort ratio (30% weight) was applied to map every entry to its canonical form against a curated dictionary of valid brands and models. A threshold parameter controls matching strictness.

```
"udi" → "Audi"    "BMW 3sereis" → "3SERIES"    "mercdes" → "Mercedes"
```

Year was transformed from registration date to **car age** (`2020 - year`), with pre-1970 entries treated as data entry errors and set to NaN.

### 2. Preprocessing

- **Outlier handling:** MAD-based robust z-scores (threshold = 3.5) with automatic winsorization bounds — chosen over IQR because most features have skewed distributions even after transformation
- **Skew correction:** automatic log transform applied to features with |skew| > 1
- **Missing values:** median/mode imputation fitted on training fold only
- **Encoding:** One-Hot Encoding for categorical features
- **Scaling:** not applied (tree-based models are invariant to monotonic transforms; Ridge performance degradation was noted as a limitation)
- All preprocessing artifacts fitted on training data and applied to validation/test to prevent leakage

### 3. Feature Selection

Three methods evaluated in parallel:
- Spearman correlation with permutation null distribution
- Mutual information regression with permutation null distribution
- Permutation importance via Decision Tree

Feature selection results were computed but ultimately not applied to the final pipeline due to time constraints.

### 4. Model Benchmarking

Five models evaluated using holdout split (70/30), assessed on MAE:
- Ridge Regression
- Gradient Boosting Regressor
- Histogram Gradient Boosting Regressor
- Random Forest Regressor
- Extra Trees Regressor

Histogram Gradient Boosting identified as the best individual model based on bias-variance tradeoff — negligible train/val gap at competitive MAE.

### 5. Ensemble

Weighted average of the three best individual models:

```
Final prediction = 0.2 × ExtraTrees + 0.4 × RandomForest + 0.4 × HistGradientBoosting
```

Weights were tuned empirically against Kaggle scores. Achieved **MAE £1,263** — best result overall.

### 6. Gradio Interface (Open-Ended Section)

An interactive prediction dashboard built with Gradio. Given a set of car inputs, it returns independent predictions from all 5 base models plus the ensemble, and reports a **model disagreement metric** (std across base models) to surface uncertainty.

---

## Project Structure

```
cars4you-price-prediction/
│
├── Grupo57_notebook.ipynb    # Full pipeline: cleaning → preprocessing → modelling → interface
├── train.csv                 # Training data
├── test.csv                  # Test data (no price labels)
├── requirements.txt
└── README.md
```

---

## Reproducing Results

```bash
pip install -r requirements.txt
```

Run the notebook top to bottom in Google Colab (recommended) or locally. `train.csv` and `test.csv` must be in the working directory.

```python
RANDOM_SEED = 1907
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange?style=flat-square&logo=scikit-learn)
![Gradio](https://img.shields.io/badge/Gradio-Interface-ff7c00?style=flat-square)
![Colab](https://img.shields.io/badge/Run_on-Google_Colab-F9AB00?style=flat-square&logo=googlecolab)

---

## Authors

Carlos Amorim · Rodrigo Andrade · Francisco Cerdeira · António Varela

*MSc. Data Science & Advanced Analytics — Nova IMS, 2024/2025*  
*Machine Learning — Group 57*
