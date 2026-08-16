# Rainfall Prediction in India — Logistic Regression

Predicting whether it will rain tomorrow, per Indian state, using daily rainfall records (2009–2024) from the IMD.

## Problem

Given today's (and recent) rainfall conditions in a state, predict whether it will rain tomorrow — a binary classification problem solved with Logistic Regression.

## Dataset

- **Source:** [Daily Rainfall Data – India (2009–2024)](https://www.kaggle.com/datasets/wydoinn/daily-rainfall-at-state-level), IMD-derived, via Kaggle.
- 36 states/UTs, daily records, ~205,000 rows.
- Columns: `date`, `state_name`, `actual` (rainfall mm), `normal` (climatological average), `deviation` (% from normal).

## Approach

1. Exploratory data analysis (seasonality, state-level patterns, missing data)
2. Target definition: `RainToday` (actual ≥ 2.5mm, IMD's rainy-day threshold), `RainTomorrow` (next day's `RainToday`, per state)
3. Feature engineering: lag features, rolling rainfall sums, climatology, season, state
4. **Baseline check:** compared against naive persistence ("tomorrow = today") before trusting model accuracy
5. Logistic Regression with scaling, threshold tuning, ROC-AUC, RFECV, k-fold CV, GridSearchCV
6. (Extension) Enrichment with independent weather variables (temperature, humidity, wind, pressure) via the free NASA POWER API

## Results

| Metric | Value |
|---|---|
| Naive persistence baseline | ~84.5% |
| Logistic Regression accuracy | *fill in your result* |
| ROC-AUC | *fill in your result* |

*A note on the baseline:* Indian rainfall is highly autocorrelated, so a trained model needs to be compared against naive persistence, not just the majority-class baseline, to know if it's actually learning anything new.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/rainfall_prediction_india.ipynb
```

## Project structure

```
data/raw/          original dataset
data/processed/    engineered features, cached API data (gitignored)
notebooks/         main analysis notebook
reports/figures/   saved plots
```

## Next steps

- Merge in real weather variables (temperature, humidity, wind, pressure) from NASA POWER API
- Try tree-based models (Random Forest, XGBoost) for the non-linear monsoon-transition effects
- Per-state / per-region models instead of one pooled national model
