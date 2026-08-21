# Rainfall Prediction in India — Logistic Regression

Predicting whether it will rain tomorrow across 34 Indian states and union territories, using 15 years of daily IMD rainfall records (2009–2024), built with Python and Scikit-Learn.

---

## Problem Statement

Given today's rainfall conditions in an Indian state, can we predict whether it will rain tomorrow?

This is a **binary classification problem** — the model outputs either `1` (rain expected tomorrow) or `0` (no rain expected). The dataset only contains daily rainfall measurements (no temperature, humidity, or wind data), so the model learns entirely from rainfall patterns, seasonality, and climatological context.

---

## Dataset

- **Source:** [Daily Rainfall Data – India (2009–2024)](https://www.kaggle.com/datasets/wydoinn/daily-rainfall-data-india-2009-2024), IMD-derived, via Kaggle
- **Coverage:** 36 states/UTs, 2009-01-01 to 2024-07-31, ~205,000 rows
- **Key columns:**
  - `actual` — actual rainfall recorded that day (mm)
  - `normal` — long-term climatological average rainfall for that date/state (mm)
  - `deviation` — % difference between actual and normal rainfall
- **After cleaning:** 187,714 rows across 34 states (2 states with 100% missing data removed)

---

## Approach

### 1. Data Cleaning
- Dropped Andaman & Nicobar Islands and Lakshadweep (100% missing `actual` values — no usable data)
- Dropped residual rows with missing `actual` (~3%, including 170 nationwide blackout days)
- Converted `date` to datetime, sorted chronologically within each state

### 2. Target Variable
- `RainToday` = 1 if `actual ≥ 2.5mm` (IMD's official definition of a rainy day), else 0
- `RainTomorrow` = next day's `RainToday`, shifted **within each state's own timeline** to prevent data leakage across state boundaries

### 3. Feature Engineering
Since the dataset only contains rainfall (no independent weather variables), all features are engineered from the rainfall time series itself:

| Feature | Description |
|---|---|
| `Rain_lag1/2/3` | Whether it rained 1, 2, 3 days ago (persistence pattern) |
| `actual_lag1` | Yesterday's actual rainfall in mm (captures intensity, not just yes/no) |
| `rollsum_3d` | Total rainfall over the past 3 days (short-term momentum) |
| `rollsum_7d` | Total rainfall over the past 7 days (longer wet-spell momentum) |
| `normal` | Climatological average for that date/state |
| `deviation_clipped` | % deviation from normal, clipped to [-100, 500] to remove extreme outliers |
| `month` / `doy` | Month and day-of-year (captures monsoon seasonality) |
| `season` | IMD season: Winter / Summer / Monsoon / PostMonsoon |
| `state_name` | One-hot encoded — lets model learn state-specific baseline behavior |

### 4. Honest Baseline Evaluation
Before training any model, two baselines were computed on the **same test set**:
- **Majority-class baseline** (always predict "no rain"): **67.98%**
- **Naive persistence baseline** (predict tomorrow = today): **84.55%**

The naive persistence baseline is the more meaningful bar — Indian rainfall is highly autocorrelated, so any trained model needs to beat *this*, not just the majority-class floor.

### 5. Modeling Pipeline
- Train/test split: 80/20 with `stratify=y`
- `StandardScaler` applied to numeric features only (not binary/dummy columns)
- `LogisticRegression(solver='liblinear')`
- Threshold tuning, ROC-AUC, 5-fold cross-validation, GridSearchCV

---

## Results

| Metric | Value |
|---|---|
| Majority-class baseline | 67.98% |
| Naive persistence baseline | 84.55% |
| Logistic Regression accuracy | **83.84%** |
| ROC-AUC | **0.9050** |
| Cross-validation mean accuracy | 83.73% |
| Cross-validation std | 0.002 |
| Best C (GridSearchCV) | 0.1 |

### Key findings

**Accuracy:** The model (83.84%) comfortably beats the majority-class baseline but falls just below naive persistence (84.55%). This is an honest and important result — with only rainfall-derived features available, the model can't learn anything fundamentally new beyond what persistence already captures directly.

**ROC-AUC (0.905):** Despite the accuracy ceiling, the model's probability estimates are genuinely strong — it ranks rainy days above non-rainy days well across all possible thresholds. AUC is a more reliable measure here than accuracy since it evaluates performance across all thresholds, not just the default 0.5.

**Stability:** Cross-validation std of 0.002 and virtually no change after GridSearchCV confirm the model is stable — the limiting factor is the feature set, not the model or its settings.

### Threshold tuning

| Threshold | Accuracy | Precision | Recall |
|---|---|---|---|
| 0.2 | 0.7981 | 0.6291 | 0.9007 |
| 0.3 | 0.8235 | 0.6829 | 0.8381 |
| 0.4 | 0.8367 | 0.7300 | 0.7779 |
| 0.5 (default) | 0.8384 | 0.7679 | 0.7100 |
| 0.6 | 0.8331 | 0.8036 | 0.6336 |

For practical use (e.g. agriculture), a threshold of 0.3–0.4 is preferable — missing real rain is more costly than a false alarm.

---

## Project Structure

```
rainfall-prediction-india/
├── data/
│   ├── raw/
│   │   └── daily-rainfall-at-state-level.csv
│   └── processed/                        ← gitignored
├── notebooks/
│   └── rainfall_prediction_india.ipynb
├── reports/
│   └── figures/
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/your-username/rainfall-prediction-india.git
cd rainfall-prediction-india

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle and place it at:
# data/raw/daily-rainfall-at-state-level.csv

# Launch the notebook
jupyter notebook notebooks/rainfall_prediction_india.ipynb
```

---

## Limitations and Next Steps

The current model is constrained by the available data — all features are derived from the same underlying rainfall signal, which is why the model can't significantly beat naive persistence. The path to genuine improvement:

- **Add independent weather variables** — temperature, humidity, wind speed, surface pressure via the free [NASA POWER API](https://power.larc.nasa.gov/) (no signup needed). These carry information the rainfall time series alone can't provide.
- **Try tree-based models** — Random Forest or XGBoost would better capture the non-linear monsoon transition effects that Logistic Regression's linear decision boundary misses.
- **Per-state models** — rainfall dynamics vary enormously between Kerala (heavy monsoon) and Rajasthan (arid). A single pooled national model may be missing state-specific patterns that separate models would catch.

---

## Tools and Libraries

Python · Pandas · NumPy · Scikit-Learn · Matplotlib · Seaborn · Jupyter
