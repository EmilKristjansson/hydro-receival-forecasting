# Raw Material Receival Forecasting

Forecasting cumulative raw material deliveries ("receivals") to Hydro between **January 1, 2025** and a chosen end date. For each raw material (`rm_id`), the pipeline predicts the total weight (kg) expected to be received over that window.

## Problem

Given historical purchase orders and delivery batches, predict how much of each raw material will actually arrive in 2025. The challenge has two parts:

1. **Is the material still active?** — Many raw materials are only relevant for a limited time period.
2. **If active, how much will arrive?** — Cumulative receivals tend to evolve roughly linearly over a year, but the slope varies by material, supplier, and season.

## Approach

The pipeline combines two complementary components:

- **Continuation classifier** — an ensemble of LightGBM, XGBoost, and CatBoost that estimates the probability a raw material remains active into the next year, based on recent volume, order, and fill-rate trends.
- **Trend model** — a per-material quantile regression (q=0.8) fit on the last 12 months of cumulative weight as a function of day-of-year, then extrapolated into 2025.

The classifier answers *"is it active?"*, the trend model answers *"if active, how much?"*, and the two are merged at the output stage.

## Data

Two input files under `data/kernel/` plus a prediction mapping:

| File | Rows | Description |
|---|---|---|
| `receivals.csv` | ~122,590 | Individual delivery batches with `rm_id`, `product_id`, `purchase_order_id`, `batch_id`, `date_arrival`, `net_weight`, `supplier_id`, `receival_status` |
| `purchase_orders.csv` | ~33,171 | Orders with `purchase_order_id`, `quantity`, `delivery_date`, `created_date_time`, `unit`, `status` |
| `data/prediction_mapping.csv` | — | Maps each prediction `ID` to an `rm_id` and `forecast_end_date` |

Cleaning steps include: converting timestamps to Europe/Oslo time, dropping deleted orders and non-positive quantities, converting PUND (pounds) to KG, and removing rows with missing `rm_id`.

## Feature Engineering

For each `(rm_id, year)`, the pipeline computes rolling averages over 1, 3, 6, and 12 month windows for both delivered volume and ordered quantity, then derives:

- **Fill rates** — delivered ÷ ordered, per window
- **Log trend ratios** — short-vs-medium, medium-vs-long, and short-vs-long comparisons for volume, orders, and fill rate. Positive values indicate acceleration, negative values indicate erosion.
- **Cumulative weight** — running sum per `rm_id` per year
- **Anchored cumulative ratios** — cumulative weight at a fixed anchor date in year `t` compared to the same anchor in year `t+1`, used to build the binary continuation target (label = 1 if next-year level exceeds 30% of current year).

Forward feature selection using LightGBM validation log-loss retains only features that improve out-of-sample performance.

## Models

### Continuation classifier (ensemble)

Three tree-based models are trained on data prior to 2023, validated on 2023, and blended via a grid-searched convex combination that minimizes validation log-loss:

| Model | Config |
|---|---|
| LightGBM | 2000 estimators, lr=0.02, num_leaves=31, λ=5.0, early stopping at 100 |
| XGBoost | lr=0.02, max_depth=6, λ=5.0, `tree_method="hist"`, early stopping at 100 |
| CatBoost | 2000 iterations, depth=6, l2_leaf_reg=5.0, categorical features: `rm_id`, `product_id` |

Final models are refit on train+val combined using each model's best iteration, then used to predict for 2024 → 2025 continuation.

**Top features across models:**
- `log_cumulative_weight_long_trend`
- `log_fill_rate_long_trend`
- `log_cumulative_weight_medium_trend`
- `receival_count_year`
- `rm_id` / `product_id` (CatBoost)

### Trend model

Per `rm_id` and per year in {2022, 2023, 2024}, a `QuantileRegressor(quantile=0.8, alpha=1e-6)` is fit on the last 12 months of `(doy, cum_weight)`. The fitted line is then extended into the next year (`doy + 365`) and re-based to start at zero. Using the upper quantile focuses on the robust upper envelope of observed trajectories rather than the mean.

### Combining the two

The final prediction for each `(rm_id, date)` pair in the mapping uses the trend model's extrapolated cumulative curve, zeroed out for any `rm_id` the classifier marks as non-continuing (unless it appears in the "expected 2025" order list).

A final scaling factor of `0.81` is applied to the output — overestimation is penalized more heavily than underestimation in the evaluation metric, so predictions are deliberately shrunk.

## Pipeline

```
raw CSVs
  ↓  timezone + unit normalization, filter deleted/non-positive
cleaned receivals + orders
  ↓  daily panel per rm_id (2013 → 2025-05-31)
grid_d
  ↓  yearly rolling windows, fill rates, log trends, anchored targets
features_df  →  forward feature selection  →  base_feats
  ↓                                              ↓
continuation classifier              quantile trend model
  (LGBM + XGB + CatBoost blend)        (per rm_id, per year)
  ↓                                              ↓
  └──────────────── merge ──────────────────────┘
                      ↓
         × 0.81 shrinkage  →  submission_model1_final.csv
```

## Requirements

```
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
optuna
matplotlib
seaborn
```

## Usage

1. Place `purchase_orders.csv` and `receivals.csv` in `data/kernel/`, and `prediction_mapping.csv` in `data/`.
2. Run `final_report_model_1.ipynb` top-to-bottom.
3. Output: `submission_model1_final.csv` with columns `ID` and `predicted_weight`.

## Files

```
.
├── final_report_model_1.ipynb   # full pipeline: EDA, features, models, output
├── data/
│   ├── kernel/
│   │   ├── purchase_orders.csv
│   │   └── receivals.csv
│   └── prediction_mapping.csv
└── submission_model1_final.csv  # generated
```

## Notes

- The 0.3 threshold for the continuation label and the 0.81 output shrinkage are hyperparameters chosen on validation performance; they encode asymmetric cost of over- vs. under-prediction and should be revisited if the evaluation metric changes.
- Weekend deliveries are effectively zero in the data; the trend model captures this implicitly through the cumulative weight curve rather than as an explicit feature.
- `rm_id = 3362` is the only material with a non-unique `product_id` mapping; all other mappings are treated as one-to-one.
