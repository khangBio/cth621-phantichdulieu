from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs" / "ml_experiment" / "results"
VIS_DIR = Path(
    r"C:\Users\Khangtn\.codex\visualizations\2026\06\27\019f07e4-c17e-7a93-8fad-05ca106b14c6"
)

metrics = pd.read_csv(RESULTS / "test_metrics.csv")
best = metrics.sort_values(["mae", "rmse"]).iloc[0]
column_map = {
    "Naive_Close_t": "pred_naive_close_t",
    "LinearRegression": "pred_linear_regression",
    "RandomForestRegressor": "pred_random_forest",
}
prediction_column = column_map[best["model"]]

pred = pd.read_csv(
    RESULTS / "test_predictions.csv",
    parse_dates=["date", "target_date"],
)

# Stratified sample across the price range for a readable all-test scatter plot.
rank = pred["actual_close_next"].rank(method="first", pct=True)
pred["price_bin"] = pd.cut(rank, bins=np.linspace(0, 1, 21), include_lowest=True)
scatter = (
    pred.groupby("price_bin", observed=True, group_keys=False)
    .apply(lambda group: group.sample(min(len(group), 125), random_state=42), include_groups=False)
    .reset_index(drop=True)
)

aapl = pred.loc[pred["Name"].eq("AAPL")].sort_values("target_date")
payload = {
    "model": str(best["model"]),
    "predictionColumn": prediction_column,
    "metrics": {
        "mae": round(float(best["mae"]), 4),
        "rmse": round(float(best["rmse"]), 4),
        "r2": round(float(best["r2"]), 6),
        "mape": round(float(best["mape_percent"]), 4),
        "rows": int(best["evaluation_rows"]),
    },
    "scatter": [
        [
            round(float(row.actual_close_next), 4),
            round(float(getattr(row, prediction_column)), 4),
            str(row.Name),
            row.target_date.strftime("%Y-%m-%d"),
        ]
        for row in scatter.itertuples(index=False)
    ],
    "aapl": [
        [
            row.target_date.strftime("%Y-%m-%d"),
            round(float(row.actual_close_next), 4),
            round(float(getattr(row, prediction_column)), 4),
        ]
        for row in aapl.itertuples(index=False)
    ],
}

VIS_DIR.mkdir(parents=True, exist_ok=True)
(VIS_DIR / "best-model-comparison-data.json").write_text(
    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    encoding="utf-8",
)
print(json.dumps({"best_model": payload["model"], **payload["metrics"]}, indent=2))
