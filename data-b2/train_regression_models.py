"""Thực nghiệm dự báo Close ngày kế tiếp trên all_stocks_5yr.

Mô hình:
- Naive persistence baseline: dự báo Close(t+1) = Close(t).
- Linear Regression: chuẩn hóa biến số và one-hot mã cổ phiếu.
- Random Forest Regressor: biến số gốc và mã cổ phiếu ordinal.

Đánh giá:
- Expanding-window validation dựa trên cột cv_fold = 1..5.
- Final test được giữ nguyên, không tham gia lựa chọn mô hình.
- MAE, RMSE, R2 và MAPE.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_DIR / "outputs" / "ml_experiment" / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


NUMERIC_FEATURES = ["open", "high", "low", "close", "volume"]
CATEGORICAL_FEATURES = ["Name"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target_close_next"
REQUIRED_COLUMNS = [
    "date",
    "target_date",
    *FEATURES,
    TARGET,
    "split",
    "cv_fold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train time-series regression models.")
    parser.add_argument(
        "--train-file",
        default="outputs/time_series_split/all_stocks_5yr_train.csv",
    )
    parser.add_argument(
        "--test-file",
        default="outputs/time_series_split/all_stocks_5yr_test.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ml_experiment/results",
    )
    parser.add_argument("--rf-trees", type=int, default=80)
    parser.add_argument("--cv-rf-trees", type=int, default=25)
    parser.add_argument("--rf-max-depth", type=int, default=18)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip expanding-window cross-validation and run final test only.",
    )
    return parser.parse_args()


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()


def validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for label, df in [("train", train_df), ("test", test_df)]:
        missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(f"{label} is missing columns: {missing}")
        if df[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError(f"{label} contains missing values.")
        if not (df["target_date"] > df["date"]).all():
            raise ValueError(f"{label} contains target_date <= feature date.")

    if not train_df["target_date"].max() < test_df["target_date"].min():
        raise ValueError("Time leakage detected between train and test target dates.")
    if not train_df["split"].eq("train").all():
        raise ValueError("Train file contains non-train rows.")
    if not test_df["split"].eq("test").all():
        raise ValueError("Test file contains non-test rows.")
    if not test_df["cv_fold"].eq(-1).all():
        raise ValueError("Final test must have cv_fold = -1.")


def build_linear_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", LinearRegression(n_jobs=-1)),
        ]
    )


def build_random_forest_pipeline(
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=0.8,
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape_percent": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def evaluate_predictions(
    model_name: str,
    evaluation_set: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    fit_seconds: float,
    predict_seconds: float,
    train_rows: int,
) -> dict[str, object]:
    return {
        "model": model_name,
        "evaluation_set": evaluation_set,
        "train_rows": train_rows,
        "evaluation_rows": len(y_true),
        **regression_metrics(y_true, y_pred),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }


def run_expanding_cv(
    train_df: pd.DataFrame,
    linear_template: Pipeline,
    rf_template: Pipeline,
) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    folds = sorted(int(f) for f in train_df["cv_fold"].unique() if int(f) > 0)

    for fold in folds:
        validation = train_df.loc[train_df["cv_fold"].eq(fold)].copy()
        validation_start = validation["target_date"].min()
        fold_train = train_df.loc[train_df["target_date"] < validation_start].copy()

        if not fold_train["target_date"].max() < validation_start:
            raise AssertionError(f"Temporal leakage in fold {fold}.")

        X_train = fold_train[FEATURES]
        y_train = fold_train[TARGET]
        X_validation = validation[FEATURES]
        y_validation = validation[TARGET]

        baseline_pred = X_validation["close"].to_numpy()
        results.append(
            {
                "fold": fold,
                "train_target_start": fold_train["target_date"].min(),
                "train_target_end": fold_train["target_date"].max(),
                "validation_target_start": validation_start,
                "validation_target_end": validation["target_date"].max(),
                **evaluate_predictions(
                    "Naive_Close_t",
                    f"cv_fold_{fold}",
                    y_validation,
                    baseline_pred,
                    0.0,
                    0.0,
                    len(fold_train),
                ),
            }
        )

        for model_name, template in [
            ("LinearRegression", linear_template),
            ("RandomForestRegressor", rf_template),
        ]:
            model = clone(template)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start

            start = time.perf_counter()
            prediction = model.predict(X_validation)
            predict_seconds = time.perf_counter() - start

            results.append(
                {
                    "fold": fold,
                    "train_target_start": fold_train["target_date"].min(),
                    "train_target_end": fold_train["target_date"].max(),
                    "validation_target_start": validation_start,
                    "validation_target_end": validation["target_date"].max(),
                    **evaluate_predictions(
                        model_name,
                        f"cv_fold_{fold}",
                        y_validation,
                        prediction,
                        fit_seconds,
                        predict_seconds,
                        len(fold_train),
                    ),
                }
            )
        print(f"CV fold {fold}/{len(folds)} completed.", flush=True)

    return pd.DataFrame(results)


def train_and_test_final_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    linear_model: Pipeline,
    rf_model: Pipeline,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    metrics: list[dict[str, object]] = []
    predictions = test_df[["date", "target_date", "Name", "close", TARGET]].copy()
    predictions = predictions.rename(columns={TARGET: "actual_close_next"})

    baseline_pred = X_test["close"].to_numpy()
    predictions["pred_naive_close_t"] = baseline_pred
    metrics.append(
        evaluate_predictions(
            "Naive_Close_t",
            "final_test",
            y_test,
            baseline_pred,
            0.0,
            0.0,
            len(train_df),
        )
    )

    for model_name, model, prediction_column, model_file in [
        (
            "LinearRegression",
            linear_model,
            "pred_linear_regression",
            "linear_regression.joblib",
        ),
        (
            "RandomForestRegressor",
            rf_model,
            "pred_random_forest",
            "random_forest.joblib",
        ),
    ]:
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        prediction = model.predict(X_test)
        predict_seconds = time.perf_counter() - start

        predictions[prediction_column] = prediction
        metrics.append(
            evaluate_predictions(
                model_name,
                "final_test",
                y_test,
                prediction,
                fit_seconds,
                predict_seconds,
                len(train_df),
            )
        )
        joblib.dump(model, output_dir / model_file, compress=3)
        print(f"Final {model_name} completed.", flush=True)

    return pd.DataFrame(metrics), predictions


def extract_rf_feature_importance(rf_pipeline: Pipeline) -> pd.DataFrame:
    preprocessing = rf_pipeline.named_steps["preprocessing"]
    feature_names = preprocessing.get_feature_names_out()
    importance = rf_pipeline.named_steps["model"].feature_importances_
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    train_path = resolve_project_path(args.train_file)
    test_path = resolve_project_path(args.test_file)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_columns = REQUIRED_COLUMNS
    train_df = pd.read_csv(
        train_path,
        usecols=use_columns,
        parse_dates=["date", "target_date"],
    )
    test_df = pd.read_csv(
        test_path,
        usecols=use_columns,
        parse_dates=["date", "target_date"],
    )
    validate_data(train_df, test_df)

    linear_template = build_linear_pipeline()
    cv_rf_template = build_random_forest_pipeline(
        n_estimators=args.cv_rf_trees,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        random_state=args.random_state,
    )

    cv_metrics_path = output_dir / "cv_metrics.csv"
    if args.skip_cv:
        pd.DataFrame().to_csv(cv_metrics_path, index=False)
    else:
        cv_metrics = run_expanding_cv(
            train_df,
            linear_template,
            cv_rf_template,
        )
        cv_metrics.to_csv(cv_metrics_path, index=False, date_format="%Y-%m-%d")

    final_linear = build_linear_pipeline()
    final_rf = build_random_forest_pipeline(
        n_estimators=args.rf_trees,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        random_state=args.random_state,
    )
    test_metrics, predictions = train_and_test_final_models(
        train_df,
        test_df,
        final_linear,
        final_rf,
        output_dir,
    )
    test_metrics.to_csv(output_dir / "test_metrics.csv", index=False)
    predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    extract_rf_feature_importance(final_rf).to_csv(
        output_dir / "random_forest_feature_importance.csv",
        index=False,
    )

    config = {
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features": FEATURES,
        "target": TARGET,
        "train_target_start": str(train_df["target_date"].min().date()),
        "train_target_end": str(train_df["target_date"].max().date()),
        "test_target_start": str(test_df["target_date"].min().date()),
        "test_target_end": str(test_df["target_date"].max().date()),
        "rf_trees": args.rf_trees,
        "cv_rf_trees": args.cv_rf_trees,
        "rf_max_depth": args.rf_max_depth,
        "rf_min_samples_leaf": args.rf_min_samples_leaf,
        "random_state": args.random_state,
        "cv_skipped": args.skip_cv,
        "python": sys.version,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Experiment completed.")
    print(test_metrics.to_string(index=False))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
