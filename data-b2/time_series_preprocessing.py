"""Chuẩn bị dữ liệu all_stocks_5yr cho bài toán dự báo Close ngày kế tiếp.

Quy trình:
1. Sắp xếp dữ liệu theo mã cổ phiếu và thời gian.
2. Tạo target_close_next và target_date trong từng mã cổ phiếu.
3. Chia train/test theo mốc thời gian của biến mục tiêu, không shuffle.
4. Kiểm tra rò rỉ dữ liệu tương lai.
5. Tạo 5 fold expanding-window trên các ngày duy nhất của tập train.
6. Xuất dữ liệu tổng, train, test, tóm tắt fold và tóm tắt xử lý.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume", "Name"]
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "Name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo target Close ngày kế tiếp và chia dữ liệu theo thời gian."
    )
    parser.add_argument(
        "--input",
        default="all_stocks_5yr.csv",
        help="Đường dẫn tới file CSV nguồn.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/time_series_split",
        help="Thư mục lưu các file CSV đầu ra.",
    )
    parser.add_argument(
        "--cutoff-date",
        default="2017-02-07",
        help="Ngày đầu tiên của mục tiêu thuộc tập test (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Số expanding-window fold được tạo bên trong tập train.",
    )
    return parser.parse_args()


def validate_source(df: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Thiếu cột bắt buộc: {missing_columns}")

    if df.empty:
        raise ValueError("File dữ liệu nguồn không có quan sát.")

    if df["Name"].isna().any():
        raise ValueError("Cột Name có giá trị thiếu; không thể tạo target theo từng mã.")


def create_next_day_target(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo giá Close và ngày giao dịch kế tiếp trong từng mã cổ phiếu."""
    result = df.sort_values(["Name", "date"], kind="stable").reset_index(drop=True)

    duplicated = result.duplicated(["Name", "date"], keep=False)
    if duplicated.any():
        examples = result.loc[duplicated, ["Name", "date"]].head().to_dict("records")
        raise ValueError(f"Phát hiện trùng (Name, date), ví dụ: {examples}")

    grouped = result.groupby("Name", sort=False)
    result["target_close_next"] = grouped["close"].shift(-1)
    result["target_date"] = grouped["date"].shift(-1)
    return result


def static_time_split(
    model_df: pd.DataFrame, cutoff_date: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia theo thời điểm của target để nhãn train không vượt sang test."""
    train_df = model_df.loc[model_df["target_date"] < cutoff_date].copy()
    test_df = model_df.loc[model_df["target_date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Mốc cutoff làm train hoặc test bị rỗng.")

    train_last_target = train_df["target_date"].max()
    test_first_target = test_df["target_date"].min()
    if not train_last_target < test_first_target:
        raise AssertionError(
            "Rò rỉ thời gian: target cuối của train không đứng trước target đầu của test."
        )

    return train_df, test_df


def expanding_date_splits(
    unique_dates: np.ndarray, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tạo expanding-window split tương đương TimeSeriesSplit mặc định.

    Việc chia được thực hiện trên danh sách ngày duy nhất, vì vậy mọi cổ phiếu
    tại cùng một target_date luôn nằm trong cùng một fold.
    """
    n_dates = len(unique_dates)
    if n_splits < 2:
        raise ValueError("n_splits phải từ 2 trở lên.")
    if n_dates <= n_splits:
        raise ValueError("Không đủ ngày để tạo số fold đã yêu cầu.")

    validation_size = n_dates // (n_splits + 1)
    if validation_size == 0:
        raise ValueError("Không đủ ngày cho mỗi validation fold.")

    first_validation_start = n_dates - n_splits * validation_size
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for validation_start in range(
        first_validation_start, n_dates, validation_size
    ):
        train_dates = unique_dates[:validation_start]
        validation_dates = unique_dates[
            validation_start : validation_start + validation_size
        ]
        if len(validation_dates) == 0:
            continue
        splits.append((train_dates, validation_dates))

    if len(splits) != n_splits:
        raise AssertionError(
            f"Số fold tạo được ({len(splits)}) khác n_splits ({n_splits})."
        )
    return splits


def add_cv_fold_labels(
    train_df: pd.DataFrame, test_df: pd.DataFrame, n_splits: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gắn fold validation 1..n; 0 là vùng train ban đầu; -1 là final test."""
    train_result = train_df.copy()
    test_result = test_df.copy()
    train_result["cv_fold"] = 0
    test_result["cv_fold"] = -1

    unique_train_dates = np.array(
        sorted(train_result["target_date"].drop_duplicates().to_numpy())
    )
    splits = expanding_date_splits(unique_train_dates, n_splits)
    summaries: list[dict[str, object]] = []

    for fold_number, (fold_train_dates, fold_validation_dates) in enumerate(
        splits, start=1
    ):
        validation_mask = train_result["target_date"].isin(fold_validation_dates)
        train_result.loc[validation_mask, "cv_fold"] = fold_number

        fold_train_rows = train_result["target_date"].isin(fold_train_dates)
        fold_validation_rows = validation_mask

        train_max = pd.Timestamp(fold_train_dates[-1])
        validation_min = pd.Timestamp(fold_validation_dates[0])
        if not train_max < validation_min:
            raise AssertionError(f"Fold {fold_number} bị rò rỉ thời gian.")

        summaries.append(
            {
                "fold": fold_number,
                "train_target_start": pd.Timestamp(fold_train_dates[0]),
                "train_target_end": train_max,
                "validation_target_start": validation_min,
                "validation_target_end": pd.Timestamp(fold_validation_dates[-1]),
                "train_unique_dates": len(fold_train_dates),
                "validation_unique_dates": len(fold_validation_dates),
                "train_rows": int(fold_train_rows.sum()),
                "validation_rows": int(fold_validation_rows.sum()),
            }
        )

    return train_result, test_result, pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    cutoff_date = pd.Timestamp(args.cutoff_date)

    output_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(input_path, parse_dates=["date"])
    validate_source(source_df)
    input_rows = len(source_df)

    targeted_df = create_next_day_target(source_df)
    no_next_target_rows = int(targeted_df["target_close_next"].isna().sum())
    missing_feature_mask = targeted_df[FEATURE_COLUMNS].isna().any(axis=1)
    missing_feature_rows = int(
        (missing_feature_mask & targeted_df["target_close_next"].notna()).sum()
    )

    model_df = targeted_df.dropna(
        subset=FEATURE_COLUMNS + ["target_close_next", "target_date"]
    ).copy()

    train_df, test_df = static_time_split(model_df, cutoff_date)
    train_df, test_df, fold_summary = add_cv_fold_labels(
        train_df, test_df, args.n_splits
    )

    train_df["split"] = "train"
    test_df["split"] = "test"
    model_ready = (
        pd.concat([train_df, test_df], ignore_index=True)
        .sort_values(["target_date", "Name"], kind="stable")
        .reset_index(drop=True)
    )

    output_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "Name",
        "target_date",
        "target_close_next",
        "split",
        "cv_fold",
    ]
    train_df = train_df[output_columns].sort_values(["target_date", "Name"])
    test_df = test_df[output_columns].sort_values(["target_date", "Name"])
    model_ready = model_ready[output_columns]

    all_path = output_dir / "all_stocks_5yr_model_ready.csv"
    train_path = output_dir / "all_stocks_5yr_train.csv"
    test_path = output_dir / "all_stocks_5yr_test.csv"
    folds_path = output_dir / "time_series_cv_folds.csv"
    summary_path = output_dir / "processing_summary.csv"

    model_ready.to_csv(all_path, index=False, date_format="%Y-%m-%d")
    train_df.to_csv(train_path, index=False, date_format="%Y-%m-%d")
    test_df.to_csv(test_path, index=False, date_format="%Y-%m-%d")
    fold_summary.to_csv(folds_path, index=False, date_format="%Y-%m-%d")

    summary = pd.DataFrame(
        [
            {
                "input_file": str(input_path),
                "cutoff_date": cutoff_date,
                "n_splits": args.n_splits,
                "input_rows": input_rows,
                "stocks": source_df["Name"].nunique(),
                "removed_no_next_target": no_next_target_rows,
                "removed_missing_features": missing_feature_rows,
                "model_ready_rows": len(model_ready),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_ratio": len(train_df) / len(model_ready),
                "test_ratio": len(test_df) / len(model_ready),
                "train_target_start": train_df["target_date"].min(),
                "train_target_end": train_df["target_date"].max(),
                "test_target_start": test_df["target_date"].min(),
                "test_target_end": test_df["target_date"].max(),
                "leakage_check_passed": (
                    train_df["target_date"].max() < test_df["target_date"].min()
                ),
            }
        ]
    )
    summary.to_csv(summary_path, index=False, date_format="%Y-%m-%d")

    print("Time-series preprocessing completed.")
    print(f"Input rows       : {input_rows:,}")
    print(f"Model-ready rows : {len(model_ready):,}")
    print(f"Train rows       : {len(train_df):,}")
    print(f"Test rows        : {len(test_df):,}")
    print(f"Cutoff target    : {cutoff_date.date()}")
    print(f"Leakage check    : PASSED")
    print(f"Output directory : {output_dir}")


if __name__ == "__main__":
    main()
