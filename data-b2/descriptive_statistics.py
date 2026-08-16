"""Thống kê mô tả và bảng tần suất cho dữ liệu giá cổ phiếu AAPL.

Chạy:
    python descriptive_statistics.py

Đầu ra được lưu trong outputs/thong_ke_aapl/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("all_stocks_5yr.csv")
OUTPUT_DIR = Path("outputs/thong_ke_aapl")
STOCK_CODE = "AAPL"
TARGET = "close"
QUANTITATIVE_COLUMNS = ["open", "high", "low", "close", "volume"]
PERCENTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def mode_text(series: pd.Series) -> tuple[str, int]:
    """Trả về mode và tần số; không gán mode nếu mọi giá trị chỉ xuất hiện một lần."""
    counts = series.value_counts(dropna=True)
    max_frequency = int(counts.iloc[0])
    if max_frequency == 1:
        return "Không có mode", max_frequency
    modes = counts[counts.eq(max_frequency)].index.tolist()
    shown = modes[:10]
    text = "; ".join(f"{value:g}" for value in shown)
    if len(modes) > len(shown):
        text += f"; ... (+{len(modes) - len(shown)} giá trị)"
    return text, max_frequency


def descriptive_row(name: str, series: pd.Series) -> dict[str, object]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    q1, median, q3 = x.quantile([0.25, 0.50, 0.75])
    mean = x.mean()
    std = x.std(ddof=1)
    mode, mode_frequency = mode_text(x)
    row: dict[str, object] = {
        "variable": name,
        "count": int(x.count()),
        "missing": int(series.isna().sum()),
        "mean": mean,
        "median": median,
        "mode": mode,
        "mode_frequency": mode_frequency,
        "min": x.min(),
        "max": x.max(),
        "range": x.max() - x.min(),
        "variance_sample": x.var(ddof=1),
        "std_sample": std,
        "cv_percent": std / mean * 100 if mean != 0 else np.nan,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "skewness": x.skew(),
        "kurtosis_excess": x.kurt(),
    }
    for percentile in PERCENTILES:
        row[f"p{int(percentile * 100):02d}"] = x.quantile(percentile)
    return row


def frequency_table(name: str, series: pd.Series) -> pd.DataFrame:
    """Lập bảng tần suất khoảng đều, số lớp theo quy tắc Sturges."""
    x = pd.to_numeric(series, errors="coerce").dropna()
    n = len(x)
    k = max(1, math.ceil(1 + 3.322 * math.log10(n)))
    if x.min() == x.max():
        edges = np.array([x.min() - 0.5, x.max() + 0.5])
    else:
        edges = np.linspace(x.min(), x.max(), k + 1)
    categories = pd.cut(x, bins=edges, include_lowest=True, duplicates="drop")
    counts = categories.value_counts(sort=False)
    result = pd.DataFrame(
        {
            "variable": name,
            "class_number": np.arange(1, len(counts) + 1),
            "lower_bound": [interval.left for interval in counts.index],
            "upper_bound": [interval.right for interval in counts.index],
            "midpoint": [interval.mid for interval in counts.index],
            "frequency": counts.to_numpy(),
        }
    )
    result["relative_frequency_percent"] = result["frequency"] / n * 100
    result["cumulative_frequency"] = result["frequency"].cumsum()
    result["cumulative_percent"] = result["relative_frequency_percent"].cumsum()
    result["interval"] = [str(interval) for interval in counts.index]
    return result[
        [
            "variable",
            "class_number",
            "interval",
            "lower_bound",
            "upper_bound",
            "midpoint",
            "frequency",
            "relative_frequency_percent",
            "cumulative_frequency",
            "cumulative_percent",
        ]
    ]


def shape_comment(mean: float, median: float, skewness: float) -> str:
    if abs(skewness) < 0.5:
        return "phân phối khá cân đối"
    if skewness >= 1:
        return "phân phối lệch phải mạnh"
    if skewness > 0:
        return "phân phối lệch phải vừa"
    if skewness <= -1:
        return "phân phối lệch trái mạnh"
    return "phân phối lệch trái vừa"


def variability_comment(cv: float) -> str:
    if cv < 15:
        return "mức biến động tương đối thấp"
    if cv < 30:
        return "mức biến động tương đối vừa"
    return "mức biến động tương đối cao"


def build_report(
    data: pd.DataFrame, summary: pd.DataFrame, target_summary: pd.DataFrame
) -> str:
    lines = [
        "# Thống kê mô tả và bảng tần suất – AAPL",
        "",
        f"- Phạm vi: {len(data):,} quan sát của mã {STOCK_CODE}, "
        f"từ {data['date'].min().date()} đến {data['date'].max().date()}.",
        f"- Biến định lượng: {', '.join(QUANTITATIVE_COLUMNS)}.",
        f"- Biến mục tiêu: `{TARGET}` (giá đóng cửa).",
        "- Variance và Standard Deviation được tính theo mẫu (`ddof=1`).",
        "- Bảng tần suất chia khoảng đều, số lớp xác định theo quy tắc Sturges.",
        "",
        "## Nhận xét",
        "",
    ]

    for _, row in summary.iterrows():
        x = data[row["variable"]].dropna()
        lower_fence = row["q1"] - 1.5 * row["iqr"]
        upper_fence = row["q3"] + 1.5 * row["iqr"]
        outlier_count = int(((x < lower_fence) | (x > upper_fence)).sum())
        outlier_pct = outlier_count / len(x) * 100
        lines.append(
            f"- **{row['variable']}**: mean = {row['mean']:,.2f}, "
            f"median = {row['median']:,.2f}, CV = {row['cv_percent']:.2f}%, "
            f"IQR = {row['iqr']:,.2f}. {shape_comment(row['mean'], row['median'], row['skewness']).capitalize()} "
            f"(skewness = {row['skewness']:.2f}) và {variability_comment(row['cv_percent'])}. "
            f"Theo quy tắc 1.5×IQR có {outlier_count:,} quan sát ngoại lệ ({outlier_pct:.2f}%)."
        )

    t = target_summary.iloc[0]
    change_pct = (data[TARGET].iloc[-1] / data[TARGET].iloc[0] - 1) * 100
    lines += [
        "",
        "## Diễn giải biến mục tiêu `close`",
        "",
        f"Giá đóng cửa trung bình là **{t['mean']:,.2f} USD**, trung vị **{t['median']:,.2f} USD**; "
        f"50% số phiên nằm trong khoảng **{t['q1']:,.2f}–{t['q3']:,.2f} USD**. "
        f"Khoảng biến thiên toàn kỳ là **{t['range']:,.2f} USD** (min {t['min']:,.2f}, max {t['max']:,.2f}). "
        f"Mean lớn hơn median cùng skewness {t['skewness']:.2f} cho thấy đuôi phải: các mức giá cao ở giai đoạn sau "
        f"kéo trung bình lên. Giá cuối kỳ thay đổi {change_pct:,.2f}% so với đầu kỳ, vì vậy dữ liệu giá theo thời gian "
        "không nên được diễn giải như một mẫu độc lập, ổn định theo thời gian.",
        "",
        "Các bảng CSV đi kèm chứa số liệu đầy đủ và bảng tần suất cho từng biến.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    data = (
        raw.loc[raw["Name"].eq(STOCK_CODE), ["date", *QUANTITATIVE_COLUMNS]]
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    if data.empty:
        raise ValueError(f"Không tìm thấy dữ liệu cho mã {STOCK_CODE}")

    summary = pd.DataFrame(
        [descriptive_row(column, data[column]) for column in QUANTITATIVE_COLUMNS]
    )
    frequencies = pd.concat(
        [frequency_table(column, data[column]) for column in QUANTITATIVE_COLUMNS],
        ignore_index=True,
    )
    target_summary = summary.loc[
        summary["variable"].eq(TARGET),
        [
            "variable",
            "count",
            "mean",
            "median",
            "std_sample",
            "min",
            "max",
            "q1",
            "q3",
            "range",
            "variance_sample",
            "cv_percent",
            "iqr",
            "skewness",
        ],
    ].copy()

    data.to_csv(OUTPUT_DIR / "aapl_clean.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "thong_ke_bien_dinh_luong.csv", index=False)
    target_summary.to_csv(OUTPUT_DIR / "thong_ke_bien_muc_tieu_close.csv", index=False)
    frequencies.to_csv(OUTPUT_DIR / "bang_tan_suat.csv", index=False)
    report = build_report(data, summary, target_summary)
    (OUTPUT_DIR / "nhan_xet.md").write_text(report, encoding="utf-8")

    metadata = {
        "input": str(INPUT_FILE),
        "stock_code": STOCK_CODE,
        "target": TARGET,
        "quantitative_columns": QUANTITATIVE_COLUMNS,
        "rows": len(data),
        "date_min": str(data["date"].min().date()),
        "date_max": str(data["date"].max().date()),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    workbook_data = {
        "metadata": metadata,
        "summary_columns": summary.columns.tolist(),
        "summary_rows": summary.replace({np.nan: None}).to_dict(orient="records"),
        "target_columns": target_summary.columns.tolist(),
        "target_rows": target_summary.replace({np.nan: None}).to_dict(orient="records"),
        "frequency_columns": frequencies.columns.tolist(),
        "frequency_rows": frequencies.replace({np.nan: None}).to_dict(orient="records"),
        "report": report,
    }
    (OUTPUT_DIR / "workbook_data.json").write_text(
        json.dumps(workbook_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report)


if __name__ == "__main__":
    main()
