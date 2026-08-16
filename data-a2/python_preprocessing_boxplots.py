"""Tiền xử lý age, balance, duration và vẽ boxplot dữ liệu thô/sạch.

Quy trình:
1. Ép kiểu số; giá trị không hợp lệ trở thành NaN.
2. Điền NaN bằng median của từng biến.
3. Tính Q1, Q3, IQR và capping tại Q1-1.5*IQR, Q3+1.5*IQR.
4. Tạo Min-Max [0, 1] và Standard Z-score từ dữ liệu đã capping.
5. Xuất thống kê, dữ liệu xử lý và ba ảnh boxplot.
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_FILE = Path("bank/bank-full.csv")
OUTPUT_DIR = Path("python_preprocessing_results")
VARIABLES = ["age", "balance", "duration"]
UNITS = {
    "age": "Tuổi (năm)",
    "balance": "Số dư tài khoản",
    "duration": "Thời lượng cuộc gọi (giây)",
}
RAW_COLOR = "#4C78A8"
CLEAN_COLOR = "#F58518"

OUTPUT_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})


def calculate_statistics(series: pd.Series, state: str, variable: str) -> dict:
    """Tính toàn bộ chỉ số yêu cầu; Std dùng độ lệch chuẩn mẫu (ddof=1)."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (numeric < lower_bound) | (numeric > upper_bound)

    return {
        "Variable": variable,
        "State": state,
        "Missing_count": int(numeric.isna().sum()),
        "Mean": valid.mean(),
        "Median": valid.median(),
        "Std": valid.std(ddof=1),
        "Min": valid.min(),
        "Q1": q1,
        "Q3": q3,
        "Max": valid.max(),
        "Lower_bound": lower_bound,
        "Upper_bound": upper_bound,
        "Outlier_count": int(outlier_mask.sum()),
    }


def minmax_scale(series: pd.Series) -> pd.Series:
    """Đưa dữ liệu về [0, 1]."""
    minimum, maximum = series.min(), series.max()
    if maximum == minimum:
        return pd.Series(0.0, index=series.index)
    return (series - minimum) / (maximum - minimum)


def zscore_scale(series: pd.Series) -> pd.Series:
    """Z-score kiểu StandardScaler: sử dụng độ lệch chuẩn tổng thể ddof=0."""
    mean = series.mean()
    std_population = series.std(ddof=0)
    if std_population == 0:
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std_population


def draw_boxplot(variable: str, raw: pd.Series, clean: pd.Series,
                 lower_bound: float, upper_bound: float,
                 raw_outliers: int, clean_outliers: int) -> None:
    """Một ảnh cho mỗi biến, gồm boxplot thô và capped đặt song song."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    sns.boxplot(y=raw.dropna(), color=RAW_COLOR, width=0.42,
                fliersize=2.5, linewidth=1.2, ax=axes[0])
    axes[0].axhline(lower_bound, color="#54A24B", linestyle=":", linewidth=1.3,
                    label=f"Lower = {lower_bound:,.2f}")
    axes[0].axhline(upper_bound, color="#E45756", linestyle="--", linewidth=1.3,
                    label=f"Upper = {upper_bound:,.2f}")
    axes[0].set_title(
        f"Biến thô: {variable}\nMissing={raw.isna().sum():,}; Outlier={raw_outliers:,}"
    )
    axes[0].set_xlabel("Dữ liệu thô")
    axes[0].set_ylabel(UNITS[variable])
    axes[0].legend(fontsize=9, loc="best")

    sns.boxplot(y=clean, color=CLEAN_COLOR, width=0.42,
                fliersize=2.5, linewidth=1.2, ax=axes[1])
    axes[1].axhline(lower_bound, color="#54A24B", linestyle=":", linewidth=1.3,
                    label=f"Lower = {lower_bound:,.2f}")
    axes[1].axhline(upper_bound, color="#E45756", linestyle="--", linewidth=1.3,
                    label=f"Upper = {upper_bound:,.2f}")
    axes[1].set_title(
        f"Biến sạch: {variable}_capped\nMissing={clean.isna().sum():,}; Outlier={clean_outliers:,}"
    )
    axes[1].set_xlabel("Sau median fill + IQR capping")
    axes[1].set_ylabel(UNITS[variable])
    axes[1].legend(fontsize=9, loc="upper left")

    fig.suptitle(
        f"Boxplot {variable}: dữ liệu thô và dữ liệu làm sạch",
        fontsize=16, fontweight="bold"
    )
    """fig.text(
        0.5, -0.01,
        "Hai panel dùng thang Y riêng để nhìn rõ hình hộp; capping giữ nguyên số dòng và chặn giá trị tại hàng rào IQR.",
        ha="center", fontsize=9, color="#4B5563"
    )"""
    fig.tight_layout()
    filename = OUTPUT_DIR / f"boxplot_{variable}_raw_vs_capped.png"
    fig.savefig(filename, facecolor="white")
    plt.close(fig)


def main() -> None:
    source = pd.read_csv(INPUT_FILE, sep=";")
    processed = pd.DataFrame(index=source.index)
    statistics_rows = []
    processing_rows = []

    for variable in VARIABLES:
        raw = pd.to_numeric(source[variable], errors="coerce")
        raw_stats = calculate_statistics(raw, "Raw", variable)

        # Median robust hơn mean khi balance/duration lệch phải mạnh.
        fill_value = raw.median()
        imputed = raw.fillna(fill_value)
        lower_bound = raw_stats["Lower_bound"]
        upper_bound = raw_stats["Upper_bound"]
        clean = imputed.clip(lower=lower_bound, upper=upper_bound)

        clean_stats = calculate_statistics(clean, "Clean_capped", variable)
        statistics_rows.extend([raw_stats, clean_stats])

        changed = clean.ne(imputed)
        processing_rows.append({
            "Variable": variable,
            "Fill_method": "Median",
            "Fill_value": fill_value,
            "Missing_filled": int(raw.isna().sum()),
            "Lower_bound_used": lower_bound,
            "Upper_bound_used": upper_bound,
            "Capped_low": int((imputed < lower_bound).sum()),
            "Capped_high": int((imputed > upper_bound).sum()),
            "Total_capped": int(changed.sum()),
        })

        processed[f"{variable}_raw"] = raw
        processed[f"{variable}_imputed"] = imputed
        processed[f"{variable}_capped"] = clean
        processed[f"{variable}_minmax"] = minmax_scale(clean)
        processed[f"{variable}_zscore"] = zscore_scale(clean)

        draw_boxplot(
            variable=variable,
            raw=raw,
            clean=clean,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            raw_outliers=raw_stats["Outlier_count"],
            clean_outliers=clean_stats["Outlier_count"],
        )

    statistics = pd.DataFrame(statistics_rows)
    processing = pd.DataFrame(processing_rows)

    statistics.to_csv(
        OUTPUT_DIR / "statistics_raw_vs_clean.csv", index=False, encoding="utf-8-sig"
    )
    processing.to_csv(
        OUTPUT_DIR / "processing_summary.csv", index=False, encoding="utf-8-sig"
    )
    processed.to_csv(
        OUTPUT_DIR / "age_balance_duration_processed.csv", index=False, encoding="utf-8-sig"
    )

    # Kiểm tra chất lượng scaling.
    checks = []
    for variable in VARIABLES:
        mm = processed[f"{variable}_minmax"]
        z = processed[f"{variable}_zscore"]
        checks.append({
            "Variable": variable,
            "MinMax_min": mm.min(),
            "MinMax_max": mm.max(),
            "Zscore_mean": z.mean(),
            "Zscore_std_ddof0": z.std(ddof=0),
        })
    pd.DataFrame(checks).to_csv(
        OUTPUT_DIR / "scaling_quality_check.csv", index=False, encoding="utf-8-sig"
    )

    print(f"Created results in: {OUTPUT_DIR.resolve()}")
    print(statistics.to_string(index=False))


if __name__ == "__main__":
    main()
