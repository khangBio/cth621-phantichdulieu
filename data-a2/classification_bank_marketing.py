"""
THỰC NGHIỆM PHÂN LỚP TRÊN BANK MARKETING (bank-full.csv)
=====================================================================

Chương trình thực hiện 6 thí nghiệm:
    3 target: y, housing, loan
    2 mô hình: Logistic Regression, Random Forest

Quy trình cho từng target:
    1. Chọn X và y.
    2. Chia 80% Train - 20% Test bằng stratified split.
    3. Điền khuyết số bằng Median, định tính bằng Mode.
    4. Standard Scaling biến số và One-Hot Encoding biến định tính.
    5. Huấn luyện hai mô hình có class_weight để giảm ảnh hưởng mất cân bằng.
    6. Tính Accuracy, Precision, Recall, F1, F1-macro,
       Balanced Accuracy và xuất Confusion Matrix.

Mặc định chương trình loại các biến không phù hợp về thời điểm dự đoán:
    - Khi dự đoán y: loại duration vì chỉ biết sau khi cuộc gọi kết thúc.
    - Khi dự đoán housing/loan: loại y vì đây là kết quả chiến dịch.

Muốn làm đúng nghĩa "target là một cột, toàn bộ cột còn lại là X", chạy:
    python classification_bank_marketing.py --include-leakage

Chạy mặc định:
    python classification_bank_marketing.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Bảo đảm thông báo tiếng Việt hiển thị đúng trên Windows/PowerShell.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------
# Tự nhận các thư viện được cài cục bộ trong workspace Codex.
# Hai dòng này không ảnh hưởng nếu người dùng chạy trong môi trường Python
# đã cài pandas, matplotlib, seaborn và scikit-learn theo cách thông thường.
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_ML_DEPS = SCRIPT_DIR / ".ml_deps"
LOCAL_VIZ_DEPS = SCRIPT_DIR / ".viz_deps"

for dependency_dir in reversed([LOCAL_ML_DEPS, LOCAL_VIZ_DEPS]):
    if dependency_dir.exists():
        sys.path.insert(0, str(dependency_dir))

os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))
# Backend không giao diện để chương trình chạy ổn định trong terminal/server.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGETS = ["y", "housing", "loan"]

# Các cột bị loại ngoài chính target để tránh rò rỉ/thông tin sai thời điểm.
SAFE_EXCLUSIONS = {
    "y": ["duration"],
    "housing": ["y"],
    "loan": ["y"],
}


def parse_arguments() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Chạy 3 bài toán phân lớp trên dữ liệu Bank Marketing."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SCRIPT_DIR / "bank" / "bank-full.csv",
        help="Đường dẫn bank-full.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "classification_results",
        help="Thư mục lưu CSV, biểu đồ và báo cáo nhận xét.",
    )
    parser.add_argument(
        "--include-leakage",
        action="store_true",
        help="Chỉ loại target khỏi X và giữ tất cả các cột còn lại.",
    )
    return parser.parse_args()


def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    """Đọc CSV dấu chấm phẩy và kiểm tra các cột cần thiết."""
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {data_path}")

    data = pd.read_csv(data_path, sep=";")
    missing_targets = [target for target in TARGETS if target not in data.columns]

    if missing_targets:
        raise ValueError(f"Thiếu các cột target: {missing_targets}")
    if data.empty:
        raise ValueError("Tệp dữ liệu không có quan sát nào.")

    return data


def save_dataset_overview(data: pd.DataFrame, output_dir: Path) -> None:
    """Lưu kiểm kê kiểu dữ liệu, Null/NaN và giá trị unknown."""
    overview_rows = []
    for column in data.columns:
        unknown_count = int((data[column] == "unknown").sum()) if not pd.api.types.is_numeric_dtype(data[column]) else 0
        overview_rows.append(
            {
                "Column": column,
                "Data_type": str(data[column].dtype),
                "Unique_count": int(data[column].nunique(dropna=True)),
                "Missing_count": int(data[column].isna().sum()),
                "Unknown_count": unknown_count,
            }
        )

    pd.DataFrame(overview_rows).to_csv(
        output_dir / "dataset_overview.csv", index=False, encoding="utf-8-sig"
    )


def save_class_distribution(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Thống kê và vẽ tỷ lệ lớp của ba target."""
    rows = []
    for target in TARGETS:
        counts = data[target].value_counts()
        percentages = data[target].value_counts(normalize=True) * 100
        for class_name, count in counts.items():
            rows.append(
                {
                    "Target": target,
                    "Class": class_name,
                    "Count": int(count),
                    "Percent": float(percentages[class_name]),
                }
            )

    distribution = pd.DataFrame(rows)
    distribution.to_csv(
        output_dir / "class_distribution.csv", index=False, encoding="utf-8-sig"
    )

    plt.figure(figsize=(9, 5.5))
    chart = sns.barplot(
        data=distribution,
        x="Target",
        y="Percent",
        hue="Class",
        hue_order=["no", "yes"],
        palette={"no": "#4C78A8", "yes": "#F58518"},
    )
    for container in chart.containers:
        chart.bar_label(container, fmt="%.1f%%", padding=3)
    plt.title("Phân bố lớp của ba biến mục tiêu", fontweight="bold")
    plt.xlabel("Biến mục tiêu")
    plt.ylabel("Tỷ lệ quan sát (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=220, bbox_inches="tight")
    plt.close()

    return distribution


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Tạo quy trình tiền xử lý chỉ fit trên Train để tránh data leakage."""
    numerical_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def build_models() -> dict[str, object]:
    """Khai báo hai thuật toán so sánh."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: object,
    target: str,
    model_name: str,
    output_dir: Path,
) -> None:
    """Lưu Confusion Matrix gồm số lượng và tỷ lệ theo từng hàng thực tế."""
    matrix = confusion_matrix(y_true, y_pred, labels=["no", "yes"])
    safe_model_name = model_name.lower().replace(" ", "_")

    # Chuẩn hóa theo hàng: mỗi hàng (lớp thực tế) cộng lại bằng 100%.
    # Cách này giúp đọc trực tiếp Recall của lớp No/Yes và không bị lớp đông
    # lấn át màu sắc khi dữ liệu mất cân bằng.
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percent = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    ) * 100
    annotations = np.array(
        [
            [
                f"{matrix[row, column]:,}\n({row_percent[row, column]:.1f}%)"
                for column in range(matrix.shape[1])
            ]
            for row in range(matrix.shape[0])
        ]
    )

    plt.figure(figsize=(7.2, 5.8))
    sns.heatmap(
        row_percent,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar=True,
        cbar_kws={"label": "Tỷ lệ theo hàng thực tế (%)"},
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        xticklabels=["Dự đoán No", "Dự đoán Yes"],
        yticklabels=["Thực tế No", "Thực tế Yes"],
        linewidths=1,
        linecolor="white",
    )
    plt.title(
        f"Confusion Matrix – Target: {target}\n"
        f"{model_name} | Test n = {matrix.sum():,}",
        fontweight="bold",
    )
    plt.xlabel("Nhãn dự đoán\nTrong mỗi ô: Số quan sát (Tỷ lệ theo hàng thực tế)")
    plt.ylabel("Nhãn thực tế")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"confusion_matrix_{target}_{safe_model_name}.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def run_experiments(
    data: pd.DataFrame,
    output_dir: Path,
    include_leakage: bool,
) -> pd.DataFrame:
    """Chạy 3 target x 2 mô hình và trả về bảng kết quả."""
    result_rows = []
    models = build_models()

    for target in TARGETS:
        extra_exclusions = [] if include_leakage else SAFE_EXCLUSIONS[target]
        dropped_columns = [target, *extra_exclusions]

        features = data.drop(columns=dropped_columns)
        labels = data[target]

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )

        preprocessor = build_preprocessor(features)

        print("\n" + "=" * 76)
        print(f"TARGET: {target}")
        print(f"Cột loại khỏi X: {dropped_columns}")
        print(f"Train: {len(x_train):,} | Test: {len(x_test):,}")
        print("=" * 76)

        for model_name, estimator in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessing", clone(preprocessor)),
                    ("model", clone(estimator)),
                ]
            )

            pipeline.fit(x_train, y_train)
            predictions = pipeline.predict(x_test)

            metrics = {
                "Target": target,
                "Model": model_name,
                "Excluded_features": ", ".join(extra_exclusions) or "None",
                "Train_size": len(x_train),
                "Test_size": len(x_test),
                "Accuracy": accuracy_score(y_test, predictions),
                "Balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "Precision_yes": precision_score(
                    y_test, predictions, pos_label="yes", zero_division=0
                ),
                "Recall_yes": recall_score(
                    y_test, predictions, pos_label="yes", zero_division=0
                ),
                "F1_yes": f1_score(
                    y_test, predictions, pos_label="yes", zero_division=0
                ),
                "Precision_macro": precision_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "Recall_macro": recall_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "F1_macro": f1_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
            }
            result_rows.append(metrics)

            safe_model_name = model_name.lower().replace(" ", "_")
            report = pd.DataFrame(
                classification_report(
                    y_test,
                    predictions,
                    output_dict=True,
                    zero_division=0,
                )
            ).transpose()
            report.to_csv(
                output_dir / f"classification_report_{target}_{safe_model_name}.csv",
                encoding="utf-8-sig",
            )

            prediction_table = pd.DataFrame(
                {
                    "Actual": y_test.reset_index(drop=True),
                    "Predicted": pd.Series(predictions),
                }
            )
            prediction_table.to_csv(
                output_dir / f"predictions_{target}_{safe_model_name}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            save_confusion_matrix(
                y_test, predictions, target, model_name, output_dir
            )

            print(
                f"{model_name:21s} | "
                f"Accuracy={metrics['Accuracy']:.4f} | "
                f"Precision_yes={metrics['Precision_yes']:.4f} | "
                f"Recall_yes={metrics['Recall_yes']:.4f} | "
                f"F1_yes={metrics['F1_yes']:.4f} | "
                f"F1_macro={metrics['F1_macro']:.4f}"
            )

    results = pd.DataFrame(result_rows)
    numeric_metrics = [
        "Accuracy",
        "Balanced_accuracy",
        "Precision_yes",
        "Recall_yes",
        "F1_yes",
        "Precision_macro",
        "Recall_macro",
        "F1_macro",
    ]
    results[numeric_metrics] = results[numeric_metrics].round(4)
    return results


def save_metric_comparison(results: pd.DataFrame, output_dir: Path) -> None:
    """Vẽ Accuracy, F1 lớp Yes và F1 Macro để tránh kết luận theo một chỉ số."""
    chart_data = results.melt(
        id_vars=["Target", "Model"],
        value_vars=["Accuracy", "F1_yes", "F1_macro"],
        var_name="Metric",
        value_name="Score",
    )

    chart = sns.catplot(
        data=chart_data,
        x="Target",
        y="Score",
        hue="Model",
        col="Metric",
        kind="bar",
        height=4.8,
        aspect=0.85,
        palette="Set2",
    )
    chart.set_axis_labels("Biến mục tiêu", "Điểm đánh giá")
    chart.set(ylim=(0, 1))
    chart.fig.subplots_adjust(top=0.82)
    chart.fig.suptitle(
        "So sánh hiệu năng mô hình trên ba biến mục tiêu",
        fontsize=15,
        fontweight="bold",
    )
    chart.savefig(output_dir / "model_metric_comparison.png", dpi=220)
    plt.close("all")


def save_summary_report(
    data: pd.DataFrame,
    distribution: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: Path,
    include_leakage: bool,
) -> None:
    """Sinh báo cáo Markdown mô tả cách làm và nhận xét tự động."""
    best_f1_yes = results.loc[results["F1_yes"].idxmax()]
    best_f1_macro = results.loc[results["F1_macro"].idxmax()]
    best_accuracy = results.loc[results["Accuracy"].idxmax()]

    class_lines = []
    for target in TARGETS:
        subset = distribution[distribution["Target"] == target]
        yes_row = subset[subset["Class"] == "yes"].iloc[0]
        no_row = subset[subset["Class"] == "no"].iloc[0]
        class_lines.append(
            f"- `{target}`: No = {int(no_row['Count']):,} ({no_row['Percent']:.2f}%), "
            f"Yes = {int(yes_row['Count']):,} ({yes_row['Percent']:.2f}%)."
        )

    # Dùng bảng monospace để không yêu cầu thêm gói `tabulate`.
    result_markdown = "```text\n" + results.to_string(index=False) + "\n```"
    leakage_note = (
        "Đã giữ toàn bộ cột còn lại theo tùy chọn --include-leakage."
        if include_leakage
        else "Đã loại duration khi dự đoán y và loại y khi dự đoán housing/loan để hạn chế rò rỉ dữ liệu."
    )

    report_text = f"""# Báo cáo thực nghiệm Classification – Bank Marketing

## 1. Dữ liệu và cách thực hiện

- Nguồn dữ liệu: `bank/bank-full.csv`.
- Kích thước: {len(data):,} quan sát, {data.shape[1]} cột.
- Chia dữ liệu: 80% Train và 20% Test, `random_state=42`.
- Sử dụng `stratify` để giữ tỷ lệ lớp trong Train và Test.
- Biến số: điền Median và Standard Z-score.
- Biến định tính: điền Mode và One-Hot Encoding.
- Giá trị `unknown` được giữ như một nhóm phân loại hợp lệ.
- Hai mô hình: Logistic Regression và Random Forest.
- Cả hai mô hình dùng `class_weight` để giảm ảnh hưởng mất cân bằng lớp.
- {leakage_note}

## 2. Phân bố lớp

{chr(10).join(class_lines)}

Target `y` và `loan` mất cân bằng rõ rệt. Vì vậy không thể chỉ dùng Accuracy;
cần xem thêm Recall, F1 của lớp Yes, F1 Macro và Balanced Accuracy.

## 3. Kết quả

{result_markdown}

## 4. Nhận xét chính

- Accuracy cao nhất: **{best_accuracy['Model']}**, target **{best_accuracy['Target']}**, Accuracy = **{best_accuracy['Accuracy']:.4f}**.
- F1 lớp Yes cao nhất: **{best_f1_yes['Model']}**, target **{best_f1_yes['Target']}**, F1_yes = **{best_f1_yes['F1_yes']:.4f}**.
- F1 Macro cao nhất: **{best_f1_macro['Model']}**, target **{best_f1_macro['Target']}**, F1_macro = **{best_f1_macro['F1_macro']:.4f}**.
- Khi đánh giá target mất cân bằng, F1_yes/F1_macro phù hợp hơn Accuracy.
- Recall_yes cao cho biết mô hình bỏ sót ít trường hợp Yes; Precision_yes cao cho biết các dự đoán Yes đáng tin cậy hơn.
- Confusion Matrix cần được đọc theo hàng = nhãn thực tế, cột = nhãn dự đoán.
"""

    (output_dir / "BAO_CAO_CLASSIFICATION.md").write_text(
        report_text, encoding="utf-8-sig"
    )


def main() -> None:
    args = parse_arguments()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.0)

    data = load_and_validate_data(args.data.resolve())
    save_dataset_overview(data, output_dir)
    distribution = save_class_distribution(data, output_dir)

    results = run_experiments(
        data=data,
        output_dir=output_dir,
        include_leakage=args.include_leakage,
    )

    results.to_csv(
        output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )
    save_metric_comparison(results, output_dir)
    save_summary_report(
        data,
        distribution,
        results,
        output_dir,
        args.include_leakage,
    )

    print("\n" + "=" * 76)
    print("HOÀN THÀNH")
    print(f"Kết quả được lưu tại: {output_dir}")
    print("Bảng tổng hợp: model_comparison.csv")
    print("Báo cáo nhận xét: BAO_CAO_CLASSIFICATION.md")
    print("=" * 76)


if __name__ == "__main__":
    main()
