"""So sánh kết quả phân lớp trên tập Train và Test của dữ liệu v4."""

from pathlib import Path

from classification_bank_marketing import (
    RANDOM_STATE,
    SAFE_EXCLUSIONS,
    TARGETS,
    TEST_SIZE,
    build_models,
    build_preprocessor,
)
from classification_bank_preprocessed_v4 import load_v4_data

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "bank" / "bank_full_preprocess_template_v4.xlsx"
OUTPUT_DIR = ROOT / "classification_results_preprocessed_v4"


def calculate_metrics(y_true, y_pred):
    """Tính các chỉ số tổng quát và chỉ số tập trung vào lớp Yes."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_Yes": precision_score(
            y_true, y_pred, pos_label="yes", zero_division=0
        ),
        "Recall_Yes": recall_score(y_true, y_pred, pos_label="yes", zero_division=0),
        "F1_Yes": f1_score(y_true, y_pred, pos_label="yes", zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def collect_train_test_results(data):
    rows = []

    for target in TARGETS:
        excluded = [target, *SAFE_EXCLUSIONS.get(target, [])]
        features = data.drop(columns=excluded, errors="ignore")
        labels = data[target].astype(str).str.lower()

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )

        preprocessor = build_preprocessor(features)
        for model_name, estimator in build_models().items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", clone(estimator)),
                ]
            )
            pipeline.fit(x_train, y_train)

            for split_name, x_split, y_split in (
                ("Train", x_train, y_train),
                ("Test", x_test, y_test),
            ):
                predictions = pipeline.predict(x_split)
                row = {
                    "Target": target,
                    "Model": model_name,
                    "Split": split_name,
                    "N_observations": len(y_split),
                }
                row.update(calculate_metrics(y_split, predictions))
                rows.append(row)

    return pd.DataFrame(rows)


def plot_train_test_metrics(results, output_path):
    """Vẽ bốn ô biểu đồ so sánh Train và Test theo target/mô hình."""
    sns.set_theme(style="whitegrid", context="notebook")

    model_labels = {
        "Logistic Regression": "Logistic",
        "Random Forest": "Random Forest",
    }
    target_labels = {"y": "Target y", "housing": "Target housing", "loan": "Target loan"}

    plot_data = results.copy()
    plot_data["Experiment"] = plot_data.apply(
        lambda row: f"{target_labels[row['Target']]}\n{model_labels[row['Model']]}",
        axis=1,
    )
    order = [
        f"{target_labels[target]}\n{model_labels[model]}"
        for target in TARGETS
        for model in build_models().keys()
    ]

    metric_specs = [
        ("Accuracy", "Accuracy – độ chính xác tổng thể"),
        ("Precision_Yes", "Precision lớp Yes"),
        ("Recall_Yes", "Recall lớp Yes"),
        ("F1_Yes", "F1-score lớp Yes"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(19, 12), sharey=True)
    palette = {"Train": "#2878B5", "Test": "#F39C35"}

    for ax, (metric, title) in zip(axes.flat, metric_specs):
        sns.barplot(
            data=plot_data,
            x="Experiment",
            y=metric,
            hue="Split",
            order=order,
            hue_order=["Train", "Test"],
            palette=palette,
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Biến mục tiêu và mô hình")
        ax.set_ylabel("Giá trị chỉ số (0–1)")
        ax.set_ylim(0, 1.12)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8, rotation=90)
        ax.legend(title="Tập dữ liệu", loc="lower right", frameon=True)

    fig.suptitle(
        "So sánh kết quả mô hình phân lớp trên tập Train và Test\n"
        "Dữ liệu Bank Marketing đã tiền xử lý v4 – chia có phân tầng 80%/20%",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.005,
        "Khoảng cách Train–Test càng lớn cho thấy nguy cơ mô hình học quá sát dữ liệu huấn luyện (overfitting).",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_individual_train_test_metrics(results, output_dir):
    """Xuất riêng một ảnh cho từng chỉ số đánh giá."""
    sns.set_theme(style="whitegrid", context="notebook")

    model_labels = {
        "Logistic Regression": "Logistic",
        "Random Forest": "Random Forest",
    }
    target_labels = {"y": "Target y", "housing": "Target housing", "loan": "Target loan"}

    plot_data = results.copy()
    plot_data["Experiment"] = plot_data.apply(
        lambda row: f"{target_labels[row['Target']]}\n{model_labels[row['Model']]}",
        axis=1,
    )
    order = [
        f"{target_labels[target]}\n{model_labels[model]}"
        for target in TARGETS
        for model in build_models().keys()
    ]
    metric_specs = [
        ("Accuracy", "01_accuracy_train_test_v4.png", "Accuracy – độ chính xác tổng thể"),
        ("Precision_Yes", "02_precision_yes_train_test_v4.png", "Precision của lớp Yes"),
        ("Recall_Yes", "03_recall_yes_train_test_v4.png", "Recall của lớp Yes"),
        ("F1_Yes", "04_f1_yes_train_test_v4.png", "F1-score của lớp Yes"),
    ]
    palette = {"Train": "#2878B5", "Test": "#F39C35"}

    for metric, filename, title in metric_specs:
        fig, ax = plt.subplots(figsize=(15, 7.5))
        sns.barplot(
            data=plot_data,
            x="Experiment",
            y=metric,
            hue="Split",
            order=order,
            hue_order=["Train", "Test"],
            palette=palette,
            ax=ax,
        )
        ax.set_title(
            f"So sánh {title} trên tập Train và Test\n"
            "Bank Marketing đã tiền xử lý v4 – chia có phân tầng 80%/20%",
            fontsize=17,
            fontweight="bold",
            pad=16,
        )
        ax.set_xlabel("Biến mục tiêu và mô hình", fontsize=12)
        ax.set_ylabel("Giá trị chỉ số (0–1)", fontsize=12)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10)
        ax.legend(title="Tập dữ liệu", loc="upper right", frameon=True)
        fig.text(
            0.5,
            0.01,
            "Khoảng cách Train–Test lớn là dấu hiệu mô hình có nguy cơ overfitting.",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def build_gap_table(results):
    metric_columns = ["Accuracy", "Precision_Yes", "Recall_Yes", "F1_Yes", "F1_Macro"]
    wide = results.pivot(index=["Target", "Model"], columns="Split", values=metric_columns)
    gap_rows = []
    for target, model in wide.index:
        row = {"Target": target, "Model": model}
        for metric in metric_columns:
            train_value = wide.loc[(target, model), (metric, "Train")]
            test_value = wide.loc[(target, model), (metric, "Test")]
            row[f"{metric}_Train"] = train_value
            row[f"{metric}_Test"] = test_value
            row[f"{metric}_Gap"] = train_value - test_value
        gap_rows.append(row)
    return pd.DataFrame(gap_rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, _ = load_v4_data(INPUT_FILE)

    results = collect_train_test_results(data)
    results = results.round(4)
    results.to_csv(OUTPUT_DIR / "train_test_metrics_v4.csv", index=False, encoding="utf-8-sig")

    gaps = build_gap_table(results).round(4)
    gaps.to_csv(OUTPUT_DIR / "train_test_metric_gaps_v4.csv", index=False, encoding="utf-8-sig")

    plot_train_test_metrics(
        results,
        OUTPUT_DIR / "train_test_metrics_comparison_v4.png",
    )
    plot_individual_train_test_metrics(results, OUTPUT_DIR)

    print("Đã tạo biểu đồ và bảng so sánh Train/Test:")
    print(OUTPUT_DIR / "train_test_metrics_comparison_v4.png")
    print(OUTPUT_DIR / "train_test_metrics_v4.csv")
    print(OUTPUT_DIR / "train_test_metric_gaps_v4.csv")
    print(OUTPUT_DIR / "01_accuracy_train_test_v4.png")
    print(OUTPUT_DIR / "02_precision_yes_train_test_v4.png")
    print(OUTPUT_DIR / "03_recall_yes_train_test_v4.png")
    print(OUTPUT_DIR / "04_f1_yes_train_test_v4.png")
    print("\nKhoảng cách Train - Test:")
    print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
