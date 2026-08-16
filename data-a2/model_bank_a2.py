from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# 1. ĐỌC DỮ LIỆU
# =========================================================

DATA_PATH = Path("bank/bank-full.csv")
OUTPUT_DIR = Path("classification_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# bank-full.csv sử dụng dấu chấm phẩy
df = pd.read_csv(DATA_PATH, sep=";")

print("Kích thước dữ liệu:", df.shape)
print("\nSố ô khuyết thiếu:")
print(df.isna().sum())

TARGETS = ["y", "housing", "loan"]

for target in TARGETS:
    print(f"\nPhân bố target {target}:")
    print(df[target].value_counts())
    print(df[target].value_counts(normalize=True).mul(100).round(2))


# =========================================================
# 2. CÁC CỘT LOẠI BỔ ĐỂ TRÁNH DATA LEAKAGE
# =========================================================

EXCLUDE_COLUMNS = {
    "y": ["duration"],
    "housing": ["y"],
    "loan": ["y"]
}

# Nếu cần thực hiện đúng nghĩa "tất cả cột còn lại làm X",
# thay bằng:
#
# EXCLUDE_COLUMNS = {
#     "y": [],
#     "housing": [],
#     "loan": []
# }


# =========================================================
# 3. KHAI BÁO MÔ HÌNH
# =========================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}


# =========================================================
# 4. THỰC NGHIỆM
# =========================================================

all_results = []

for target in TARGETS:

    print("\n" + "=" * 70)
    print("TARGET:", target)
    print("=" * 70)

    columns_to_drop = [target] + EXCLUDE_COLUMNS[target]

    X = df.drop(columns=columns_to_drop)
    y = df[target]

    # Chia 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Train:", X_train.shape)
    print("Test :", X_test.shape)

    numerical_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

    # Biến định lượng: điền median và Standard Scaling
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Biến định tính: điền mode và One-Hot Encoding
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    for model_name, model in models.items():

        pipeline = Pipeline([
            ("preprocessing", clone(preprocessor)),
            ("model", clone(model))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision_yes = precision_score(
            y_test, y_pred,
            pos_label="yes",
            zero_division=0
        )

        recall_yes = recall_score(
            y_test, y_pred,
            pos_label="yes",
            zero_division=0
        )

        f1_yes = f1_score(
            y_test, y_pred,
            pos_label="yes",
            zero_division=0
        )

        f1_macro = f1_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )

        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

        all_results.append({
            "Target": target,
            "Model": model_name,
            "Train_size": len(X_train),
            "Test_size": len(X_test),
            "Accuracy": accuracy,
            "Balanced_accuracy": balanced_accuracy,
            "Precision_yes": precision_yes,
            "Recall_yes": recall_yes,
            "F1_yes": f1_yes,
            "F1_macro": f1_macro
        })

        print(f"\n{model_name}")
        print(classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0
        ))

        # Lưu Classification Report
        report = pd.DataFrame(
            classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0
            )
        ).transpose()

        safe_model_name = model_name.lower().replace(" ", "_")

        report.to_csv(
            OUTPUT_DIR /
            f"classification_report_{target}_{safe_model_name}.csv",
            encoding="utf-8-sig"
        )

        # Ma trận nhầm lẫn
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=["no", "yes"]
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt=",d",
            cmap="Blues",
            xticklabels=["Dự đoán No", "Dự đoán Yes"],
            yticklabels=["Thực tế No", "Thực tế Yes"]
        )

        plt.title(
            f"Confusion Matrix – {target}\n{model_name}",
            fontweight="bold"
        )
        plt.xlabel("Nhãn dự đoán")
        plt.ylabel("Nhãn thực tế")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR /
            f"confusion_matrix_{target}_{safe_model_name}.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()


# =========================================================
# 5. TỔNG HỢP KẾT QUẢ
# =========================================================

results = pd.DataFrame(all_results)

metric_columns = [
    "Accuracy",
    "Balanced_accuracy",
    "Precision_yes",
    "Recall_yes",
    "F1_yes",
    "F1_macro"
]

results[metric_columns] = results[metric_columns].round(4)

results.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nBẢNG SO SÁNH:")
print(results.to_string(index=False))


# =========================================================
# 6. BIỂU ĐỒ SO SÁNH F1-SCORE
# =========================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results,
    x="Target",
    y="F1_yes",
    hue="Model"
)

plt.title("So sánh F1-Score của lớp Yes", fontweight="bold")
plt.xlabel("Biến mục tiêu")
plt.ylabel("F1-Score lớp Yes")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "comparison_f1_score.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# Mô hình tốt nhất theo F1 lớp Yes
best_model = results.loc[results["F1_yes"].idxmax()]

print("\nMÔ HÌNH TỐT NHẤT THEO F1 LỚP YES:")
print(best_model)