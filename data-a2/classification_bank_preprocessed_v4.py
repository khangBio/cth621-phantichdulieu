"""
CLASSIFICATION TRÊN BANK MARKETING ĐÃ TIỀN XỬ LÝ – TEMPLATE V4
======================================================================

Nguồn duy nhất: bank/bank_full_preprocess_template_v4.xlsx

Các sheet được sử dụng:
    - Data_goc: 45.211 dòng, đầy đủ 17 biến Bank Marketing.
    - Preprocess: age, balance, duration đã điền khuyết/capping/scaling.

Dữ liệu đưa vào mô hình:
    - Thay age, balance, duration trong Data_goc bằng các cột *_capped.
    - Giữ các biến còn lại từ Data_goc.
    - Không đưa đồng thời goc/fill/capped/minmax/zscore vào X vì đó là
      nhiều bản sao dẫn xuất của cùng một biến.

Thực nghiệm:
    - Target: y, housing, loan.
    - Chia 80% Train, 20% Test, có stratify.
    - Mô hình: Logistic Regression và Random Forest.
    - Chỉ số: Accuracy, Precision, Recall, F1 lớp Yes, F1 Macro,
      Balanced Accuracy và Confusion Matrix.

Chạy mặc định (khuyến nghị, hạn chế rò rỉ dữ liệu):
    python classification_bank_preprocessed_v4.py

Chạy theo nghĩa đen "target là một cột, mọi cột còn lại làm X":
    python classification_bank_preprocessed_v4.py --include-leakage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Dùng lại các hàm pipeline/vẽ biểu đồ đã được kiểm thử ở chương trình gốc.
from classification_bank_marketing import (
    TARGETS,
    run_experiments,
    save_class_distribution,
    save_dataset_overview,
    save_metric_comparison,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN_VARIABLES = ["age", "balance", "duration"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classification từ bank_full_preprocess_template_v4.xlsx."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "bank" / "bank_full_preprocess_template_v4.xlsx",
        help="Đường dẫn file Excel v4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "classification_results_preprocessed_v4",
        help="Thư mục lưu kết quả.",
    )
    parser.add_argument(
        "--include-leakage",
        action="store_true",
        help="Chỉ loại target khỏi X và giữ tất cả các cột còn lại.",
    )
    return parser.parse_args()


def load_v4_data(workbook_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc hai sheet, kiểm tra căn chỉnh và tạo dữ liệu mô hình hóa."""
    if not workbook_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file v4: {workbook_path}")

    excel_file = pd.ExcelFile(workbook_path)
    required_sheets = {"Data_goc", "Preprocess"}
    missing_sheets = required_sheets.difference(excel_file.sheet_names)
    if missing_sheets:
        raise ValueError(f"File v4 thiếu sheet: {sorted(missing_sheets)}")

    original = pd.read_excel(workbook_path, sheet_name="Data_goc")
    preprocess = pd.read_excel(workbook_path, sheet_name="Preprocess")

    missing_targets = [target for target in TARGETS if target not in original.columns]
    if missing_targets:
        raise ValueError(f"Sheet Data_goc thiếu target: {missing_targets}")

    required_preprocess_columns = []
    for variable in CLEAN_VARIABLES:
        required_preprocess_columns.extend(
            [
                f"{variable}_goc",
                f"{variable}_capped",
                f"{variable}_minmax",
                f"{variable}_zscore",
            ]
        )
    missing_columns = [
        column
        for column in required_preprocess_columns
        if column not in preprocess.columns
    ]
    if missing_columns:
        raise ValueError(f"Sheet Preprocess thiếu cột: {missing_columns}")

    if len(original) != len(preprocess):
        raise ValueError(
            f"Số dòng không khớp: Data_goc={len(original):,}, "
            f"Preprocess={len(preprocess):,}."
        )

    audit_rows = []
    modeling_data = original.copy()

    for variable in CLEAN_VARIABLES:
        original_values = pd.to_numeric(original[variable], errors="coerce")
        preprocess_original = pd.to_numeric(
            preprocess[f"{variable}_goc"], errors="coerce"
        )
        capped_values = pd.to_numeric(
            preprocess[f"{variable}_capped"], errors="coerce"
        )
        minmax_values = pd.to_numeric(
            preprocess[f"{variable}_minmax"], errors="coerce"
        )
        zscore_values = pd.to_numeric(
            preprocess[f"{variable}_zscore"], errors="coerce"
        )

        aligned = bool(
            np.allclose(
                original_values.to_numpy(dtype=float),
                preprocess_original.to_numpy(dtype=float),
                equal_nan=True,
            )
        )
        if not aligned:
            raise ValueError(
                f"{variable}_goc không khớp Data_goc; dừng để tránh ghép sai dòng."
            )
        if capped_values.isna().any():
            raise ValueError(f"{variable}_capped còn dữ liệu khuyết/không hợp lệ.")

        modeling_data[variable] = capped_values.to_numpy()
        audit_rows.append(
            {
                "Variable": variable,
                "Rows": len(original),
                "Original_matches_preprocess": aligned,
                "Original_min": original_values.min(),
                "Original_max": original_values.max(),
                "Capped_min": capped_values.min(),
                "Capped_max": capped_values.max(),
                "Capped_changed_count": int(
                    (original_values.to_numpy() != capped_values.to_numpy()).sum()
                ),
                "Existing_minmax_min": minmax_values.min(),
                "Existing_minmax_max": minmax_values.max(),
                "Existing_zscore_mean": zscore_values.mean(),
                "Existing_zscore_std_population": zscore_values.std(ddof=0),
                "Selected_model_input": f"{variable}_capped",
                "Scaling_in_pipeline": "StandardScaler fit on Train only",
            }
        )

    return modeling_data, pd.DataFrame(audit_rows)


def save_v4_report(
    data: pd.DataFrame,
    distribution: pd.DataFrame,
    audit: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: Path,
    include_leakage: bool,
) -> None:
    """Tạo báo cáo Markdown mô tả phương pháp và kết quả thực tế."""
    best_accuracy = results.loc[results["Accuracy"].idxmax()]
    best_f1_yes = results.loc[results["F1_yes"].idxmax()]
    best_f1_macro = results.loc[results["F1_macro"].idxmax()]

    class_lines = []
    for target in TARGETS:
        subset = distribution[distribution["Target"] == target]
        no_row = subset[subset["Class"] == "no"].iloc[0]
        yes_row = subset[subset["Class"] == "yes"].iloc[0]
        class_lines.append(
            f"- `{target}`: No = {int(no_row['Count']):,} ({no_row['Percent']:.2f}%), "
            f"Yes = {int(yes_row['Count']):,} ({yes_row['Percent']:.2f}%)."
        )

    exclusion_note = (
        "Chỉ loại target khỏi X theo tùy chọn `--include-leakage`."
        if include_leakage
        else "Loại `duration` khi dự đoán `y`; loại `y` khi dự đoán `housing` và `loan` để hạn chế rò rỉ dữ liệu."
    )

    report = f"""# Classification – Bank Marketing Preprocessed Template v4

## 1. Thiết lập dữ liệu

- Nguồn duy nhất: `bank_full_preprocess_template_v4.xlsx`.
- Sheet `Data_goc`: {len(data):,} quan sát, {data.shape[1]} biến.
- Sheet `Preprocess`: cung cấp `age_capped`, `balance_capped`, `duration_capped`.
- Ba cột gốc trong `Data_goc` đã được đối chiếu và khớp hoàn toàn với `*_goc`.
- Mô hình chỉ dùng một phiên bản sạch `*_capped` cho mỗi biến, không dùng đồng thời các bản dẫn xuất.
- Các cột `*_minmax` và `*_zscore` có sẵn không được chọn; StandardScaler được fit lại chỉ trên Train.
- {exclusion_note}

## 2. Quy trình

1. Chọn lần lượt `y`, `housing`, `loan` làm target.
2. Chia 80% Train và 20% Test, `random_state=42`, có `stratify`.
3. Biến số: Median Imputation + StandardScaler trong Pipeline.
4. Biến định tính: Mode Imputation + One-Hot Encoding.
5. So sánh Logistic Regression và Random Forest.
6. Dùng class weight để giảm ảnh hưởng mất cân bằng.
7. Đánh giá Accuracy, Precision/Recall/F1 lớp Yes, F1 Macro và Balanced Accuracy.

## 3. Phân bố lớp

{chr(10).join(class_lines)}

`y` và `loan` mất cân bằng mạnh; không được kết luận chỉ dựa trên Accuracy.

## 4. Kết quả

```text
{results.to_string(index=False)}
```

## 5. Kết luận

- Accuracy cao nhất: **{best_accuracy['Model']}**, target **{best_accuracy['Target']}**, Accuracy = **{best_accuracy['Accuracy']:.4f}**.
- F1 lớp Yes cao nhất: **{best_f1_yes['Model']}**, target **{best_f1_yes['Target']}**, F1_yes = **{best_f1_yes['F1_yes']:.4f}**.
- F1 Macro cao nhất: **{best_f1_macro['Model']}**, target **{best_f1_macro['Target']}**, F1_macro = **{best_f1_macro['F1_macro']:.4f}**.
- Target tốt nhất được xác định ưu tiên theo F1_yes và F1_macro, không chỉ Accuracy.
- Recall Yes cao nghĩa là bỏ sót ít mẫu Yes; Precision Yes cao nghĩa là dự đoán Yes đáng tin cậy hơn.
"""

    (output_dir / "BAO_CAO_CLASSIFICATION_PREPROCESSED_V4.md").write_text(
        report, encoding="utf-8-sig"
    )


def main() -> None:
    args = parse_arguments()
    input_path = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modeling_data, audit = load_v4_data(input_path)
    audit.to_csv(
        output_dir / "v4_data_audit.csv", index=False, encoding="utf-8-sig"
    )
    save_dataset_overview(modeling_data, output_dir)
    distribution = save_class_distribution(modeling_data, output_dir)

    results = run_experiments(
        data=modeling_data,
        output_dir=output_dir,
        include_leakage=args.include_leakage,
    )
    results.to_csv(
        output_dir / "model_comparison_preprocessed_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_metric_comparison(results, output_dir)
    save_v4_report(
        modeling_data,
        distribution,
        audit,
        results,
        output_dir,
        args.include_leakage,
    )

    print("\n" + "=" * 76)
    print("HOÀN THÀNH CLASSIFICATION TRÊN PREPROCESSED TEMPLATE V4")
    print(f"Kết quả: {output_dir}")
    print("Bảng so sánh: model_comparison_preprocessed_v4.csv")
    print("Báo cáo: BAO_CAO_CLASSIFICATION_PREPROCESSED_V4.md")
    print("=" * 76)


if __name__ == "__main__":
    main()
