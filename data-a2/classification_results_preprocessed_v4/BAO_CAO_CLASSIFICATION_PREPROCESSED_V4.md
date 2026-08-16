# Classification – Bank Marketing Preprocessed Template v4

## 1. Thiết lập dữ liệu

- Nguồn duy nhất: `bank_full_preprocess_template_v4.xlsx`.
- Sheet `Data_goc`: 45,211 quan sát, 17 biến.
- Sheet `Preprocess`: cung cấp `age_capped`, `balance_capped`, `duration_capped`.
- Ba cột gốc trong `Data_goc` đã được đối chiếu và khớp hoàn toàn với `*_goc`.
- Mô hình chỉ dùng một phiên bản sạch `*_capped` cho mỗi biến, không dùng đồng thời các bản dẫn xuất.
- Các cột `*_minmax` và `*_zscore` có sẵn không được chọn; StandardScaler được fit lại chỉ trên Train.
- Loại `duration` khi dự đoán `y`; loại `y` khi dự đoán `housing` và `loan` để hạn chế rò rỉ dữ liệu.

## 2. Quy trình

1. Chọn lần lượt `y`, `housing`, `loan` làm target.
2. Chia 80% Train và 20% Test, `random_state=42`, có `stratify`.
3. Biến số: Median Imputation + StandardScaler trong Pipeline.
4. Biến định tính: Mode Imputation + One-Hot Encoding.
5. So sánh Logistic Regression và Random Forest.
6. Dùng class weight để giảm ảnh hưởng mất cân bằng.
7. Đánh giá Accuracy, Precision/Recall/F1 lớp Yes, F1 Macro và Balanced Accuracy.

## 3. Phân bố lớp

- `y`: No = 39,922 (88.30%), Yes = 5,289 (11.70%).
- `housing`: No = 20,081 (44.42%), Yes = 25,130 (55.58%).
- `loan`: No = 37,967 (83.98%), Yes = 7,244 (16.02%).

`y` và `loan` mất cân bằng mạnh; không được kết luận chỉ dựa trên Accuracy.

## 4. Kết quả

```text
 Target               Model Excluded_features  Train_size  Test_size  Accuracy  Balanced_accuracy  Precision_yes  Recall_yes  F1_yes  Precision_macro  Recall_macro  F1_macro
      y Logistic Regression          duration       36168       9043    0.7567             0.7081         0.2721      0.6446  0.3827           0.6073        0.7081    0.6156
      y       Random Forest          duration       36168       9043    0.8860             0.6739         0.5166      0.3970  0.4490           0.7195        0.6739    0.6927
housing Logistic Regression                 y       36168       9043    0.7417             0.7458         0.8031      0.7091  0.7532           0.7428        0.7458    0.7411
housing       Random Forest                 y       36168       9043    0.7914             0.7932         0.8360      0.7772  0.8055           0.7899        0.7932    0.7903
   loan Logistic Regression                 y       36168       9043    0.6271             0.6322         0.2454      0.6398  0.3548           0.5732        0.6322    0.5463
   loan       Random Forest                 y       36168       9043    0.8265             0.5798         0.4198      0.2167  0.2858           0.6415        0.5798    0.5935
```

## 5. Kết luận

- Accuracy cao nhất: **Random Forest**, target **y**, Accuracy = **0.8860**.
- F1 lớp Yes cao nhất: **Random Forest**, target **housing**, F1_yes = **0.8055**.
- F1 Macro cao nhất: **Random Forest**, target **housing**, F1_macro = **0.7903**.
- Target tốt nhất được xác định ưu tiên theo F1_yes và F1_macro, không chỉ Accuracy.
- Recall Yes cao nghĩa là bỏ sót ít mẫu Yes; Precision Yes cao nghĩa là dự đoán Yes đáng tin cậy hơn.
