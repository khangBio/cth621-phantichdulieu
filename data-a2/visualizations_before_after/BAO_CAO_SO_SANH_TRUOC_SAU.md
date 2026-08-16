# So sánh dữ liệu trước và sau tiền xử lý

**Nguồn:** `bank/bank-full.csv` — 45,211 dòng.  
**Phạm vi:** age, balance, duration, job và y.  
**Nguyên tắc:** dữ liệu gốc không bị ghi đè; bản sau xử lý nằm ở `bank_5vars_processed.csv`.

## Quy trình đã áp dụng

1. Kiểm kê Null/NaN thật: cả 5 biến đều có **0 ô Null/NaN**.
2. Nhận diện khuyết thiếu ngữ nghĩa: `job='unknown'` có **288 ô**; chuyển thành NaN và điền bằng mode `blue-collar`.
3. Nếu biến số có khuyết, dùng median: age=39, balance=448, duration=180. Trong dữ liệu này không có ô số nào cần điền.
4. Xử lý outlier bằng IQR capping: chặn tại [Q1−1.5×IQR, Q3+1.5×IQR], không xóa dòng.
5. Tạo thêm hai bộ đặc trưng: Z-score (mean≈0, std≈1) và Min–Max ([0,1]).
6. `y` được giữ nguyên; số dòng mục tiêu bị loại do thiếu = 0.

## Thay đổi thống kê

| Biến | Mean trước → sau | Median trước → sau | Std trước → sau | Skewness trước → sau | Outlier IQR trước → sau |
|---|---:|---:|---:|---:|---:|
| age | 40.94 → 40.87 | 39.00 → 39.00 | 10.62 → 10.39 (-2.1%) | 0.68 → 0.53 | 487 → 0 |
| balance | 1362.27 → 933.71 | 448.00 → 448.00 | 3044.77 → 1176.77 (-61.4%) | 8.36 → 1.10 | 4,729 → 0 |
| duration | 258.16 → 234.94 | 180.00 → 180.00 | 257.53 → 176.75 (-31.4%) | 3.14 → 1.04 | 3,235 → 0 |

Lưu ý: sau capping, nhiều điểm nằm đúng tại hàng rào cũ. Bảng “outlier sau” được tính lại bằng hàng rào IQR mới nên có thể vẫn xuất hiện nếu phân phối vốn lệch mạnh; capping đã loại bỏ ảnh hưởng của các cực trị vượt hàng rào ban đầu chứ không ép dữ liệu thành phân phối chuẩn.

## Giải thích từng biểu đồ

### 01 — Kiểm kê khuyết thiếu
- **Trục X:** 5 biến. **Trục Y:** số ô thiếu/unknown.
- **Nhận xét:** Null/NaN thật đều bằng 0. Chỉ `job` có 288 giá trị `unknown`; sau quy đổi và điền mode, số thiếu bằng 0. Không tạo dữ liệu thiếu giả.

### 02 — Histogram trước–sau
- **Trục X:** giá trị age, balance, duration. **Trục Y:** số quan sát trong bin.
- **Nhận xét:** capping rút ngắn đuôi balance và duration nên phần trung tâm dễ đọc hơn. Các cột cao ở biên sau xử lý là những outlier được chặn về hàng rào, không phải quan sát mới.

### 03 — Boxplot trước–sau
- **Trục X:** giá trị biến; hộp là Q1–Q3, đường giữa là median, điểm ngoài râu là outlier. **Trục Y:** vị trí hộp.
- **Nhận xét:** trước xử lý có age=487, balance=4,729, duration=3,235 outlier. Giá trị cực đại được ghi trực tiếp trên hình; danh sách dòng cụ thể nằm trong `VI_DU_OUTLIER_CU_THE.csv`.

### 04 — Line chart theo thứ hạng
- **Trục X:** thứ hạng sau sắp tăng. **Trục Y:** giá trị biến.
- **Nhận xét:** đường trước xử lý có đoạn tăng gãy mạnh ở đuôi; sau xử lý tạo plateau tại hàng rào IQR. Đây là hình ảnh rõ nhất về việc giới hạn ảnh hưởng của cực trị.

### 05 — Scatter các cặp biến
- **Trục X/Y:** lần lượt age–balance, age–duration, balance–duration; màu là y.
- **Nhận xét:** sau xử lý, các điểm không còn bị vài cực trị kéo giãn trục nên cấu trúc trung tâm rõ hơn. Quan hệ tuyến tính giữa ba biến vẫn yếu; việc capping không tạo tương quan giả rõ rệt.

### 06 — Heatmap tương quan
- **Hai trục:** age, balance, duration và y mã hóa 0/1. **Màu/số:** Pearson.
- **Nhận xét:** hệ số thay đổi vì cực trị có leverage lớn. Quan hệ duration–y vẫn nổi bật, chứng tỏ tín hiệu chính không mất sau capping.

### 07 — Bar chart job
- **Trục X:** số khách hàng. **Trục Y:** job.
- **Nhận xét:** 288 `unknown` được chuyển vào mode `blue-collar`, do đó thanh unknown biến mất và thanh `blue-collar` tăng tương ứng. Đây là thay đổi có chủ đích nhưng có thể làm nhóm mode trội hơn.

### 08 — Doughnut job
- **Lát:** tỷ trọng từng job; không có trục tọa độ.
- **Nhận xét:** cơ cấu tổng thể hầu như giữ nguyên vì unknown chỉ chiếm 0.64%; khác biệt tập trung ở nhóm mode.

### 09 — Biến mục tiêu y
- **Trục X:** no/yes. **Trục Y:** số khách hàng.
- **Nhận xét:** số lượng và tỷ trọng y giống hệt trước–sau. Đây là kiểm tra quan trọng để bảo đảm tiền xử lý predictors không làm méo nhãn.

### 10 — Tỷ lệ yes theo job
- **Trục X:** job. **Trục Y:** tỷ lệ y=yes.
- **Nhận xét:** các job có tên xác định giữ nguyên tỷ lệ; `blue-collar` thay đổi nhẹ vì nhận thêm 288 dòng unknown. Đường của unknown sau xử lý bị khuyết vì nhóm này không còn tồn tại.

### 11 — Z-score
- **Trục X:** số độ lệch chuẩn so với mean. **Trục Y:** số quan sát.
- **Nhận xét:** cả ba biến có mean xấp xỉ 0 và std xấp xỉ 1, giúp các thuật toán nhạy thang đo như KNN, SVM, PCA và hồi quy có regularization.

### 12 — Min–Max
- **Trục X:** tên biến. **Trục Y:** giá trị trong [0,1].
- **Nhận xét:** ba biến được đưa về cùng miền. Min–Max không làm phân phối trở thành chuẩn; nó chỉ đổi thang đo và vẫn phản ánh độ lệch tương đối.

### 13 — Dashboard thống kê
- **Trục X:** biến. **Trục Y:** lần lượt std, skewness và số outlier; màu là trạng thái.
- **Nhận xét:** std và skewness giảm rõ nhất ở balance/duration, định lượng giá trị của capping. Không nên coi std giảm là tốt một cách tự động; nó tốt ở đây vì giảm ảnh hưởng của cực trị bất thường trong khi giữ đủ dòng.

### 14 — ECDF
- **Trục X:** giá trị biến. **Trục Y:** tỷ lệ tích lũy ≤ X.
- **Nhận xét:** phần lớn hai đường chồng nhau ở trung tâm, còn khác biệt tập trung tại hai đuôi. Điều này cho thấy capping có tính cục bộ, không biến đổi hàng loạt các quan sát bình thường.

### 15 — Duration theo y
- **Trục X:** no/yes. **Trục Y:** duration.
- **Nhận xét:** khoảng cách trung vị và phân phối giữa hai lớp vẫn còn sau xử lý. Duration vẫn là tín hiệu mạnh nhưng chỉ biết sau khi cuộc gọi kết thúc, nên có nguy cơ leakage nếu dự báo trước cuộc gọi.

## Đánh giá chất lượng sau xử lý

- **Đạt:** không còn ô thiếu trong 5 biến; giữ nguyên số dòng và nhãn y; thang đo chuẩn hóa sẵn; ảnh hưởng cực trị giảm mạnh.
- **Đạt có điều kiện:** histogram “mượt/dễ đọc” hơn nhưng chưa và không cần trở thành phân phối chuẩn. Capping tạo khối lượng tại biên — một đánh đổi minh bạch.
- **Cần thận trọng:** điền mode cho `job=unknown` có thể gây thiên lệch về nhóm phổ biến. Với mô hình thực tế, giữ `unknown` như một category riêng cũng là phương án hợp lệ và nên được kiểm chứng chéo.
- **Tránh leakage:** các tham số median/mode, hàng rào IQR, mean/std và min/max phải được fit **chỉ trên tập train**, rồi áp dụng sang validation/test.
