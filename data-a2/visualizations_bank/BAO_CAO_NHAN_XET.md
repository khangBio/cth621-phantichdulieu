# Báo cáo trực quan hóa: age, balance, duration, job và y

**Nguồn:** `bank/bank-full.csv`  
**Quy mô:** 45,211 dòng, 5 biến phân tích; không có giá trị thiếu trong 5 biến.  
**Mục tiêu:** `y=yes` nghĩa là khách hàng đăng ký tiền gửi kỳ hạn.

Tên file PNG bắt đầu bằng số thứ tự tương ứng với từng mục giải thích dưới đây.

## Thống kê nền để đối chiếu

- **age**: Q1=33, trung vị=39, Q3=48; hàng rào IQR [10, 70], 487 điểm ngoài hàng rào (1.1%), độ lệch=0.68.
- **balance**: Q1=72, trung vị=448, Q3=1,428; hàng rào IQR [-1,962, 3,462], 4,729 điểm ngoài hàng rào (10.5%), độ lệch=8.36.
- **duration**: Q1=103, trung vị=180, Q3=319; hàng rào IQR [-221, 643], 3,235 điểm ngoài hàng rào (7.2%), độ lệch=3.14.

- `y=yes`: 5,289/45,211 = **11.7%**; dữ liệu mất cân bằng mạnh về lớp `no` (88.3%).
- Trung vị theo `y`: age no/yes = 39/38; balance = 417/733; duration = 164/426.
- Pearson với `y`: age=0.025, balance=0.053, duration=0.395. Duration nổi bật nhất, nhưng lưu ý đây là thông tin chỉ biết sau khi cuộc gọi kết thúc.

## Giải thích từng biểu đồ

### 01 — Histogram và KDE
- **Trục X:** giá trị age, balance, duration. **Trục Y:** số quan sát trong mỗi khoảng; đường KDE biểu diễn mật độ làm trơn.
- **Nhận xét:** age tập trung quanh 30–50 và lệch phải nhẹ. Balance và duration lệch phải rất mạnh: trung bình cao hơn trung vị rõ rệt (balance 1,362 so với 448; duration 258 so với 180). Co biên 1%–99% chỉ phục vụ khả năng đọc, không xóa dữ liệu khi tính thống kê.

### 02 — Histogram toàn miền với trục Y log
- **Trục X:** giá trị thật trên toàn miền. **Trục Y:** số quan sát theo thang log.
- **Nhận xét:** nhìn thấy các đuôi hiếm mà histogram thường che khuất: balance từ -8,019 đến 102,127; duration tới 4,918 giây; age tới 95. Đây là bằng chứng hình học cho độ lệch và outlier trong thống kê.

### 03 — Boxplot toàn miền
- **Trục X:** giá trị biến; hộp từ Q1 đến Q3, đường giữa là trung vị, điểm ngoài râu là dị biệt theo quy tắc 1.5×IQR. **Trục Y:** chỉ là vị trí biến.
- **Nhận xét:** balance có nhiều điểm cực trị nhất về biên độ; duration có đuôi dài; age có nhóm tuổi cao hiếm. Boxplot bị nén ở balance chính là dấu hiệu cực trị rất lớn, không phải lỗi vẽ.

### 04 — Boxplot theo y
- **Trục X:** lớp `no/yes`. **Trục Y:** giá trị biến.
- **Nhận xét:** chênh lệch nổi bật nhất là duration: trung vị `yes`=426s so với `no`=164s. Balance của nhóm yes cũng cao hơn, còn age chồng lấn mạnh nên khó tách lớp một mình.

### 05 — Scatter age–balance
- **Trục X:** tuổi. **Trục Y:** balance; màu là y.
- **Nhận xét:** đám mây phân tán rộng, không có quan hệ tuyến tính mạnh (Pearson=0.098); hai lớp y chồng lấn đáng kể.

### 06 — Scatter age–duration
- **Trục X:** tuổi. **Trục Y:** thời lượng; màu là y.
- **Nhận xét:** không thấy tuổi quyết định thời lượng; các điểm yes xuất hiện dày hơn ở vùng thời lượng cao. Quan hệ của duration với y mạnh hơn age.

### 07 — Scatter balance–duration
- **Trục X:** balance. **Trục Y:** duration; màu là y.
- **Nhận xét:** không có đường xu hướng tuyến tính rõ giữa hai biến (Pearson=0.022); y=yes tập trung tương đối nhiều ở nửa trên của trục duration, bất kể balance.

### 08 — Heatmap tương quan
- **Hai trục:** cùng liệt kê age, balance, duration và y mã hóa 0/1; màu/số là hệ số tương quan.
- **Nhận xét:** Pearson đo tuyến tính, Spearman đo đơn điệu theo thứ hạng. Duration–y cao nhất (0.395/0.342); các cặp biến đầu vào có tương quan thấp, nên không có dấu hiệu đa cộng tuyến mạnh trong 3 biến này.

### 09 — Bar chart số lượng job
- **Trục X:** số khách hàng. **Trục Y:** nhóm nghề nghiệp, sắp tăng dần.
- **Nhận xét:** blue-collar (9,732) và management (9,458) chiếm nhiều nhất; unknown chỉ 288. So sánh tỷ lệ yes của nhóm nhỏ cần thận trọng vì bất định cao hơn.

### 10 — Doughnut tỷ trọng job
- **Góc/diện tích lát:** tỷ trọng số quan sát; chú giải ghi job và phần trăm. Biểu đồ tròn không có trục tọa độ.
- **Nhận xét:** cơ cấu tập trung vào blue-collar, management và technician; nhiều nhóm nhỏ khiến bar chart 09 chính xác hơn để so hạng, doughnut hữu ích cho cái nhìn cơ cấu.

### 11 — Bar và pie cho y
- **Bar:** X là lớp y, Y là số khách hàng. **Pie:** diện tích lát là tỷ trọng lớp.
- **Nhận xét:** `no`=39,922 (88.3%), `yes`=5,289 (11.7%). Khi xây mô hình, accuracy đơn thuần dễ gây hiểu lầm; nên xem recall, precision, F1/PR-AUC.

### 12 — 100% stacked bar job × y
- **Trục X:** tỷ trọng trong từng job (mỗi thanh = 100%). **Trục Y:** job; màu phân rã no/yes.
- **Nhận xét:** tỷ trọng yes khác đáng kể theo nghề; student và retired có phần màu yes lớn hơn, blue-collar nhỏ hơn. Biểu đồ chuẩn hóa giúp không bị số lượng nhóm chi phối.

### 13 — Tỷ lệ yes theo job và khoảng tin cậy
- **Trục X:** tỷ lệ y=yes. **Trục Y:** job. Điểm là tỷ lệ mẫu, thanh ngang là Wilson 95%.
- **Nhận xét:** cao nhất là student (28.7%), tiếp theo retired (22.8%); thấp nhất là blue-collar (7.3%). Khoảng rộng hơn ở nhóm nhỏ thể hiện đúng mức bất định.

### 14 — Violin plot theo y
- **Trục X:** no/yes. **Trục Y:** age hoặc biến đổi log có dấu/log1p để nén đuôi; bề rộng violin là mật độ, vạch trong là các tứ phân vị.
- **Nhận xét:** hai lớp chồng lấn mạnh ở age và balance; duration của yes dịch lên rõ rệt. Biến đổi log chỉ dùng để nhìn hình dạng, không thay đổi thứ tự quan sát.

### 15 — ECDF theo y
- **Trục X:** giá trị biến. **Trục Y:** tỷ lệ quan sát có giá trị ≤ X.
- **Nhận xét:** khoảng cách dọc giữa hai đường thể hiện khác biệt phân phối. Duration tách hai lớp rõ nhất; age và balance chỉ tách nhẹ ở một số vùng. ECDF ít phụ thuộc lựa chọn số bins hơn histogram.

### 16 — Hexbin balance–duration
- **Trục X:** balance. **Trục Y:** duration. Màu là số điểm trong ô lục giác theo log.
- **Nhận xét:** mật độ lớn nhất nằm ở balance thấp-vừa và duration ngắn-vừa; vùng giá trị cao thưa dần. Hexbin khắc phục hiện tượng các điểm đè lên nhau trong scatter.

### 17 — Nhóm tuổi và tỷ lệ yes
- **Trục X:** khoảng tuổi. **Trục Y trái:** số quan sát (cột). **Trục Y phải:** tỷ lệ yes (đường).
- **Nhận xét:** tỷ lệ không tuyến tính theo tuổi; các nhóm rất trẻ và cao tuổi thường cao hơn nhóm trung niên. Cần đọc cùng số lượng vì nhóm biên có ít mẫu hơn.

### 18 — Decile balance và tỷ lệ yes
- **Trục X:** các khoảng phân vị của balance; do nhiều giá trị balance trùng nhau (đặc biệt quanh 0), kích thước nhóm có thể không bằng nhau. **Y trái:** số quan sát. **Y phải:** tỷ lệ yes.
- **Nhận xét:** tỷ lệ yes có xu hướng tăng ở các nhóm balance cao, nhưng không hoàn toàn tuyến tính; điều này giải thích tương quan Pearson với y nhỏ dù vẫn có tín hiệu phân nhóm.

### 19 — Decile duration và tỷ lệ yes
- **Trục X:** 10 nhóm duration gần bằng nhau về số mẫu. **Y trái:** số quan sát. **Y phải:** tỷ lệ yes.
- **Nhận xét:** tỷ lệ yes tăng rất mạnh theo duration, củng cố boxplot/ECDF/correlation. Đây có thể là biến dự báo mạnh nhưng gây rò rỉ thời điểm nếu mục tiêu là dự báo trước cuộc gọi.

### 20 — Pairplot tổng hợp
- **Các trục:** mỗi hàng/cột là một biến; đường chéo là histogram, ô dưới là scatter; màu là y.
- **Nhận xét:** xác nhận trực quan rằng các cặp biến định lượng không có cấu trúc tuyến tính mạnh, trong khi sự phân lớp chủ yếu hiện rõ theo duration. Pairplot dùng mẫu cân bằng để nhìn lớp yes rõ hơn, không dùng để suy ra tỷ trọng.

## Kết luận chính

1. `balance` và `duration` lệch phải, có nhiều outlier; nên dùng median/IQR, biến đổi log có dấu hoặc mô hình bền vững thay vì chỉ dựa mean/std.
2. `duration` liên hệ mạnh nhất với y, nhưng chỉ biết sau cuộc gọi; cần loại biến này nếu bài toán là chấm điểm khách hàng trước khi gọi.
3. `job` chứa tín hiệu phân nhóm: student/retired có tỷ lệ yes cao, nhưng phải xét kích thước và khoảng tin cậy.
4. y mất cân bằng 88.3%/11.7%; đánh giá mô hình cần chỉ số phù hợp mất cân bằng.
5. age và balance đơn lẻ tách lớp yếu; quy luật có vẻ phi tuyến và cần kết hợp thêm biến khác.
