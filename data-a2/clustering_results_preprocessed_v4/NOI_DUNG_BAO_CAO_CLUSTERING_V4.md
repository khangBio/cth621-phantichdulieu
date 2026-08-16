# Nội dung báo cáo Gom cụm – Bank Marketing v4

## 1. Quá trình ẩn nhãn và đưa không gian đặc trưng vào mô hình

Để bảo đảm đúng bản chất học không giám sát, ba cột từng đóng vai trò nhãn
trong bài Classification gồm `y`, `housing`, `loan` được tách khỏi ma trận X
trước khi huấn luyện. Các cột này không tham gia tính khoảng cách, xác định tâm
cụm hoặc mật độ; chúng chỉ được nối lại sau cùng để đọc vị ý nghĩa kinh doanh
của từng cụm.

Ba biến số có ngoại lai `age`, `balance`, `duration` được thay bằng
`age_capped`, `balance_capped`, `duration_capped`. Không đưa đồng thời các bản
gốc/fill/capped/minmax/zscore vào X để tránh lặp lại cùng một thông tin.

Không gian huấn luyện gồm 14 biến: 7 biến số và 7 biến định tính. Biến số được
điền Median và StandardScaler; biến định tính được điền Mode và One-Hot
Encoding. Sau mã hóa có 47 chiều. Truncated SVD giảm còn 10 chiều và giữ
78.29% phương sai.

## 2. Lập luận lựa chọn số cụm và so sánh thuật toán

K-Means được thử từ K=2 đến K=10. Elbow xác định điểm gãy tại K=5.
Silhouette cao nhất tuyệt đối là K=2
(0.2493), nhưng chỉ tạo hai nhóm tổng quát.
K=5 được chọn để cân bằng giữa điểm gãy WCSS và khả năng tạo phân
khúc marketing chi tiết. Tại K=5, Silhouette =
0.1497, Davies–Bouldin =
1.7788.

| Thuật toán | Cấu hình chính | Số cụm | Silhouette | Nhận xét |
|---|---|---:|---:|---|
| K-Means | K=5 | 5 | 0.1497 | Bao phủ toàn bộ khách hàng, phù hợp làm phân khúc chính |
| Hierarchical Ward | Mẫu 2,000, cắt K=5 | 5 | 0.1222 | Trực quan hóa cấu trúc lồng nhau bằng Dendrogram |
| DBSCAN | eps=1.1088, min_samples=20 | 4 + Noise | 0.1400 | Phù hợp phát hiện vùng mật độ thấp và Noise |

Silhouette của DBSCAN chỉ tính trên phần không phải Noise; Hierarchical tính
trên mẫu 2.000 dòng, nên các con số dùng để tham khảo tương đối chứ không phải
so sánh tuyệt đối hoàn toàn đồng nhất.

## 3. Đọc vị bản chất các cụm K-Means

```text
 KMeans_cluster  Cluster_size  Cluster_percent    age  balance  duration  campaign   pdays  previous  y_yes_rate  housing_yes_rate  loan_yes_rate     Top_job Top_education Top_contact Top_poutcome
              0          7642           16.903 40.915 1090.834   242.230     2.074 236.746     3.262      22.167            64.159         13.949 blue-collar     secondary    cellular      failure
              1         13761           30.437 41.613  858.988   243.713     2.426   0.236     0.037      11.765            47.039         19.236 blue-collar     secondary    cellular      unknown
              2          8993           19.891 39.257 1060.066   233.181     2.483   2.063     0.085      14.745            39.019         13.599  management      tertiary    cellular      unknown
              3         13392           29.621 41.182  849.537   232.073     2.320  -0.991     0.001       4.383            71.110         15.300 blue-collar     secondary     unknown      unknown
              4          1423            3.147 40.641  806.013   149.030    15.691  -0.253     0.015       4.427            50.738         18.201 blue-collar     secondary    cellular      unknown
```

- **Cụm 0 – khách hàng từng tương tác và tiềm năng cao:** previous và pdays cao,
  tỷ lệ `y=yes` hậu nghiệm đạt 22,17%, cao nhất trong năm cụm.
- **Cụm 1 – nhóm phổ thông quy mô lớn:** chiếm khoảng 30,44%, thường liên hệ
  cellular; tỷ lệ vay cá nhân cao nhất trong các cụm lớn.
- **Cụm 2 – nhóm quản lý/học vấn cao:** nghề phổ biến management, trình độ
  tertiary, housing thấp và tỷ lệ `y=yes` khoảng 14,74%.
- **Cụm 3 – nhóm vay nhà cao, khó chuyển đổi:** housing khoảng 71,11%,
  contact thường unknown, `y=yes` chỉ khoảng 4,38%.
- **Cụm 4 – nhóm bị liên hệ dày:** campaign trung bình khoảng 15,69 lần nhưng
  duration thấp và `y=yes` chỉ 4,43%, biểu hiện mệt mỏi do chiến dịch.

Các tỷ lệ `y`, `housing`, `loan` là phân tích hậu nghiệm, không tham gia tạo cụm.

DBSCAN tạo một cụm chính chiếm
80.29%,
ba vi cụm và 19.55% Noise. Vì vậy DBSCAN hữu
ích để cô lập điểm khác biệt hơn là dùng làm phân khúc khách hàng chính.

## 4. Biểu đồ và kiểm định Train/Test

Clustering chính được huấn luyện trên toàn bộ dữ liệu vì không có nhãn mục tiêu
và không cần tập Test để tính Accuracy. Để kiểm tra khả năng ổn định ngoài mẫu,
dữ liệu được chia ngẫu nhiên 80% Train và 20% Test; toàn bộ preprocessing và SVD
chỉ fit trên Train.

```text
   Algorithm Split  Rows  Cluster_count_excluding_noise  Noise_count  Noise_percent  Silhouette_non_noise             Out_of_sample_assignment
     K-Means Train 36168                              5            0         0.0000                0.1551                          fit_predict
     K-Means  Test  9043                              5            0         0.0000                0.1565                       KMeans.predict
Hierarchical Train 36168                              5            0         0.0000                0.1402    Nearest centroid from Ward sample
Hierarchical  Test  9043                              5            0         0.0000                0.1379    Nearest centroid from Ward sample
      DBSCAN Train 36168                              9         8003        22.1273                0.0504                          fit_predict
      DBSCAN  Test  9043                              9         2106        23.2887                0.0607 Nearest DBSCAN core point within eps
```

- K-Means có `predict()` tự nhiên: tâm cụm học trên Train được dùng gán Test.
- Hierarchical không có `predict()`: cây được tạo trên mẫu Train, sau đó Train
  và Test được gán vào tâm cụm gần nhất; đây là phép xấp xỉ để kiểm định.
- DBSCAN không có `predict()` chuẩn: Test được gán theo core point gần nhất nếu
  khoảng cách không vượt eps; ngoài vùng mật độ được gán Noise.
- Nếu Silhouette và tỷ lệ Noise giữa Train/Test gần nhau, cấu trúc cụm có tính
  ổn định tương đối; chênh lệch lớn cho thấy mô hình nhạy với mẫu dữ liệu.
