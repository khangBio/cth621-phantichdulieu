# Báo cáo Gom cụm – Bank Marketing Preprocessed v4

## 1. Không gian đặc trưng

- Số quan sát: 45,211.
- Số biến trước mã hóa: 14.
- Loại hoàn toàn khỏi đầu vào: `y`, `housing`, `loan`.
- Dùng `age_capped`, `balance_capped`, `duration_capped`.
- Biến số: Median Imputation + StandardScaler.
- Biến định tính: Mode Imputation + One-Hot Encoding.
- Sau mã hóa: 47 chiều.
- Truncated SVD: 10 chiều, giải thích 78.30% phương sai.

Các nhãn cũ chỉ dùng để mô tả cụm sau huấn luyện:

- `y` Yes = 11.70% (không dùng khi huấn luyện).
- `housing` Yes = 55.58% (không dùng khi huấn luyện).
- `loan` Yes = 16.02% (không dùng khi huấn luyện).

## 2. K-Means và lựa chọn K

- Thử K từ 2 đến 10.
- K theo Elbow: **5**.
- K có Silhouette cao nhất toàn miền thử: **2** với Silhouette = **0.2493**.
- K được chọn: **5**, chọn trong vùng lân cận Elbow bằng Silhouette cao và Davies–Bouldin thấp.
- K=5: Silhouette = **0.1497**, Calinski–Harabasz = **5514.34**, Davies–Bouldin = **1.7788**.

### Hồ sơ định lượng K-Means

```text
 KMeans_cluster  Cluster_size  Cluster_percent    age  balance  duration  campaign   pdays  previous
              0          7642           16.903 40.915 1090.834   242.230     2.074 236.746     3.262
              1         13761           30.437 41.613  858.988   243.713     2.426   0.236     0.037
              2          8993           19.891 39.257 1060.066   233.181     2.483   2.063     0.085
              3         13392           29.621 41.182  849.537   232.073     2.320  -0.991     0.001
              4          1423            3.147 40.641  806.013   149.030    15.691  -0.253     0.015
```

### Hồ sơ định tính và tỷ lệ nhãn giữ lại để diễn giải

```text
 KMeans_cluster  y_yes_rate  housing_yes_rate  loan_yes_rate     Top_job Top_marital Top_education Top_contact Top_poutcome
              0      22.167            64.159         13.949 blue-collar     married     secondary    cellular      failure
              1      11.765            47.039         19.236 blue-collar     married     secondary    cellular      unknown
              2      14.745            39.019         13.599  management     married      tertiary    cellular      unknown
              3       4.383            71.110         15.300 blue-collar     married     secondary     unknown      unknown
              4       4.427            50.738         18.201 blue-collar     married     secondary    cellular      unknown
```

Tỷ lệ `y`, `housing`, `loan` ở trên là phân tích hậu nghiệm, không tham gia tạo cụm.

## 3. Hierarchical Clustering

- Phương pháp: Ward linkage với khoảng cách Euclidean.
- Dùng mẫu cố định 2,000 quan sát vì ma trận khoảng cách phân cấp tăng theo O(n²).
- Số cụm cắt từ cây: 5.
- Silhouette trên mẫu: **0.1222**.
- Adjusted Rand so với K-Means trên cùng mẫu: **0.4115**.
- Cophenetic correlation: **0.5280**.

Dendrogram cho thấy thứ tự hợp nhất các nhóm; độ cao trục Y càng lớn nghĩa là hai nhánh được ghép ở khoảng cách càng xa.

## 4. DBSCAN

- `min_samples` = 20.
- `eps` được chọn từ k-distance search: **1.1088**.
- Số cụm không tính Noise: **4**.
- Tỷ lệ Noise (`cluster=-1`): **19.55%**.
- Cụm mật độ lớn nhất chiếm **80.29%** toàn bộ dữ liệu.

DBSCAN không bắt buộc mọi điểm thuộc một cụm. Các dòng có nhãn `-1` là khách hàng nằm trong vùng mật độ thấp và được xem là Noise/điểm khác biệt trong không gian đặc trưng.

### Hồ sơ các cụm và Noise

```text
 DBSCAN_cluster  Cluster_size  Cluster_percent    age  balance  duration  campaign   pdays  previous
             -1          8841           19.555 43.260 1382.345   280.002     4.479 132.449     2.389
              0         36300           80.290 40.303  822.076   224.074     2.349  17.539     0.137
              1            39            0.086 33.410 3357.615   200.000     1.333 144.333     1.667
              2            17            0.038 32.118  436.706   115.588     1.353 160.471     1.765
              3            14            0.031 25.786  911.786   192.214     1.071  98.143     2.000
```

```text
 DBSCAN_cluster  y_yes_rate  housing_yes_rate  loan_yes_rate     Top_job Top_marital Top_education
             -1      24.364            52.562         13.551  management     married     secondary
              0       8.587            56.339         16.634 blue-collar     married     secondary
              1      20.513            66.667          5.128  management      single      tertiary
              2      23.529            29.412         29.412      admin.      single      tertiary
              3      42.857             7.143          7.143     student      single     secondary
```

Nếu một cụm chiếm phần lớn dữ liệu và các cụm còn lại rất nhỏ, DBSCAN nên được
dùng chủ yếu để nhận diện Noise thay vì làm phương án phân khúc khách hàng chính.

### Các cấu hình eps đã thử

```text
   Eps  Min_samples  Cluster_count  Noise_count  Noise_percent  Silhouette_non_noise  Valid_candidate  Selection_objective  Selected_eps
0.8811           20             12        18133        40.1075               -0.1566            False            -999.0000        1.1088
0.9620           20              9        14300        31.6295               -0.0916             True              -0.1781        1.1088
1.0560           20              9        10551        23.3372                0.0867             True               0.0334        1.1088
1.1088           20              4         8841        19.5550                0.1400             True               0.1018        1.1088
1.1681           20              6         7237        16.0072                0.0378             True               0.0138        1.1088
1.2391           20              2         5698        12.6031                0.0625             True               0.0521        1.1088
1.3248           20              1         4348         9.6171                   NaN            False            -999.0000        1.1088
1.4316           20              1         3106         6.8700                   NaN            False            -999.0000        1.1088
1.5894           20              1         1780         3.9371                   NaN            False            -999.0000        1.1088
1.8481           20              1          758         1.6766                   NaN            False            -999.0000        1.1088
2.2012           20              1          293         0.6481                   NaN            False            -999.0000        1.1088
```

## 5. Nhận xét phương pháp

- K-Means phù hợp để tạo phân khúc bao phủ toàn bộ khách hàng và dễ lập hồ sơ cụm.
- Hierarchical giúp quan sát cấu trúc lồng nhau nhưng không thực tế khi chạy linkage đầy đủ trên 45.211 dòng, nên lấy mẫu tái lập bằng `random_state=42`.
- DBSCAN hữu ích để phát hiện Noise, nhưng kết quả nhạy với `eps`, `min_samples` và số chiều.
- One-Hot Encoding + SVD biến dữ liệu hỗn hợp thành không gian Euclidean; đây là xấp xỉ thực dụng. Nếu mục tiêu chủ yếu là dữ liệu phân loại, có thể khảo sát thêm Gower Distance hoặc K-Prototypes.
