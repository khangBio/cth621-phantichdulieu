"""
KIỂM ĐỊNH TRAIN/TEST CHO BA THUẬT TOÁN GOM CỤM
======================================================================

Clustering không bắt buộc Train/Test như Classification. File này tạo
một kiểm định ổn định bổ sung theo tỷ lệ 80/20:

    - Toàn bộ preprocessing và SVD chỉ fit trên Train.
    - K-Means: fit Train, predict trực tiếp Test.
    - Hierarchical: tạo cây trên mẫu Train, lấy tâm cụm rồi gán Train/Test
      về tâm gần nhất (thuật toán gốc không có predict).
    - DBSCAN: fit Train; Test được gán theo core point gần nhất nếu khoảng
      cách <= eps, ngược lại là Noise (-1).

Kết quả không thay thế bản clustering chính trên toàn bộ dữ liệu; nó dùng
để trình bày mức độ ổn định ngoài mẫu trong báo cáo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
for dependency_dir in reversed(
    [SCRIPT_DIR / ".ml_deps", SCRIPT_DIR / ".viz_deps"]
):
    if dependency_dir.exists():
        sys.path.insert(0, str(dependency_dir))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib_cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clustering_bank_preprocessed_v4 import load_and_prepare_data


RANDOM_STATE = 42
TEST_SIZE = 0.20
KMEANS_K = 5
HIERARCHICAL_K = 5
HIERARCHICAL_TRAIN_SAMPLE = 2_000
DBSCAN_EPS = 1.1088466854779546
DBSCAN_MIN_SAMPLES = 20
SVD_COMPONENTS = 10
PLOT_SAMPLE = 4_000


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm định Train/Test bổ sung cho Clustering v4."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "bank" / "bank_full_preprocess_template_v4.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "clustering_results_preprocessed_v4",
    )
    return parser.parse_args()


def fit_train_feature_space(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    numerical = train_features.select_dtypes(include="number").columns.tolist()
    categorical = train_features.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )

    encoded_train = preprocessor.fit_transform(train_features)
    encoded_test = preprocessor.transform(test_features)

    components = min(SVD_COMPONENTS, encoded_train.shape[1] - 1)
    svd = TruncatedSVD(n_components=components, random_state=RANDOM_STATE)
    reduced_train_unscaled = svd.fit_transform(encoded_train)
    reduced_test_unscaled = svd.transform(encoded_test)

    scaler = StandardScaler()
    reduced_train = scaler.fit_transform(reduced_train_unscaled)
    reduced_test = scaler.transform(reduced_test_unscaled)

    return (
        reduced_train,
        reduced_test,
        float(svd.explained_variance_ratio_.sum()),
    )


def safe_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != -1
    usable_labels = labels[mask]
    if mask.sum() < 3 or len(np.unique(usable_labels)) < 2:
        return float("nan")
    return float(
        silhouette_score(
            matrix[mask],
            usable_labels,
            sample_size=min(5_000, int(mask.sum())),
            random_state=RANDOM_STATE,
        )
    )


def metric_row(
    algorithm: str,
    split_name: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    prediction_method: str,
) -> dict[str, object]:
    non_noise_clusters = len(set(labels) - {-1})
    noise_count = int(np.sum(labels == -1))
    return {
        "Algorithm": algorithm,
        "Split": split_name,
        "Rows": len(labels),
        "Cluster_count_excluding_noise": non_noise_clusters,
        "Noise_count": noise_count,
        "Noise_percent": noise_count / len(labels) * 100,
        "Silhouette_non_noise": safe_silhouette(matrix, labels),
        "Out_of_sample_assignment": prediction_method,
    }


def nearest_core_predict(
    model: DBSCAN,
    test_matrix: np.ndarray,
) -> np.ndarray:
    """Xấp xỉ predict DBSCAN bằng core point gần nhất trong bán kính eps."""
    if len(model.core_sample_indices_) == 0:
        return np.full(len(test_matrix), -1, dtype=int)

    core_matrix = model.components_
    core_labels = model.labels_[model.core_sample_indices_]
    neighbor_model = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    neighbor_model.fit(core_matrix)
    distances, indices = neighbor_model.kneighbors(test_matrix)

    predictions = np.full(len(test_matrix), -1, dtype=int)
    inside = distances[:, 0] <= model.eps
    predictions[inside] = core_labels[indices[inside, 0]]
    return predictions


def plot_panel(
    axis: plt.Axes,
    matrix: np.ndarray,
    labels: np.ndarray,
    title: str,
    show_legend: bool,
) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(PLOT_SAMPLE, len(matrix))
    sample_indices = rng.choice(len(matrix), size=sample_size, replace=False)
    sample_matrix = matrix[sample_indices]
    sample_labels = labels[sample_indices]

    colors = plt.get_cmap("tab10")
    unique_labels = sorted(np.unique(sample_labels))
    for label in unique_labels:
        mask = sample_labels == label
        if label == -1:
            axis.scatter(
                sample_matrix[mask, 0],
                sample_matrix[mask, 1],
                marker="x",
                c="#303030",
                s=15,
                alpha=0.55,
                label="Noise (-1)",
            )
        else:
            axis.scatter(
                sample_matrix[mask, 0],
                sample_matrix[mask, 1],
                s=13,
                alpha=0.5,
                linewidth=0,
                color=colors(int(label) % 10),
                label=f"Cụm {label}",
            )

    axis.set_title(title, fontweight="bold")
    axis.set_xlabel("Thành phần giảm chiều 1")
    axis.set_ylabel("Thành phần giảm chiều 2")
    axis.grid(alpha=0.2)
    if show_legend:
        axis.legend(
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            title="Nhãn cụm",
        )


def save_train_test_plot(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    label_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 17))
    algorithms = ["K-Means", "Hierarchical", "DBSCAN"]

    for row_index, algorithm in enumerate(algorithms):
        train_labels, test_labels = label_sets[algorithm]
        train_metric = metrics[
            (metrics["Algorithm"] == algorithm)
            & (metrics["Split"] == "Train")
        ].iloc[0]
        test_metric = metrics[
            (metrics["Algorithm"] == algorithm)
            & (metrics["Split"] == "Test")
        ].iloc[0]

        plot_panel(
            axes[row_index, 0],
            train_matrix,
            train_labels,
            (
                f"{algorithm} – Train\n"
                f"n={int(train_metric['Rows']):,}, "
                f"Silhouette={train_metric['Silhouette_non_noise']:.3f}, "
                f"Noise={train_metric['Noise_percent']:.1f}%"
            ),
            show_legend=False,
        )
        plot_panel(
            axes[row_index, 1],
            test_matrix,
            test_labels,
            (
                f"{algorithm} – Test\n"
                f"n={int(test_metric['Rows']):,}, "
                f"Silhouette={test_metric['Silhouette_non_noise']:.3f}, "
                f"Noise={test_metric['Noise_percent']:.1f}%"
            ),
            show_legend=True,
        )

        x_values = np.concatenate([train_matrix[:, 0], test_matrix[:, 0]])
        y_values = np.concatenate([train_matrix[:, 1], test_matrix[:, 1]])
        x_limits = (
            float(np.quantile(x_values, 0.002)),
            float(np.quantile(x_values, 0.998)),
        )
        y_limits = (
            float(np.quantile(y_values, 0.002)),
            float(np.quantile(y_values, 0.998)),
        )
        axes[row_index, 0].set_xlim(x_limits)
        axes[row_index, 1].set_xlim(x_limits)
        axes[row_index, 0].set_ylim(y_limits)
        axes[row_index, 1].set_ylim(y_limits)

    fig.suptitle(
        "Kiểm định ổn định Train/Test của ba thuật toán gom cụm",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_report_section(
    metrics: pd.DataFrame,
    explained_variance: float,
    output_dir: Path,
) -> None:
    k_selection = pd.read_csv(output_dir / "kmeans_k_selection.csv")
    k_numeric = pd.read_csv(output_dir / "kmeans_numeric_profile.csv")
    k_categorical = pd.read_csv(
        output_dir / "kmeans_categorical_profile.csv"
    )
    hierarchical = pd.read_csv(output_dir / "hierarchical_summary.csv")
    dbscan_search = pd.read_csv(output_dir / "dbscan_eps_search.csv")
    dbscan_profile = pd.read_csv(output_dir / "dbscan_numeric_profile.csv")

    selected_k = int(k_selection["Selected_K"].iloc[0])
    elbow_k = int(k_selection["Elbow_K"].iloc[0])
    selected_k_row = k_selection[k_selection["K"] == selected_k].iloc[0]
    max_silhouette_row = k_selection.loc[k_selection["Silhouette"].idxmax()]
    selected_eps = float(dbscan_search["Selected_eps"].iloc[0])
    selected_dbscan = dbscan_search.loc[
        (dbscan_search["Eps"] - selected_eps).abs().idxmin()
    ]

    joined_profile = k_numeric.merge(
        k_categorical,
        on=["KMeans_cluster", "Cluster_size", "Cluster_percent"],
        how="left",
    )
    report_columns = [
        "KMeans_cluster",
        "Cluster_size",
        "Cluster_percent",
        "age",
        "balance",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "y_yes_rate",
        "housing_yes_rate",
        "loan_yes_rate",
        "Top_job",
        "Top_education",
        "Top_contact",
        "Top_poutcome",
    ]
    report_columns = [c for c in report_columns if c in joined_profile.columns]

    report = f"""# Nội dung báo cáo Gom cụm – Bank Marketing v4

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
{explained_variance * 100:.2f}% phương sai.

## 2. Lập luận lựa chọn số cụm và so sánh thuật toán

K-Means được thử từ K=2 đến K=10. Elbow xác định điểm gãy tại K={elbow_k}.
Silhouette cao nhất tuyệt đối là K={int(max_silhouette_row['K'])}
({max_silhouette_row['Silhouette']:.4f}), nhưng chỉ tạo hai nhóm tổng quát.
K={selected_k} được chọn để cân bằng giữa điểm gãy WCSS và khả năng tạo phân
khúc marketing chi tiết. Tại K={selected_k}, Silhouette =
{selected_k_row['Silhouette']:.4f}, Davies–Bouldin =
{selected_k_row['Davies_Bouldin']:.4f}.

| Thuật toán | Cấu hình chính | Số cụm | Silhouette | Nhận xét |
|---|---|---:|---:|---|
| K-Means | K={selected_k} | {selected_k} | {selected_k_row['Silhouette']:.4f} | Bao phủ toàn bộ khách hàng, phù hợp làm phân khúc chính |
| Hierarchical Ward | Mẫu {int(hierarchical.iloc[0]['Sample_size']):,}, cắt K={selected_k} | {int(hierarchical.iloc[0]['Actual_cluster_count'])} | {hierarchical.iloc[0]['Silhouette']:.4f} | Trực quan hóa cấu trúc lồng nhau bằng Dendrogram |
| DBSCAN | eps={selected_eps:.4f}, min_samples={DBSCAN_MIN_SAMPLES} | {int(selected_dbscan['Cluster_count'])} + Noise | {selected_dbscan['Silhouette_non_noise']:.4f} | Phù hợp phát hiện vùng mật độ thấp và Noise |

Silhouette của DBSCAN chỉ tính trên phần không phải Noise; Hierarchical tính
trên mẫu 2.000 dòng, nên các con số dùng để tham khảo tương đối chứ không phải
so sánh tuyệt đối hoàn toàn đồng nhất.

## 3. Đọc vị bản chất các cụm K-Means

```text
{joined_profile[report_columns].round(3).to_string(index=False)}
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
{dbscan_profile.loc[dbscan_profile['DBSCAN_cluster'] == 0, 'Cluster_percent'].iloc[0]:.2f}%,
ba vi cụm và {selected_dbscan['Noise_percent']:.2f}% Noise. Vì vậy DBSCAN hữu
ích để cô lập điểm khác biệt hơn là dùng làm phân khúc khách hàng chính.

## 4. Biểu đồ và kiểm định Train/Test

Clustering chính được huấn luyện trên toàn bộ dữ liệu vì không có nhãn mục tiêu
và không cần tập Test để tính Accuracy. Để kiểm tra khả năng ổn định ngoài mẫu,
dữ liệu được chia ngẫu nhiên 80% Train và 20% Test; toàn bộ preprocessing và SVD
chỉ fit trên Train.

```text
{metrics.round(4).to_string(index=False)}
```

- K-Means có `predict()` tự nhiên: tâm cụm học trên Train được dùng gán Test.
- Hierarchical không có `predict()`: cây được tạo trên mẫu Train, sau đó Train
  và Test được gán vào tâm cụm gần nhất; đây là phép xấp xỉ để kiểm định.
- DBSCAN không có `predict()` chuẩn: Test được gán theo core point gần nhất nếu
  khoảng cách không vượt eps; ngoài vùng mật độ được gán Noise.
- Nếu Silhouette và tỷ lệ Noise giữa Train/Test gần nhau, cấu trúc cụm có tính
  ổn định tương đối; chênh lệch lớn cho thấy mô hình nhạy với mẫu dữ liệu.
"""

    (output_dir / "NOI_DUNG_BAO_CAO_CLUSTERING_V4.md").write_text(
        report, encoding="utf-8-sig"
    )


def main() -> None:
    args = parse_arguments()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features, _, _ = load_and_prepare_data(args.input.resolve())
    all_indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train_features = features.iloc[train_indices]
    test_features = features.iloc[test_indices]

    train_matrix, test_matrix, explained_variance = fit_train_feature_space(
        train_features, test_features
    )

    rows = []
    label_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # K-Means
    kmeans = KMeans(
        n_clusters=KMEANS_K,
        n_init=20,
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    kmeans_train = kmeans.fit_predict(train_matrix)
    kmeans_test = kmeans.predict(test_matrix)
    label_sets["K-Means"] = (kmeans_train, kmeans_test)
    rows.append(
        metric_row(
            "K-Means", "Train", train_matrix, kmeans_train, "fit_predict"
        )
    )
    rows.append(
        metric_row(
            "K-Means", "Test", test_matrix, kmeans_test, "KMeans.predict"
        )
    )

    # Hierarchical trên mẫu Train, sau đó dùng nearest centroid.
    rng = np.random.default_rng(RANDOM_STATE)
    hierarchy_sample_size = min(HIERARCHICAL_TRAIN_SAMPLE, len(train_matrix))
    hierarchy_sample_indices = rng.choice(
        len(train_matrix), size=hierarchy_sample_size, replace=False
    )
    hierarchy_sample = train_matrix[hierarchy_sample_indices]
    linkage_matrix = linkage(hierarchy_sample, method="ward")
    hierarchy_sample_labels = (
        fcluster(linkage_matrix, t=HIERARCHICAL_K, criterion="maxclust") - 1
    )
    hierarchy_centroids = np.vstack(
        [
            hierarchy_sample[hierarchy_sample_labels == cluster].mean(axis=0)
            for cluster in sorted(np.unique(hierarchy_sample_labels))
        ]
    )
    hierarchy_train, _ = pairwise_distances_argmin_min(
        train_matrix, hierarchy_centroids
    )
    hierarchy_test, _ = pairwise_distances_argmin_min(
        test_matrix, hierarchy_centroids
    )
    label_sets["Hierarchical"] = (hierarchy_train, hierarchy_test)
    rows.append(
        metric_row(
            "Hierarchical",
            "Train",
            train_matrix,
            hierarchy_train,
            "Nearest centroid from Ward sample",
        )
    )
    rows.append(
        metric_row(
            "Hierarchical",
            "Test",
            test_matrix,
            hierarchy_test,
            "Nearest centroid from Ward sample",
        )
    )

    # DBSCAN fit Train; Test theo core point gần nhất.
    dbscan = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        n_jobs=-1,
    )
    dbscan_train = dbscan.fit_predict(train_matrix)
    dbscan_test = nearest_core_predict(dbscan, test_matrix)
    label_sets["DBSCAN"] = (dbscan_train, dbscan_test)
    rows.append(
        metric_row(
            "DBSCAN", "Train", train_matrix, dbscan_train, "fit_predict"
        )
    )
    rows.append(
        metric_row(
            "DBSCAN",
            "Test",
            test_matrix,
            dbscan_test,
            "Nearest DBSCAN core point within eps",
        )
    )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        output_dir / "train_test_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    assignments = pd.DataFrame(
        {
            "Row_ID": all_indices + 1,
            "Split": "Train",
        }
    )
    assignments.loc[test_indices, "Split"] = "Test"
    for algorithm, (train_labels, test_labels) in label_sets.items():
        column = f"{algorithm.replace('-', '').replace(' ', '_')}_cluster"
        values = np.full(len(features), -999, dtype=int)
        values[train_indices] = train_labels
        values[test_indices] = test_labels
        assignments[column] = values
    assignments.to_csv(
        output_dir / "train_test_cluster_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_train_test_plot(
        train_matrix,
        test_matrix,
        label_sets,
        metrics,
        output_dir / "clustering_train_test_comparison.png",
    )
    save_report_section(metrics, explained_variance, output_dir)

    print(metrics.to_string(index=False))
    print(f"\nKết quả: {output_dir}")


if __name__ == "__main__":
    main()
