"""
GOM CỤM KHÁCH HÀNG BANK MARKETING – PREPROCESSED TEMPLATE V4
=======================================================================

Nguồn dữ liệu:
    bank/bank_full_preprocess_template_v4.xlsx

Nguyên tắc:
    - Lấy dữ liệu đầy đủ từ sheet Data_goc.
    - Thay age, balance, duration bằng các phiên bản *_capped trong
      sheet Preprocess.
    - Loại hoàn toàn y, housing, loan khỏi không gian huấn luyện.
      Ba cột này chỉ được dùng sau khi gom cụm để mô tả/đánh giá cụm.
    - One-Hot Encoding biến định tính, Standard Scaling biến số.
    - Truncated SVD giảm chiều trước khi dùng khoảng cách Euclidean.

Thuật toán:
    1. K-Means: thử K=2..10, dùng Elbow, Silhouette,
       Calinski-Harabasz và Davies-Bouldin để chọn K.
    2. Hierarchical Clustering: Ward linkage trên mẫu đại diện vì thuật
       toán cần bộ nhớ/thời gian xấp xỉ O(n^2); xuất Dendrogram.
    3. DBSCAN: chọn eps bằng k-distance và thử nhiều phân vị; điểm có
       nhãn -1 được xem là Noise/Outlier.

Chạy:
    python clustering_bank_preprocessed_v4.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_ML_DEPS = SCRIPT_DIR / ".ml_deps"
LOCAL_VIZ_DEPS = SCRIPT_DIR / ".viz_deps"

for dependency_dir in reversed([LOCAL_ML_DEPS, LOCAL_VIZ_DEPS]):
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
import seaborn as sns
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COLUMNS = ["y", "housing", "loan"]
CLEAN_VARIABLES = ["age", "balance", "duration"]
K_VALUES = list(range(2, 11))
SVD_COMPONENTS = 10
HIERARCHICAL_SAMPLE_SIZE = 2_000
PLOT_SAMPLE_SIZE = 8_000
DBSCAN_MIN_SAMPLES = 20
DBSCAN_EPS_QUANTILES = [
    0.40,
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.98,
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="K-Means, Hierarchical và DBSCAN trên Bank Marketing v4."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "bank" / "bank_full_preprocess_template_v4.xlsx",
        help="Đường dẫn workbook v4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "clustering_results_preprocessed_v4",
        help="Thư mục lưu kết quả.",
    )
    parser.add_argument(
        "--hierarchical-sample",
        type=int,
        default=HIERARCHICAL_SAMPLE_SIZE,
        help="Số quan sát dùng cho Hierarchical/Dendrogram.",
    )
    return parser.parse_args()


def load_and_prepare_data(
    workbook_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc v4, kiểm tra căn chỉnh và tạo X không chứa nhãn."""
    if not workbook_path.exists():
        raise FileNotFoundError(f"Không tìm thấy workbook: {workbook_path}")

    excel_file = pd.ExcelFile(workbook_path)
    required_sheets = {"Data_goc", "Preprocess"}
    missing_sheets = required_sheets.difference(excel_file.sheet_names)
    if missing_sheets:
        raise ValueError(f"Workbook thiếu sheet: {sorted(missing_sheets)}")

    original = pd.read_excel(workbook_path, sheet_name="Data_goc")
    preprocess = pd.read_excel(workbook_path, sheet_name="Preprocess")

    missing_targets = [c for c in TARGET_COLUMNS if c not in original.columns]
    if missing_targets:
        raise ValueError(f"Data_goc thiếu cột nhãn: {missing_targets}")
    if len(original) != len(preprocess):
        raise ValueError(
            f"Số dòng không khớp: Data_goc={len(original):,}, "
            f"Preprocess={len(preprocess):,}."
        )

    modeling_data = original.copy()
    audit_rows = []

    for variable in CLEAN_VARIABLES:
        required = [f"{variable}_goc", f"{variable}_capped"]
        missing = [c for c in required if c not in preprocess.columns]
        if missing:
            raise ValueError(f"Preprocess thiếu cột: {missing}")

        source_values = pd.to_numeric(original[variable], errors="coerce")
        preprocess_source = pd.to_numeric(
            preprocess[f"{variable}_goc"], errors="coerce"
        )
        capped_values = pd.to_numeric(
            preprocess[f"{variable}_capped"], errors="coerce"
        )

        aligned = bool(
            np.allclose(
                source_values.to_numpy(dtype=float),
                preprocess_source.to_numpy(dtype=float),
                equal_nan=True,
            )
        )
        if not aligned:
            raise ValueError(
                f"{variable}_goc không khớp Data_goc; không thể ghép an toàn."
            )
        if capped_values.isna().any():
            raise ValueError(f"{variable}_capped còn giá trị không hợp lệ.")

        modeling_data[variable] = capped_values.to_numpy()
        audit_rows.append(
            {
                "Variable": variable,
                "Rows": len(original),
                "Original_matches_preprocess": aligned,
                "Original_min": float(source_values.min()),
                "Original_max": float(source_values.max()),
                "Capped_min": float(capped_values.min()),
                "Capped_max": float(capped_values.max()),
                "Capped_changed_count": int(
                    (source_values.to_numpy() != capped_values.to_numpy()).sum()
                ),
            }
        )

    held_out_labels = modeling_data[TARGET_COLUMNS].copy()
    features = modeling_data.drop(columns=TARGET_COLUMNS)

    return features, held_out_labels, pd.DataFrame(audit_rows)


def build_feature_space(
    features: pd.DataFrame,
) -> tuple[np.ndarray, ColumnTransformer, TruncatedSVD, pd.DataFrame]:
    """OHE + scaling + SVD, trả về ma trận đặc trưng giảm chiều."""
    numerical_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    encoded = preprocessor.fit_transform(features)
    max_components = min(SVD_COMPONENTS, encoded.shape[1] - 1)
    svd = TruncatedSVD(n_components=max_components, random_state=RANDOM_STATE)
    reduced_unscaled = svd.fit_transform(encoded)
    reduced = StandardScaler().fit_transform(reduced_unscaled)

    summary = pd.DataFrame(
        [
            {
                "Rows": features.shape[0],
                "Original_feature_count": features.shape[1],
                "Numerical_feature_count": len(numerical_columns),
                "Categorical_feature_count": len(categorical_columns),
                "Encoded_feature_count": encoded.shape[1],
                "SVD_component_count": max_components,
                "SVD_explained_variance_ratio": float(
                    svd.explained_variance_ratio_.sum()
                ),
                "Excluded_labels": ", ".join(TARGET_COLUMNS),
                "Numerical_features": ", ".join(numerical_columns),
                "Categorical_features": ", ".join(categorical_columns),
            }
        ]
    )

    return reduced, preprocessor, svd, summary


def save_svd_loadings(
    preprocessor: ColumnTransformer,
    svd: TruncatedSVD,
    output_dir: Path,
) -> None:
    """Lưu các biến mã hóa đóng góp mạnh nhất cho từng thành phần SVD."""
    feature_names = preprocessor.get_feature_names_out()
    rows = []
    for component_index, weights in enumerate(svd.components_, start=1):
        top_indices = np.argsort(np.abs(weights))[::-1][:10]
        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "Component": component_index,
                    "Rank": rank,
                    "Encoded_feature": feature_names[feature_index],
                    "Loading": float(weights[feature_index]),
                    "Absolute_loading": float(abs(weights[feature_index])),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "svd_top_component_loadings.csv",
        index=False,
        encoding="utf-8-sig",
    )


def elbow_from_inertia(k_values: list[int], inertia_values: list[float]) -> int:
    """Tìm điểm xa nhất so với đường nối hai đầu của đường Elbow."""
    points = np.column_stack([k_values, inertia_values]).astype(float)
    start = points[0]
    end = points[-1]
    line_vector = end - start
    line_norm = np.linalg.norm(line_vector)
    if line_norm == 0:
        return k_values[0]
    relative = points - start
    distances = np.abs(
        line_vector[0] * relative[:, 1] - line_vector[1] * relative[:, 0]
    ) / line_norm
    return int(k_values[int(np.argmax(distances))])


def run_kmeans(
    reduced: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, KMeans, pd.DataFrame, int]:
    """Đánh giá K=2..10 và chạy K-Means với K được khuyến nghị."""
    rows = []
    fitted_models: dict[int, KMeans] = {}

    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            n_init=20,
            max_iter=500,
            random_state=RANDOM_STATE,
        )
        labels = model.fit_predict(reduced)
        fitted_models[k] = model
        rows.append(
            {
                "K": k,
                "Inertia": float(model.inertia_),
                "Silhouette": float(
                    silhouette_score(
                        reduced,
                        labels,
                        sample_size=min(10_000, len(reduced)),
                        random_state=RANDOM_STATE,
                    )
                ),
                "Calinski_Harabasz": float(
                    calinski_harabasz_score(reduced, labels)
                ),
                "Davies_Bouldin": float(davies_bouldin_score(reduced, labels)),
            }
        )

    metrics = pd.DataFrame(rows)
    elbow_k = elbow_from_inertia(
        metrics["K"].tolist(), metrics["Inertia"].tolist()
    )
    candidate_k = [
        k for k in [elbow_k - 1, elbow_k, elbow_k + 1] if k in K_VALUES
    ]
    candidate_metrics = metrics[metrics["K"].isin(candidate_k)]
    optimal_k = int(
        candidate_metrics.sort_values(
            ["Silhouette", "Davies_Bouldin"],
            ascending=[False, True],
        ).iloc[0]["K"]
    )
    metrics["Elbow_K"] = elbow_k
    metrics["Selected_K"] = optimal_k
    metrics.to_csv(
        output_dir / "kmeans_k_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model = fitted_models[optimal_k]
    labels = model.labels_

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plots = [
        ("Inertia", "Inertia (WCSS)", axes[0, 0], "#4C78A8"),
        ("Silhouette", "Silhouette Score", axes[0, 1], "#F58518"),
        (
            "Calinski_Harabasz",
            "Calinski–Harabasz",
            axes[1, 0],
            "#54A24B",
        ),
        ("Davies_Bouldin", "Davies–Bouldin (thấp tốt)", axes[1, 1], "#E45756"),
    ]
    for column, title, axis, color in plots:
        axis.plot(metrics["K"], metrics[column], marker="o", color=color)
        axis.axvline(optimal_k, color="#222222", linestyle="--", alpha=0.8)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Số cụm K")
        axis.grid(alpha=0.25)
    axes[0, 0].axvline(elbow_k, color="#B279A2", linestyle=":", linewidth=2)
    fig.suptitle(
        f"Chọn K cho K-Means: Elbow={elbow_k}, K đề xuất={optimal_k}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "kmeans_k_selection.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    return labels, model, metrics, optimal_k


def build_cluster_profile(
    features: pd.DataFrame,
    held_out_labels: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tạo hồ sơ cụm; các nhãn cũ chỉ dùng sau clustering."""
    profile_source = features.copy()
    profile_source[cluster_column] = cluster_labels
    for column in TARGET_COLUMNS:
        profile_source[column] = held_out_labels[column].to_numpy()

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    numeric_profile = (
        profile_source.groupby(cluster_column, dropna=False)[numeric_columns]
        .mean()
        .reset_index()
    )
    sizes = (
        profile_source.groupby(cluster_column, dropna=False)
        .size()
        .rename("Cluster_size")
        .reset_index()
    )
    numeric_profile = sizes.merge(numeric_profile, on=cluster_column)
    numeric_profile["Cluster_percent"] = (
        numeric_profile["Cluster_size"] / len(profile_source) * 100
    )

    categorical_rows = []
    for cluster_value, group in profile_source.groupby(
        cluster_column, dropna=False
    ):
        row = {
            cluster_column: cluster_value,
            "Cluster_size": len(group),
            "Cluster_percent": len(group) / len(profile_source) * 100,
            "y_yes_rate": float((group["y"] == "yes").mean() * 100),
            "housing_yes_rate": float((group["housing"] == "yes").mean() * 100),
            "loan_yes_rate": float((group["loan"] == "yes").mean() * 100),
        }
        for column in ["job", "marital", "education", "contact", "poutcome"]:
            if column in group.columns:
                row[f"Top_{column}"] = str(group[column].mode(dropna=True).iloc[0])
        categorical_rows.append(row)

    return numeric_profile, pd.DataFrame(categorical_rows)


def save_cluster_scatter(
    reduced: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    noise_label: int | None = None,
) -> None:
    """Biểu diễn mẫu trên hai thành phần giảm chiều đầu tiên."""
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(PLOT_SAMPLE_SIZE, len(reduced))
    sample_indices = rng.choice(len(reduced), size=sample_size, replace=False)
    sample_labels = labels[sample_indices]

    plot_data = pd.DataFrame(
        {
            "Component_1": reduced[sample_indices, 0],
            "Component_2": reduced[sample_indices, 1],
            "Cluster": sample_labels.astype(str),
        }
    )

    plt.figure(figsize=(10, 7))
    if noise_label is None:
        sns.scatterplot(
            data=plot_data,
            x="Component_1",
            y="Component_2",
            hue="Cluster",
            palette="tab10",
            s=18,
            alpha=0.6,
            linewidth=0,
        )
    else:
        non_noise = plot_data[plot_data["Cluster"] != str(noise_label)]
        noise = plot_data[plot_data["Cluster"] == str(noise_label)]
        sns.scatterplot(
            data=non_noise,
            x="Component_1",
            y="Component_2",
            hue="Cluster",
            palette="tab10",
            s=18,
            alpha=0.6,
            linewidth=0,
        )
        if not noise.empty:
            plt.scatter(
                noise["Component_1"],
                noise["Component_2"],
                c="#222222",
                marker="x",
                s=20,
                alpha=0.65,
                label="Noise (-1)",
            )
    plt.title(title, fontweight="bold")
    plt.xlabel("Thành phần giảm chiều 1")
    plt.ylabel("Thành phần giảm chiều 2")
    plt.legend(title="Cụm", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def run_hierarchical(
    reduced: np.ndarray,
    kmeans_labels: np.ndarray,
    held_out_labels: pd.DataFrame,
    optimal_k: int,
    sample_size: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Ward Hierarchical trên mẫu, vẽ Dendrogram và đo độ tương đồng."""
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(sample_size, len(reduced))
    sample_indices = np.sort(
        rng.choice(len(reduced), size=sample_size, replace=False)
    )
    sample_matrix = reduced[sample_indices]

    linkage_matrix = linkage(sample_matrix, method="ward")
    hierarchical_labels = (
        fcluster(linkage_matrix, t=optimal_k, criterion="maxclust") - 1
    )
    silhouette = float(silhouette_score(sample_matrix, hierarchical_labels))
    ari = float(
        adjusted_rand_score(kmeans_labels[sample_indices], hierarchical_labels)
    )
    cophenetic_correlation, _ = cophenet(
        linkage_matrix, pdist(sample_matrix)
    )

    plt.figure(figsize=(15, 8))
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=40,
        leaf_rotation=70,
        leaf_font_size=8,
        show_contracted=True,
        color_threshold=None,
    )
    plt.title(
        f"Dendrogram Ward – mẫu {sample_size:,} quan sát (hiển thị 40 nhánh cuối)",
        fontweight="bold",
    )
    plt.xlabel("Nhóm quan sát / cụm trung gian")
    plt.ylabel("Khoảng cách Ward")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        output_dir / "hierarchical_dendrogram.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    assignments = pd.DataFrame(
        {
            "Row_ID": sample_indices + 1,
            "Hierarchical_cluster": hierarchical_labels,
            "KMeans_cluster": kmeans_labels[sample_indices],
            "Component_1": sample_matrix[:, 0],
            "Component_2": sample_matrix[:, 1],
        }
    )
    for column in TARGET_COLUMNS:
        assignments[column] = held_out_labels.iloc[sample_indices][
            column
        ].to_numpy()
    assignments.to_csv(
        output_dir / "hierarchical_sample_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        [
            {
                "Algorithm": "Ward Hierarchical",
                "Sample_size": sample_size,
                "Requested_cluster_count": optimal_k,
                "Actual_cluster_count": int(
                    len(np.unique(hierarchical_labels))
                ),
                "Silhouette": silhouette,
                "Adjusted_Rand_vs_KMeans": ari,
                "Cophenetic_correlation": float(cophenetic_correlation),
                "Reason_for_sampling": (
                    "Hierarchical linkage có chi phí bộ nhớ/thời gian O(n^2)"
                ),
            }
        ]
    )
    summary.to_csv(
        output_dir / "hierarchical_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary


def run_dbscan(
    reduced: np.ndarray,
    features: pd.DataFrame,
    held_out_labels: pd.DataFrame,
    output_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, float, int, float]:
    """Tìm eps từ k-distance, thử ứng viên và chạy DBSCAN."""
    neighbors = NearestNeighbors(
        n_neighbors=DBSCAN_MIN_SAMPLES,
        n_jobs=-1,
    )
    neighbors.fit(reduced)
    distances, _ = neighbors.kneighbors(reduced)
    kth_distances = distances[:, -1]
    sorted_distances = np.sort(kth_distances)

    eps_candidates = sorted(
        {
            float(np.quantile(kth_distances, quantile))
            for quantile in DBSCAN_EPS_QUANTILES
        }
    )
    rows = []
    candidate_labels: dict[float, np.ndarray] = {}

    for eps in eps_candidates:
        model = DBSCAN(
            eps=eps,
            min_samples=DBSCAN_MIN_SAMPLES,
            n_jobs=-1,
        )
        labels = model.fit_predict(reduced)
        candidate_labels[eps] = labels
        cluster_values = set(labels)
        cluster_count = len(cluster_values - {-1})
        noise_count = int(np.sum(labels == -1))
        noise_percent = noise_count / len(labels) * 100

        non_noise_mask = labels != -1
        non_noise_labels = labels[non_noise_mask]
        silhouette = np.nan
        if (
            cluster_count >= 2
            and non_noise_mask.sum() > cluster_count
            and len(np.unique(non_noise_labels)) >= 2
        ):
            silhouette = float(
                silhouette_score(
                    reduced[non_noise_mask],
                    non_noise_labels,
                    sample_size=min(5_000, int(non_noise_mask.sum())),
                    random_state=RANDOM_STATE,
                )
            )

        valid = (
            2 <= cluster_count <= 20
            and 1.0 <= noise_percent <= 40.0
            and not np.isnan(silhouette)
        )
        objective = (
            float(silhouette)
            - 0.004 * abs(noise_percent - 10.0)
            - 0.005 * max(cluster_count - 10, 0)
            if valid
            else -999.0
        )
        rows.append(
            {
                "Eps": eps,
                "Min_samples": DBSCAN_MIN_SAMPLES,
                "Cluster_count": cluster_count,
                "Noise_count": noise_count,
                "Noise_percent": noise_percent,
                "Silhouette_non_noise": silhouette,
                "Valid_candidate": valid,
                "Selection_objective": objective,
            }
        )

    search = pd.DataFrame(rows)
    valid_search = search[search["Valid_candidate"]]
    if not valid_search.empty:
        selected_row = valid_search.sort_values(
            ["Selection_objective", "Silhouette_non_noise"],
            ascending=False,
        ).iloc[0]
    else:
        fallback = search[
            (search["Cluster_count"] >= 2)
            & search["Silhouette_non_noise"].notna()
        ]
        if fallback.empty:
            selected_row = search.iloc[
                (search["Noise_percent"] - 10.0).abs().argmin()
            ]
        else:
            selected_row = fallback.sort_values(
                ["Silhouette_non_noise", "Noise_percent"],
                ascending=[False, True],
            ).iloc[0]

    selected_eps = float(selected_row["Eps"])
    labels = candidate_labels[selected_eps]
    selected_cluster_count = int(selected_row["Cluster_count"])
    selected_noise_percent = float(selected_row["Noise_percent"])
    search["Selected_eps"] = selected_eps
    search.to_csv(
        output_dir / "dbscan_eps_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    x = np.arange(1, len(sorted_distances) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    axes[0].plot(x, sorted_distances, color="#4C78A8", linewidth=1.3)
    axes[0].axhline(
        selected_eps,
        color="#E45756",
        linestyle="--",
        label=f"eps chọn = {selected_eps:.3f}",
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Toàn miền – trục Y log", fontweight="bold")
    axes[0].set_xlabel("Quan sát đã sắp xếp")
    axes[0].set_ylabel(
        f"Khoảng cách tới láng giềng thứ {DBSCAN_MIN_SAMPLES}"
    )
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    tail_start = int(len(sorted_distances) * 0.70)
    axes[1].plot(
        x[tail_start:],
        sorted_distances[tail_start:],
        color="#4C78A8",
        linewidth=1.5,
    )
    axes[1].axhline(
        selected_eps,
        color="#E45756",
        linestyle="--",
        label=f"eps chọn = {selected_eps:.3f}",
    )
    zoom_ceiling = max(
        selected_eps * 2.2,
        float(np.quantile(sorted_distances, 0.985)) * 1.08,
    )
    axes[1].set_ylim(0, zoom_ceiling)
    axes[1].set_title("Phóng to 30% phần đuôi", fontweight="bold")
    axes[1].set_xlabel("Quan sát đã sắp xếp")
    axes[1].set_ylabel(
        f"Khoảng cách tới láng giềng thứ {DBSCAN_MIN_SAMPLES}"
    )
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        f"k-distance plot (k = min_samples = {DBSCAN_MIN_SAMPLES})",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "dbscan_k_distance.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    assignments = pd.DataFrame(
        {
            "Row_ID": np.arange(1, len(labels) + 1),
            "DBSCAN_cluster": labels,
            "Is_noise": labels == -1,
            "Kth_neighbor_distance": kth_distances,
            "Component_1": reduced[:, 0],
            "Component_2": reduced[:, 1],
        }
    )
    for column in TARGET_COLUMNS:
        assignments[column] = held_out_labels[column].to_numpy()
    assignments.to_csv(
        output_dir / "dbscan_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    noise_indices = np.flatnonzero(labels == -1)
    if len(noise_indices):
        ranked_noise = noise_indices[
            np.argsort(kth_distances[noise_indices])[::-1]
        ][:200]
        noise_examples = features.iloc[ranked_noise].copy()
        noise_examples.insert(0, "Row_ID", ranked_noise + 1)
        noise_examples.insert(
            1, "Kth_neighbor_distance", kth_distances[ranked_noise]
        )
        for column in TARGET_COLUMNS:
            noise_examples[column] = held_out_labels.iloc[ranked_noise][
                column
            ].to_numpy()
        noise_examples.to_csv(
            output_dir / "dbscan_top_noise_examples.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return (
        labels,
        search,
        selected_eps,
        selected_cluster_count,
        selected_noise_percent,
    )


def save_kmeans_assignments_and_profiles(
    features: pd.DataFrame,
    held_out_labels: pd.DataFrame,
    reduced: np.ndarray,
    labels: np.ndarray,
    model: KMeans,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    distances = model.transform(reduced).min(axis=1)
    assignments = pd.DataFrame(
        {
            "Row_ID": np.arange(1, len(labels) + 1),
            "KMeans_cluster": labels,
            "Distance_to_centroid": distances,
            "Component_1": reduced[:, 0],
            "Component_2": reduced[:, 1],
        }
    )
    for column in TARGET_COLUMNS:
        assignments[column] = held_out_labels[column].to_numpy()
    assignments.to_csv(
        output_dir / "kmeans_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    numeric_profile, categorical_profile = build_cluster_profile(
        features,
        held_out_labels,
        labels,
        "KMeans_cluster",
    )
    numeric_profile.to_csv(
        output_dir / "kmeans_numeric_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    categorical_profile.to_csv(
        output_dir / "kmeans_categorical_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(9, 5.5))
    size_data = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Count")
    )
    size_data["Percent"] = size_data["Count"] / len(labels) * 100
    axis = sns.barplot(
        data=size_data,
        x="Cluster",
        y="Percent",
        color="#4C78A8",
    )
    axis.bar_label(axis.containers[0], fmt="%.1f%%", padding=3)
    plt.title("Quy mô các cụm K-Means", fontweight="bold")
    plt.xlabel("Cụm")
    plt.ylabel("Tỷ lệ quan sát (%)")
    plt.ylim(0, max(size_data["Percent"]) * 1.18)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(
        output_dir / "kmeans_cluster_sizes.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    return numeric_profile, categorical_profile


def save_dbscan_profiles(
    features: pd.DataFrame,
    held_out_labels: pd.DataFrame,
    labels: np.ndarray,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric_profile, categorical_profile = build_cluster_profile(
        features,
        held_out_labels,
        labels,
        "DBSCAN_cluster",
    )
    numeric_profile.to_csv(
        output_dir / "dbscan_numeric_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    categorical_profile.to_csv(
        output_dir / "dbscan_categorical_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return numeric_profile, categorical_profile


def save_report(
    features: pd.DataFrame,
    held_out_labels: pd.DataFrame,
    feature_summary: pd.DataFrame,
    k_metrics: pd.DataFrame,
    optimal_k: int,
    k_numeric_profile: pd.DataFrame,
    k_categorical_profile: pd.DataFrame,
    hierarchical_summary: pd.DataFrame,
    dbscan_search: pd.DataFrame,
    selected_eps: float,
    dbscan_cluster_count: int,
    dbscan_noise_percent: float,
    dbscan_numeric_profile: pd.DataFrame,
    dbscan_categorical_profile: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Tạo báo cáo phương pháp, kết quả và diễn giải cụm."""
    selected_k_row = k_metrics[k_metrics["K"] == optimal_k].iloc[0]
    best_silhouette_row = k_metrics.loc[k_metrics["Silhouette"].idxmax()]
    elbow_k = int(k_metrics["Elbow_K"].iloc[0])
    hierarchical_row = hierarchical_summary.iloc[0]

    class_distribution_lines = []
    for column in TARGET_COLUMNS:
        yes_rate = float((held_out_labels[column] == "yes").mean() * 100)
        class_distribution_lines.append(
            f"- `{column}` Yes = {yes_rate:.2f}% (không dùng khi huấn luyện)."
        )

    profile_columns = [
        c
        for c in [
            "KMeans_cluster",
            "Cluster_size",
            "Cluster_percent",
            "age",
            "balance",
            "duration",
            "campaign",
            "pdays",
            "previous",
        ]
        if c in k_numeric_profile.columns
    ]
    categorical_columns = [
        c
        for c in [
            "KMeans_cluster",
            "y_yes_rate",
            "housing_yes_rate",
            "loan_yes_rate",
            "Top_job",
            "Top_marital",
            "Top_education",
            "Top_contact",
            "Top_poutcome",
        ]
        if c in k_categorical_profile.columns
    ]
    dbscan_numeric_columns = [
        c
        for c in [
            "DBSCAN_cluster",
            "Cluster_size",
            "Cluster_percent",
            "age",
            "balance",
            "duration",
            "campaign",
            "pdays",
            "previous",
        ]
        if c in dbscan_numeric_profile.columns
    ]
    dbscan_categorical_columns = [
        c
        for c in [
            "DBSCAN_cluster",
            "y_yes_rate",
            "housing_yes_rate",
            "loan_yes_rate",
            "Top_job",
            "Top_marital",
            "Top_education",
        ]
        if c in dbscan_categorical_profile.columns
    ]
    non_noise_profile = dbscan_numeric_profile[
        dbscan_numeric_profile["DBSCAN_cluster"] != -1
    ]
    largest_dbscan_cluster_percent = float(
        non_noise_profile["Cluster_percent"].max()
    )

    report = f"""# Báo cáo Gom cụm – Bank Marketing Preprocessed v4

## 1. Không gian đặc trưng

- Số quan sát: {len(features):,}.
- Số biến trước mã hóa: {features.shape[1]}.
- Loại hoàn toàn khỏi đầu vào: `y`, `housing`, `loan`.
- Dùng `age_capped`, `balance_capped`, `duration_capped`.
- Biến số: Median Imputation + StandardScaler.
- Biến định tính: Mode Imputation + One-Hot Encoding.
- Sau mã hóa: {int(feature_summary.iloc[0]['Encoded_feature_count'])} chiều.
- Truncated SVD: {int(feature_summary.iloc[0]['SVD_component_count'])} chiều, giải thích {feature_summary.iloc[0]['SVD_explained_variance_ratio'] * 100:.2f}% phương sai.

Các nhãn cũ chỉ dùng để mô tả cụm sau huấn luyện:

{chr(10).join(class_distribution_lines)}

## 2. K-Means và lựa chọn K

- Thử K từ 2 đến 10.
- K theo Elbow: **{elbow_k}**.
- K có Silhouette cao nhất toàn miền thử: **{int(best_silhouette_row['K'])}** với Silhouette = **{best_silhouette_row['Silhouette']:.4f}**.
- K được chọn: **{optimal_k}**, chọn trong vùng lân cận Elbow bằng Silhouette cao và Davies–Bouldin thấp.
- K={optimal_k}: Silhouette = **{selected_k_row['Silhouette']:.4f}**, Calinski–Harabasz = **{selected_k_row['Calinski_Harabasz']:.2f}**, Davies–Bouldin = **{selected_k_row['Davies_Bouldin']:.4f}**.

### Hồ sơ định lượng K-Means

```text
{k_numeric_profile[profile_columns].round(3).to_string(index=False)}
```

### Hồ sơ định tính và tỷ lệ nhãn giữ lại để diễn giải

```text
{k_categorical_profile[categorical_columns].round(3).to_string(index=False)}
```

Tỷ lệ `y`, `housing`, `loan` ở trên là phân tích hậu nghiệm, không tham gia tạo cụm.

## 3. Hierarchical Clustering

- Phương pháp: Ward linkage với khoảng cách Euclidean.
- Dùng mẫu cố định {int(hierarchical_row['Sample_size']):,} quan sát vì ma trận khoảng cách phân cấp tăng theo O(n²).
- Số cụm cắt từ cây: {int(hierarchical_row['Actual_cluster_count'])}.
- Silhouette trên mẫu: **{hierarchical_row['Silhouette']:.4f}**.
- Adjusted Rand so với K-Means trên cùng mẫu: **{hierarchical_row['Adjusted_Rand_vs_KMeans']:.4f}**.
- Cophenetic correlation: **{hierarchical_row['Cophenetic_correlation']:.4f}**.

Dendrogram cho thấy thứ tự hợp nhất các nhóm; độ cao trục Y càng lớn nghĩa là hai nhánh được ghép ở khoảng cách càng xa.

## 4. DBSCAN

- `min_samples` = {DBSCAN_MIN_SAMPLES}.
- `eps` được chọn từ k-distance search: **{selected_eps:.4f}**.
- Số cụm không tính Noise: **{dbscan_cluster_count}**.
- Tỷ lệ Noise (`cluster=-1`): **{dbscan_noise_percent:.2f}%**.
- Cụm mật độ lớn nhất chiếm **{largest_dbscan_cluster_percent:.2f}%** toàn bộ dữ liệu.

DBSCAN không bắt buộc mọi điểm thuộc một cụm. Các dòng có nhãn `-1` là khách hàng nằm trong vùng mật độ thấp và được xem là Noise/điểm khác biệt trong không gian đặc trưng.

### Hồ sơ các cụm và Noise

```text
{dbscan_numeric_profile[dbscan_numeric_columns].round(3).to_string(index=False)}
```

```text
{dbscan_categorical_profile[dbscan_categorical_columns].round(3).to_string(index=False)}
```

Nếu một cụm chiếm phần lớn dữ liệu và các cụm còn lại rất nhỏ, DBSCAN nên được
dùng chủ yếu để nhận diện Noise thay vì làm phương án phân khúc khách hàng chính.

### Các cấu hình eps đã thử

```text
{dbscan_search.round(4).to_string(index=False)}
```

## 5. Nhận xét phương pháp

- K-Means phù hợp để tạo phân khúc bao phủ toàn bộ khách hàng và dễ lập hồ sơ cụm.
- Hierarchical giúp quan sát cấu trúc lồng nhau nhưng không thực tế khi chạy linkage đầy đủ trên 45.211 dòng, nên lấy mẫu tái lập bằng `random_state=42`.
- DBSCAN hữu ích để phát hiện Noise, nhưng kết quả nhạy với `eps`, `min_samples` và số chiều.
- One-Hot Encoding + SVD biến dữ liệu hỗn hợp thành không gian Euclidean; đây là xấp xỉ thực dụng. Nếu mục tiêu chủ yếu là dữ liệu phân loại, có thể khảo sát thêm Gower Distance hoặc K-Prototypes.
"""

    (output_dir / "BAO_CAO_CLUSTERING_PREPROCESSED_V4.md").write_text(
        report, encoding="utf-8-sig"
    )


def main() -> None:
    args = parse_arguments()
    workbook_path = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.0)

    print("Đang đọc workbook và kiểm tra dữ liệu...")
    features, held_out_labels, audit = load_and_prepare_data(workbook_path)
    audit.to_csv(
        output_dir / "data_alignment_and_capping_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Đang mã hóa, chuẩn hóa và giảm chiều...")
    reduced, preprocessor, svd, feature_summary = build_feature_space(features)
    feature_summary.to_csv(
        output_dir / "feature_space_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_svd_loadings(preprocessor, svd, output_dir)

    print("Đang đánh giá K và chạy K-Means...")
    kmeans_labels, kmeans_model, k_metrics, optimal_k = run_kmeans(
        reduced, output_dir
    )
    k_numeric_profile, k_categorical_profile = (
        save_kmeans_assignments_and_profiles(
            features,
            held_out_labels,
            reduced,
            kmeans_labels,
            kmeans_model,
            output_dir,
        )
    )
    save_cluster_scatter(
        reduced,
        kmeans_labels,
        f"K-Means – K={optimal_k} (mẫu hiển thị)",
        output_dir / "kmeans_cluster_scatter.png",
    )

    print("Đang chạy Hierarchical và vẽ Dendrogram...")
    hierarchical_summary = run_hierarchical(
        reduced,
        kmeans_labels,
        held_out_labels,
        optimal_k,
        args.hierarchical_sample,
        output_dir,
    )

    print("Đang tìm eps và chạy DBSCAN...")
    (
        dbscan_labels,
        dbscan_search,
        selected_eps,
        dbscan_cluster_count,
        dbscan_noise_percent,
    ) = run_dbscan(reduced, features, held_out_labels, output_dir)
    dbscan_numeric_profile, dbscan_categorical_profile = save_dbscan_profiles(
        features,
        held_out_labels,
        dbscan_labels,
        output_dir,
    )
    save_cluster_scatter(
        reduced,
        dbscan_labels,
        (
            f"DBSCAN – eps={selected_eps:.3f}, "
            f"Noise={dbscan_noise_percent:.1f}% (mẫu hiển thị)"
        ),
        output_dir / "dbscan_cluster_scatter.png",
        noise_label=-1,
    )

    save_report(
        features,
        held_out_labels,
        feature_summary,
        k_metrics,
        optimal_k,
        k_numeric_profile,
        k_categorical_profile,
        hierarchical_summary,
        dbscan_search,
        selected_eps,
        dbscan_cluster_count,
        dbscan_noise_percent,
        dbscan_numeric_profile,
        dbscan_categorical_profile,
        output_dir,
    )

    print("\n" + "=" * 78)
    print("HOÀN THÀNH GOM CỤM")
    print(f"K-Means: K={optimal_k}")
    print(
        f"DBSCAN: eps={selected_eps:.4f}, cụm={dbscan_cluster_count}, "
        f"Noise={dbscan_noise_percent:.2f}%"
    )
    print(f"Kết quả: {output_dir}")
    print("Báo cáo: BAO_CAO_CLUSTERING_PREPROCESSED_V4.md")
    print("=" * 78)


if __name__ == "__main__":
    main()
