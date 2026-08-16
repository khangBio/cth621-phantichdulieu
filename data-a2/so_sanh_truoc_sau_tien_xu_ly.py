from pathlib import Path
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


DATA = Path("bank/bank-full.csv")
OUT = Path("visualizations_before_after")
OUT.mkdir(exist_ok=True)

NUM = ["age", "balance", "duration"]
COLUMNS = NUM + ["job", "y"]
LABEL = {
    "age": "Tuổi (năm)",
    "balance": "Số dư tài khoản",
    "duration": "Thời lượng cuộc gọi (giây)",
    "job": "Nghề nghiệp",
    "y": "Đăng ký tiền gửi kỳ hạn",
}
BEFORE = "#4C78A8"
AFTER = "#F58518"
Y_COLORS = {"no": "#4C78A8", "yes": "#F58518"}

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})


def save(fig, filename):
    fig.savefig(OUT / filename, facecolor="white")
    plt.close(fig)


def fences(series):
    q1, q3 = series.quantile([.25, .75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def outlier_mask(series, lo=None, hi=None):
    if lo is None or hi is None:
        lo, hi = fences(series.dropna())
    return (series < lo) | (series > hi)


# -----------------------------------------------------------------------------
# 1) Raw audit and preprocessing
# -----------------------------------------------------------------------------
source = pd.read_csv(DATA, sep=";")
raw = source[COLUMNS].copy()
raw.insert(0, "source_row", np.arange(2, len(raw) + 2))  # physical CSV line number

true_missing_before = raw[COLUMNS].isna().sum()
semantic_unknown_before = pd.Series(0, index=COLUMNS, dtype=int)
semantic_unknown_before["job"] = raw["job"].eq("unknown").sum()

processed = raw.copy()
processed["job"] = processed["job"].replace("unknown", np.nan)

# Median for numeric predictors; mode for categorical predictors.
imputation_values = {}
for col in NUM:
    imputation_values[col] = processed[col].median()
    processed[col] = processed[col].fillna(imputation_values[col])
for col in ["job"]:
    imputation_values[col] = processed[col].mode(dropna=True).iloc[0]
    processed[col] = processed[col].fillna(imputation_values[col])

# A missing target should normally be dropped, not inferred. There are none here.
missing_target_rows_removed = int(processed["y"].isna().sum())
processed = processed.dropna(subset=["y"]).copy()

# IQR capping keeps rows while limiting extreme leverage.
cap_info = {}
for col in NUM:
    lo, hi = fences(processed[col])
    mask = outlier_mask(processed[col], lo, hi)
    cap_info[col] = {
        "lower_fence": lo,
        "upper_fence": hi,
        "capped_count": int(mask.sum()),
        "capped_low": int((processed[col] < lo).sum()),
        "capped_high": int((processed[col] > hi).sum()),
    }
    processed[col] = processed[col].clip(lo, hi)

# Add both standard scaling variants; main columns remain in interpretable units.
for col in NUM:
    mean, std = processed[col].mean(), processed[col].std(ddof=0)
    mn, mx = processed[col].min(), processed[col].max()
    processed[f"{col}_z"] = (processed[col] - mean) / std
    processed[f"{col}_minmax"] = (processed[col] - mn) / (mx - mn)

true_missing_after = processed[COLUMNS].isna().sum()


def stats_table(frame, state):
    records = []
    for col in NUM:
        s = frame[col]
        lo, hi = fences(s.dropna())
        records.append({
            "state": state,
            "variable": col,
            "count": s.count(),
            "missing": s.isna().sum(),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "q1": s.quantile(.25),
            "q3": s.quantile(.75),
            "max": s.max(),
            "skewness": s.skew(),
            "iqr_lower_fence": lo,
            "iqr_upper_fence": hi,
            "outlier_count": int(outlier_mask(s, lo, hi).sum()),
        })
    return pd.DataFrame(records)


stats = pd.concat([
    stats_table(raw, "Trước xử lý"),
    stats_table(processed, "Sau xử lý"),
], ignore_index=True)
stats.to_csv(OUT / "THONG_KE_TRUOC_SAU.csv", index=False, encoding="utf-8-sig")

missing_audit = pd.DataFrame({
    "variable": COLUMNS,
    "true_null_before": [int(true_missing_before[c]) for c in COLUMNS],
    "semantic_unknown_before": [int(semantic_unknown_before[c]) for c in COLUMNS],
    "null_after": [int(true_missing_after[c]) for c in COLUMNS],
})
missing_audit.to_csv(OUT / "KIEM_KE_KHUYET_THIEU.csv", index=False, encoding="utf-8-sig")

cap_df = pd.DataFrame(cap_info).T.reset_index(names="variable")
cap_df.to_csv(OUT / "NGUONG_VA_SO_LUONG_IQR_CAPPING.csv", index=False, encoding="utf-8-sig")

# Concrete outlier examples: CSV line, value, fence and context.
examples = []
for col in NUM:
    lo, hi = fences(raw[col])
    candidates = raw.loc[outlier_mask(raw[col], lo, hi), ["source_row", col, "job", "y"]].copy()
    low = candidates[candidates[col] < lo].nsmallest(5, col)
    high = candidates[candidates[col] > hi].nlargest(5, col)
    for direction, subset in [("low", low), ("high", high)]:
        for _, row in subset.iterrows():
            examples.append({
                "variable": col,
                "direction": direction,
                "source_csv_line": int(row["source_row"]),
                "value": row[col],
                "lower_fence": lo,
                "upper_fence": hi,
                "job": row["job"],
                "y": row["y"],
            })
pd.DataFrame(examples).to_csv(OUT / "VI_DU_OUTLIER_CU_THE.csv", index=False, encoding="utf-8-sig")

processed.to_csv(OUT / "bank_5vars_processed.csv", index=False, encoding="utf-8-sig")

# -----------------------------------------------------------------------------
# 2) Visual comparisons
# -----------------------------------------------------------------------------

# 01. Missingness audit (true NaN + semantic unknown).
x = np.arange(len(COLUMNS))
fig, ax = plt.subplots(figsize=(11, 6))
w = .25
ax.bar(x-w, true_missing_before.values, w, color=BEFORE, label="Null/NaN thật – trước")
ax.bar(x, semantic_unknown_before.values, w, color="#E45756", label="'unknown' ngữ nghĩa – trước")
ax.bar(x+w, true_missing_after.values, w, color=AFTER, label="Null/NaN – sau")
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=9)
ax.set(title="Kiểm kê khuyết thiếu trước và sau xử lý",
       xlabel="Biến", ylabel="Số ô khuyết thiếu / unknown", xticks=x, xticklabels=COLUMNS)
ax.legend()
ax.set_ylim(0, max(330, semantic_unknown_before.max()*1.2))
fig.tight_layout()
save(fig, "01_missingness_before_after.png")

# 02. Histograms, two rows.
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for j, col in enumerate(NUM):
    sns.histplot(raw[col], bins=50, kde=True, color=BEFORE, ax=axes[0, j])
    axes[0, j].axvline(raw[col].median(), color="#E45756", ls="--", label="Median")
    axes[0, j].set(title=f"TRƯỚC: {col}", xlabel=LABEL[col], ylabel="Số quan sát")
    axes[0, j].legend()
    sns.histplot(processed[col], bins=50, kde=True, color=AFTER, ax=axes[1, j])
    axes[1, j].axvline(processed[col].median(), color="#E45756", ls="--", label="Median")
    axes[1, j].set(title=f"SAU: {col} (đã IQR capping)", xlabel=LABEL[col], ylabel="Số quan sát")
    axes[1, j].legend()
fig.suptitle("Histogram trước–sau: hình dạng phân phối và đuôi cực trị", fontsize=17)
fig.tight_layout()
save(fig, "02_histogram_before_after.png")

# 03. Boxplots with exact outlier counts and extreme labels.
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for j, col in enumerate(NUM):
    lo, hi = fences(raw[col])
    count = int(outlier_mask(raw[col], lo, hi).sum())
    sns.boxplot(x=raw[col], color=BEFORE, fliersize=2, ax=axes[0, j])
    axes[0, j].set(title=f"TRƯỚC: {count:,} outlier IQR\nmin={raw[col].min():,.0f}; max={raw[col].max():,.0f}",
                   xlabel=LABEL[col], ylabel="")
    axes[0, j].annotate(f"max {raw[col].max():,.0f}", xy=(raw[col].max(), 0), xytext=(-80, 25),
                        textcoords="offset points", arrowprops=dict(arrowstyle="->", color="#E45756"), fontsize=9)
    lo2, hi2 = fences(processed[col])
    count2 = int(outlier_mask(processed[col], lo2, hi2).sum())
    sns.boxplot(x=processed[col], color=AFTER, fliersize=2, ax=axes[1, j])
    axes[1, j].set(title=f"SAU: {count2:,} outlier IQR\nmin={processed[col].min():,.1f}; max={processed[col].max():,.1f}",
                   xlabel=LABEL[col], ylabel="")
fig.suptitle("Boxplot trước–sau: outlier, trung vị và khoảng tứ phân vị", fontsize=17)
fig.tight_layout()
save(fig, "03_boxplot_before_after.png")

# 04. Sorted line charts show extreme jumps and cap plateaus.
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, col in zip(axes, NUM):
    before_sorted = np.sort(raw[col].dropna().to_numpy())
    after_sorted = np.sort(processed[col].dropna().to_numpy())
    ax.plot(before_sorted, color=BEFORE, lw=1.2, label="Trước")
    ax.plot(after_sorted, color=AFTER, lw=1.5, label="Sau IQR capping")
    lo, hi = fences(raw[col])
    ax.axhline(hi, color="#E45756", ls="--", lw=1, label="Hàng rào IQR trên")
    if lo >= 0 or col == "balance":
        ax.axhline(lo, color="#54A24B", ls=":", lw=1, label="Hàng rào IQR dưới")
    if col == "balance":
        ax.set_yscale("symlog", linthresh=1000)
    ax.set(title=f"Giá trị {col} theo thứ hạng", xlabel="Thứ hạng sau khi sắp tăng", ylabel=LABEL[col])
    ax.legend(fontsize=8)
fig.suptitle("Line chart: đoạn tăng gãy ở đuôi và hiệu ứng làm mịn cực trị", fontsize=17)
fig.tight_layout()
save(fig, "04_sorted_line_outlier_jumps.png")

# 05. Pairwise scatter before/after with the same stratified row sample.
rng = np.random.default_rng(42)
sample_idx = np.concatenate([
    rng.choice(raw.index[raw["y"] == cls], size=min(3000, (raw["y"] == cls).sum()), replace=False)
    for cls in ["no", "yes"]
])
pairs = [("age", "balance"), ("age", "duration"), ("balance", "duration")]
fig, axes = plt.subplots(3, 2, figsize=(14, 18))
for i, (xcol, ycol) in enumerate(pairs):
    for j, (frame, state, color_title) in enumerate([(raw, "TRƯỚC", BEFORE), (processed, "SAU", AFTER)]):
        plot = frame.loc[sample_idx]
        sns.scatterplot(data=plot, x=xcol, y=ycol, hue="y", palette=Y_COLORS,
                        alpha=.32, s=14, linewidth=0, ax=axes[i, j])
        axes[i, j].set(title=f"{state}: {xcol} – {ycol}", xlabel=LABEL[xcol], ylabel=LABEL[ycol])
        axes[i, j].legend(title="y", fontsize=8)
fig.suptitle("Scatter plot các cặp biến định lượng: trước và sau xử lý", fontsize=17)
fig.tight_layout()
save(fig, "05_scatter_pairs_before_after.png")

# 06. Correlation matrices before/after.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, frame, title in [(axes[0], raw, "TRƯỚC"), (axes[1], processed, "SAU")]:
    temp = frame[NUM].copy()
    temp["y_num"] = frame["y"].eq("yes").astype(int)
    sns.heatmap(temp.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"{title}: tương quan Pearson")
    ax.set_xticklabels(NUM + ["y (0/1)"], rotation=30)
    ax.set_yticklabels(NUM + ["y (0/1)"], rotation=0)
fig.suptitle("Tương quan trước–sau IQR capping", fontsize=17)
fig.tight_layout()
save(fig, "06_correlation_before_after.png")

# 07. Job frequency bars.
job_before = raw["job"].value_counts()
job_after = processed["job"].value_counts()
jobs = sorted(set(job_before.index) | set(job_after.index))
comparison = pd.DataFrame({"Trước": job_before.reindex(jobs, fill_value=0),
                           "Sau": job_after.reindex(jobs, fill_value=0)}).sort_values("Trước")
fig, ax = plt.subplots(figsize=(12, 8))
ypos = np.arange(len(comparison))
ax.barh(ypos-.19, comparison["Trước"], height=.38, color=BEFORE, label="Trước")
ax.barh(ypos+.19, comparison["Sau"], height=.38, color=AFTER, label="Sau")
ax.set(title="Bar chart job trước–sau điền 'unknown' bằng mode",
       xlabel="Số khách hàng", ylabel="Nghề nghiệp", yticks=ypos, yticklabels=comparison.index)
ax.legend()
fig.tight_layout()
save(fig, "07_job_bar_before_after.png")

# 08. Job doughnut before/after.
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, counts, title in [(axes[0], job_before, "TRƯỚC"), (axes[1], job_after, "SAU")]:
    counts = counts.sort_values(ascending=False)
    wedges, _ = ax.pie(counts.values, startangle=90, wedgeprops=dict(width=.42, edgecolor="white"))
    ax.legend(wedges, [f"{k}: {v/len(raw):.1%}" for k, v in counts.items()],
              loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
    ax.set_title(f"{title}: tỷ trọng job")
fig.suptitle("Doughnut chart cơ cấu nghề nghiệp trước–sau", fontsize=17)
fig.tight_layout()
save(fig, "08_job_doughnut_before_after.png")

# 09. Target y is deliberately unchanged.
y_before = raw["y"].value_counts().reindex(["no", "yes"])
y_after = processed["y"].value_counts().reindex(["no", "yes"])
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, counts, title in [(axes[0], y_before, "TRƯỚC"), (axes[1], y_after, "SAU")]:
    bars = ax.bar(counts.index, counts.values, color=[Y_COLORS[k] for k in counts.index])
    ax.bar_label(bars, labels=[f"{v:,}\n({v/counts.sum():.1%})" for v in counts.values], padding=3)
    ax.set(title=f"{title}: biến mục tiêu y", xlabel="y", ylabel="Số khách hàng",
           ylim=(0, counts.max()*1.12))
fig.suptitle("Biến mục tiêu được bảo toàn sau tiền xử lý", fontsize=17)
fig.tight_layout()
save(fig, "09_y_before_after_unchanged.png")

# 10. Job response rates before/after.
rate_before = raw.assign(y_num=raw["y"].eq("yes")).groupby("job")["y_num"].mean()
rate_after = processed.assign(y_num=processed["y"].eq("yes")).groupby("job")["y_num"].mean()
rate_jobs = sorted(set(rate_before.index) | set(rate_after.index))
fig, ax = plt.subplots(figsize=(12, 7))
xpos = np.arange(len(rate_jobs))
ax.plot(xpos, rate_before.reindex(rate_jobs), color=BEFORE, marker="o", label="Trước")
ax.plot(xpos, rate_after.reindex(rate_jobs), color=AFTER, marker="o", label="Sau")
ax.set(title="Tỷ lệ y=yes theo job trước–sau",
       xlabel="Nghề nghiệp", ylabel="Tỷ lệ y=yes", xticks=xpos, xticklabels=rate_jobs)
ax.tick_params(axis="x", rotation=40)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.legend()
fig.tight_layout()
save(fig, "10_job_yes_rate_before_after.png")

# 11. Standard Z-score distributions.
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, col in zip(axes, NUM):
    zcol = f"{col}_z"
    sns.histplot(processed[zcol], bins=45, kde=True, color=AFTER, ax=ax)
    ax.axvline(0, color="#E45756", ls="--", label="Mean = 0")
    ax.set(title=f"Z-score của {col}", xlabel=f"{col}_z (đơn vị độ lệch chuẩn)", ylabel="Số quan sát")
    ax.legend()
fig.suptitle("Standardization sau xử lý: cùng tâm 0 và độ lệch chuẩn 1", fontsize=17)
fig.tight_layout()
save(fig, "11_zscore_distributions_after.png")

# 12. Min-Max scaling boxplots.
fig, ax = plt.subplots(figsize=(10, 6))
mm_long = processed[[f"{c}_minmax" for c in NUM]].rename(
    columns={f"{c}_minmax": c for c in NUM}).melt(var_name="variable", value_name="minmax")
sns.boxplot(data=mm_long, x="variable", y="minmax", hue="variable", palette="Set2", legend=False, ax=ax)
ax.set(title="Min–Max Scaling sau xử lý", xlabel="Biến", ylabel="Giá trị chuẩn hóa [0, 1]", ylim=(-.04, 1.04))
fig.tight_layout()
save(fig, "12_minmax_boxplots_after.png")

# 13. Statistical quality dashboard: std, skew, outliers.
metric_data = stats.pivot(index="variable", columns="state", values=["std", "skewness", "outlier_count"])
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, metric, title in zip(axes, ["std", "skewness", "outlier_count"],
                             ["Độ lệch chuẩn", "Độ lệch (skewness)", "Số outlier theo IQR mới"]):
    vals = metric_data[metric].reindex(NUM)
    vals.plot(kind="bar", ax=ax, color=[AFTER, BEFORE] if list(vals.columns)[0] == "Sau xử lý" else [BEFORE, AFTER])
    ax.set(title=title, xlabel="Biến", ylabel=title)
    if metric == "std":
        ax.set_yscale("symlog", linthresh=10)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f" if metric == "outlier_count" else "%.1f",
                     padding=2, fontsize=7)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Trạng thái", fontsize=8)
fig.suptitle("So sánh chỉ số thống kê trước–sau", fontsize=17)
fig.tight_layout()
save(fig, "13_statistical_metrics_before_after.png")

# 14. ECDF comparison is robust to histogram bin choices.
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, col in zip(axes, NUM):
    sns.ecdfplot(raw[col], color=BEFORE, label="Trước", ax=ax)
    sns.ecdfplot(processed[col], color=AFTER, label="Sau", ax=ax)
    ax.set(title=f"ECDF {col}", xlabel=LABEL[col], ylabel="Tỷ lệ tích lũy")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
fig.suptitle("ECDF trước–sau: phần trung tâm được giữ, đuôi được chặn", fontsize=17)
fig.tight_layout()
save(fig, "14_ecdf_before_after.png")

# 15. Preserve the analytical y relationship while capping duration extremes.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
sns.boxplot(data=raw, x="y", y="duration", hue="y", palette=Y_COLORS, legend=False, ax=axes[0], fliersize=2)
axes[0].set(title="TRƯỚC: duration theo y", xlabel="y", ylabel=LABEL["duration"])
sns.boxplot(data=processed, x="y", y="duration", hue="y", palette=Y_COLORS, legend=False, ax=axes[1], fliersize=2)
axes[1].set(title="SAU: duration theo y", xlabel="y", ylabel=LABEL["duration"])
fig.suptitle("Tín hiệu duration–y trước và sau xử lý outlier", fontsize=17)
fig.tight_layout()
save(fig, "15_duration_by_y_before_after.png")

# -----------------------------------------------------------------------------
# 3) Data-backed report
# -----------------------------------------------------------------------------
before_stats = stats[stats["state"] == "Trước xử lý"].set_index("variable")
after_stats = stats[stats["state"] == "Sau xử lý"].set_index("variable")

rows = []
for col in NUM:
    b, a = before_stats.loc[col], after_stats.loc[col]
    rows.append(
        f"| {col} | {b['mean']:.2f} → {a['mean']:.2f} | {b['median']:.2f} → {a['median']:.2f} | "
        f"{b['std']:.2f} → {a['std']:.2f} ({(a['std']/b['std']-1):+.1%}) | "
        f"{b['skewness']:.2f} → {a['skewness']:.2f} | {int(b['outlier_count']):,} → {int(a['outlier_count']):,} |"
    )

report = f"""# So sánh dữ liệu trước và sau tiền xử lý

**Nguồn:** `bank/bank-full.csv` — {len(raw):,} dòng.  
**Phạm vi:** age, balance, duration, job và y.  
**Nguyên tắc:** dữ liệu gốc không bị ghi đè; bản sau xử lý nằm ở `bank_5vars_processed.csv`.

## Quy trình đã áp dụng

1. Kiểm kê Null/NaN thật: cả 5 biến đều có **0 ô Null/NaN**.
2. Nhận diện khuyết thiếu ngữ nghĩa: `job='unknown'` có **{semantic_unknown_before['job']:,} ô**; chuyển thành NaN và điền bằng mode `{imputation_values['job']}`.
3. Nếu biến số có khuyết, dùng median: age={imputation_values['age']:.0f}, balance={imputation_values['balance']:.0f}, duration={imputation_values['duration']:.0f}. Trong dữ liệu này không có ô số nào cần điền.
4. Xử lý outlier bằng IQR capping: chặn tại [Q1−1.5×IQR, Q3+1.5×IQR], không xóa dòng.
5. Tạo thêm hai bộ đặc trưng: Z-score (mean≈0, std≈1) và Min–Max ([0,1]).
6. `y` được giữ nguyên; số dòng mục tiêu bị loại do thiếu = {missing_target_rows_removed}.

## Thay đổi thống kê

| Biến | Mean trước → sau | Median trước → sau | Std trước → sau | Skewness trước → sau | Outlier IQR trước → sau |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

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
- **Nhận xét:** trước xử lý có age={int(before_stats.loc['age','outlier_count']):,}, balance={int(before_stats.loc['balance','outlier_count']):,}, duration={int(before_stats.loc['duration','outlier_count']):,} outlier. Giá trị cực đại được ghi trực tiếp trên hình; danh sách dòng cụ thể nằm trong `VI_DU_OUTLIER_CU_THE.csv`.

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
- **Nhận xét:** 288 `unknown` được chuyển vào mode `{imputation_values['job']}`, do đó thanh unknown biến mất và thanh `{imputation_values['job']}` tăng tương ứng. Đây là thay đổi có chủ đích nhưng có thể làm nhóm mode trội hơn.

### 08 — Doughnut job
- **Lát:** tỷ trọng từng job; không có trục tọa độ.
- **Nhận xét:** cơ cấu tổng thể hầu như giữ nguyên vì unknown chỉ chiếm {semantic_unknown_before['job']/len(raw):.2%}; khác biệt tập trung ở nhóm mode.

### 09 — Biến mục tiêu y
- **Trục X:** no/yes. **Trục Y:** số khách hàng.
- **Nhận xét:** số lượng và tỷ trọng y giống hệt trước–sau. Đây là kiểm tra quan trọng để bảo đảm tiền xử lý predictors không làm méo nhãn.

### 10 — Tỷ lệ yes theo job
- **Trục X:** job. **Trục Y:** tỷ lệ y=yes.
- **Nhận xét:** các job có tên xác định giữ nguyên tỷ lệ; `{imputation_values['job']}` thay đổi nhẹ vì nhận thêm 288 dòng unknown. Đường của unknown sau xử lý bị khuyết vì nhóm này không còn tồn tại.

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
"""

(OUT / "BAO_CAO_SO_SANH_TRUOC_SAU.md").write_text(report, encoding="utf-8")

files = sorted(p.name for p in OUT.glob("*.png"))
pd.DataFrame({"file": files}).to_csv(OUT / "DANH_MUC_BIEU_DO.csv", index=False, encoding="utf-8-sig")

print(f"Created {len(files)} comparison charts in {OUT.resolve()}")
print(f"Rows before/after: {len(raw)}/{len(processed)}")
