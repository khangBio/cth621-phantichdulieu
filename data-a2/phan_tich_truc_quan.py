from pathlib import Path
import math
import os
import textwrap

import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter, FuncFormatter


DATA_PATH = Path("bank/bank-full.csv")
OUT = Path("visualizations_bank")
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

COLORS = {"no": "#4C78A8", "yes": "#F58518"}
NUM = ["age", "balance", "duration"]
LABEL = {
    "age": "Tuổi (năm)",
    "balance": "Số dư tài khoản (đơn vị dữ liệu)",
    "duration": "Thời lượng cuộc gọi (giây)",
}

df = pd.read_csv(DATA_PATH, sep=";")
df = df[["age", "balance", "duration", "job", "y"]].copy()
df["y_num"] = (df["y"] == "yes").astype(int)


def pct(x):
    return f"{100*x:.1f}%"


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)


def annotate_bars(ax, values, fmt="count"):
    maxv = max(values) if len(values) else 1
    for patch, val in zip(ax.patches, values):
        if fmt == "pct":
            txt = f"{val:.1%}"
        else:
            txt = f"{int(val):,}"
        ax.text(patch.get_width() + maxv * 0.012,
                patch.get_y() + patch.get_height()/2,
                txt, va="center", fontsize=9)


# Core statistics used both in plots and in the written interpretation.
desc = df[NUM].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
for col in NUM:
    q1, q3 = df[col].quantile([.25, .75])
    iqr = q3 - q1
    desc.loc[col, "iqr"] = iqr
    desc.loc[col, "lower_fence"] = q1 - 1.5 * iqr
    desc.loc[col, "upper_fence"] = q3 + 1.5 * iqr
    desc.loc[col, "outlier_count"] = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
    desc.loc[col, "skewness"] = df[col].skew()
desc.to_csv(OUT / "thong_ke_mo_ta.csv", encoding="utf-8-sig")

# 01. Histograms: full range plus a central 98% view where tails are extreme.
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, col in zip(axes, NUM):
    lo, hi = df[col].quantile([.01, .99])
    view = df[col].clip(lo, hi) if col in ["balance", "duration"] else df[col]
    sns.histplot(view, bins=40, kde=True, color="#4C78A8", ax=ax)
    ax.axvline(df[col].median(), color="#E45756", linestyle="--", label=f"Trung vị: {df[col].median():,.0f}")
    ax.axvline(df[col].mean(), color="#54A24B", linestyle=":", label=f"Trung bình: {df[col].mean():,.1f}")
    ax.set(title=f"Phân phối {col}" + (" (co biên 1%–99%)" if col != "age" else ""),
           xlabel=LABEL[col], ylabel="Số quan sát")
    ax.legend(fontsize=8)
fig.suptitle("Histogram và đường mật độ của các biến định lượng", fontsize=16)
fig.tight_layout()
save(fig, "01_histogram_age_balance_duration.png")

# 02. Individual full-range histograms on log-scaled count axis to preserve rare tails.
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, col in zip(axes, NUM):
    sns.histplot(df[col], bins=60, color="#72B7B2", ax=ax)
    ax.set_yscale("log")
    ax.set(title=f"Toàn miền {col} – thang log ở trục Y", xlabel=LABEL[col], ylabel="Số quan sát (log)")
fig.suptitle("Histogram toàn miền: nhìn rõ cả phần đuôi hiếm", fontsize=16)
fig.tight_layout()
save(fig, "02_histogram_full_range_log_count.png")

# 03. Boxplots in separate panels, preserving true values.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col in zip(axes, NUM):
    sns.boxplot(x=df[col], color="#A0CBE8", ax=ax, fliersize=2)
    q1, med, q3 = df[col].quantile([.25, .5, .75])
    ax.set(title=f"{col}: Q1={q1:,.0f}, Median={med:,.0f}, Q3={q3:,.0f}", xlabel=LABEL[col], ylabel="")
fig.suptitle("Boxplot: tứ phân vị, độ phân tán và điểm dị biệt", fontsize=16)
fig.tight_layout()
save(fig, "03_boxplot_numeric_full_range.png")

# 04. Boxplots by target. Balance and duration are clipped only for legibility.
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, col in zip(axes, NUM):
    tmp = df[[col, "y"]].copy()
    if col in ["balance", "duration"]:
        lo, hi = tmp[col].quantile([.01, .99])
        tmp[col] = tmp[col].clip(lo, hi)
    sns.boxplot(data=tmp, x="y", y=col, hue="y", palette=COLORS, legend=False, ax=ax)
    ax.set(title=f"{col} theo y" + (" (co biên 1%–99%)" if col != "age" else ""),
           xlabel="Kết quả đăng ký tiền gửi kỳ hạn (y)", ylabel=LABEL[col])
fig.suptitle("So sánh phân phối biến định lượng giữa hai lớp mục tiêu", fontsize=16)
fig.tight_layout()
save(fig, "04_boxplot_numeric_by_y.png")

# Reproducible stratified sample for scatter plots (keeps image size manageable).
sample = (df.groupby("y", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), 5000), random_state=42), include_groups=False)
            .reset_index(drop=True))
sample["y"] = np.where(sample["y_num"] == 1, "yes", "no")

# 05–07. Pairwise quantitative scatter plots.
pairs = [("age", "balance"), ("age", "duration"), ("balance", "duration")]
for idx, (x, y) in enumerate(pairs, start=5):
    plot_df = sample.copy()
    if x in ["balance", "duration"]:
        plot_df[x] = plot_df[x].clip(df[x].quantile(.01), df[x].quantile(.99))
    if y in ["balance", "duration"]:
        plot_df[y] = plot_df[y].clip(df[y].quantile(.01), df[y].quantile(.99))
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x=x, y=y, hue="y", palette=COLORS,
                    alpha=.35, s=18, linewidth=0, ax=ax)
    ax.set(title=f"Scatter plot: {x} và {y} (mẫu phân tầng, co biên 1%–99% nếu cần)",
           xlabel=LABEL[x], ylabel=LABEL[y])
    ax.legend(title="y")
    fig.tight_layout()
    save(fig, f"{idx:02d}_scatter_{x}_{y}_by_y.png")

# 08. Pearson and Spearman correlation heatmaps.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, method in zip(axes, ["pearson", "spearman"]):
    corr = df[NUM + ["y_num"]].corr(method=method)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Tương quan {method.title()}")
    ax.set_xticklabels(["age", "balance", "duration", "y (0/1)"], rotation=30)
    ax.set_yticklabels(["age", "balance", "duration", "y (0/1)"], rotation=0)
fig.suptitle("Ma trận tương quan: tuyến tính và đơn điệu", fontsize=16)
fig.tight_layout()
save(fig, "08_correlation_heatmaps.png")

# 09. Job frequency bar chart.
job_count = df["job"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(job_count.index, job_count.values, color="#4C78A8")
annotate_bars(ax, job_count.values)
ax.set(title="Cơ cấu nghề nghiệp theo số lượng", xlabel="Số khách hàng", ylabel="Nghề nghiệp (job)")
ax.set_xlim(0, job_count.max()*1.14)
fig.tight_layout()
save(fig, "09_job_bar_count.png")

# 10. Job share doughnut chart.
fig, ax = plt.subplots(figsize=(10, 8))
wedges, _ = ax.pie(job_count.sort_values(ascending=False).values,
                   startangle=90, wedgeprops=dict(width=.42, edgecolor="white"))
ax.legend(wedges,
          [f"{j}: {n/len(df):.1%}" for j, n in job_count.sort_values(ascending=False).items()],
          title="job – tỷ trọng", loc="center left", bbox_to_anchor=(1, .5), fontsize=9)
ax.set_title("Doughnut chart: tỷ trọng nghề nghiệp")
fig.tight_layout()
save(fig, "10_job_doughnut_share.png")

# 11. Target count and share.
y_count = df["y"].value_counts().reindex(["no", "yes"])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(y_count.index, y_count.values, color=[COLORS[k] for k in y_count.index])
for p, v in zip(axes[0].patches, y_count.values):
    axes[0].text(p.get_x()+p.get_width()/2, v+500, f"{v:,}\n({v/len(df):.1%})", ha="center")
axes[0].set(title="Số lượng theo biến mục tiêu y", xlabel="y", ylabel="Số khách hàng", ylim=(0, y_count.max()*1.12))
axes[1].pie(y_count.values, labels=y_count.index, autopct="%1.1f%%", startangle=90,
            colors=[COLORS[k] for k in y_count.index], wedgeprops=dict(edgecolor="white"))
axes[1].set_title("Tỷ trọng hai lớp y")
fig.suptitle("Bar chart và pie chart cho biến mục tiêu", fontsize=16)
fig.tight_layout()
save(fig, "11_y_bar_and_pie.png")

# 12. 100% stacked composition of y within each job.
job_y = pd.crosstab(df["job"], df["y"], normalize="index")[["no", "yes"]]
job_y = job_y.sort_values("yes")
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(job_y.index, job_y["no"], color=COLORS["no"], label="no")
ax.barh(job_y.index, job_y["yes"], left=job_y["no"], color=COLORS["yes"], label="yes")
for i, rate in enumerate(job_y["yes"]):
    ax.text(1.005, i, f"{rate:.1%}", va="center", fontsize=9)
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.set(title="Cơ cấu y trong từng nhóm nghề nghiệp", xlabel="Tỷ trọng trong từng job", ylabel="Nghề nghiệp (job)")
ax.legend(title="y", loc="lower center", bbox_to_anchor=(.5, -0.19), ncol=2)
ax.set_xlim(0, 1.08)
fig.subplots_adjust(bottom=.22)
fig.tight_layout()
save(fig, "12_job_y_100pct_stacked.png")

# 13. Job conversion rate and Wilson 95% intervals.
job_stats = df.groupby("job")["y_num"].agg(["mean", "sum", "count"])
z = 1.96
p = job_stats["mean"]
n = job_stats["count"]
center = (p + z*z/(2*n))/(1+z*z/n)
half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
job_stats["lo"], job_stats["hi"] = center-half, center+half
job_stats = job_stats.sort_values("mean")
fig, ax = plt.subplots(figsize=(11, 7))
ax.errorbar(job_stats["mean"], job_stats.index,
            xerr=[job_stats["mean"]-job_stats["lo"], job_stats["hi"]-job_stats["mean"]],
            fmt="o", color="#F58518", ecolor="#9D755D", capsize=3)
for i, rate in enumerate(job_stats["mean"]):
    ax.text(rate+.005, i, f"{rate:.1%}", va="center", fontsize=9)
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.set(title="Tỷ lệ đăng ký 'yes' theo nghề nghiệp (kèm khoảng tin cậy Wilson 95%)",
       xlabel="Tỷ lệ y = yes", ylabel="Nghề nghiệp (job)")
fig.tight_layout()
save(fig, "13_job_yes_rate_ci.png")

# 14. Violin distributions by y.
long = df.melt(id_vars="y", value_vars=NUM, var_name="variable", value_name="value")
long["value_plot"] = long["value"].astype(float)
long.loc[long["variable"] == "balance", "value_plot"] = np.sign(long.loc[long["variable"] == "balance", "value"]) * np.log1p(np.abs(long.loc[long["variable"] == "balance", "value"]))
long.loc[long["variable"] == "duration", "value_plot"] = np.log1p(long.loc[long["variable"] == "duration", "value"])
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, col in zip(axes, NUM):
    tmp = long[long["variable"] == col]
    sns.violinplot(data=tmp, x="y", y="value_plot", hue="y", palette=COLORS,
                   inner="quartile", cut=0, legend=False, ax=ax)
    ylabel = LABEL[col]
    if col == "balance": ylabel = "signed log1p(balance)"
    if col == "duration": ylabel = "log1p(duration)"
    ax.set(title=f"Violin: {col} theo y", xlabel="Biến mục tiêu y", ylabel=ylabel)
fig.suptitle("Violin plot: hình dạng mật độ và tứ phân vị theo lớp y", fontsize=16)
fig.tight_layout()
save(fig, "14_violin_numeric_by_y.png")

# 15. Empirical cumulative distributions by y.
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, col in zip(axes, NUM):
    tmp = df[[col, "y"]].copy()
    if col in ["balance", "duration"]:
        tmp = tmp[tmp[col].between(tmp[col].quantile(.01), tmp[col].quantile(.99))]
    sns.ecdfplot(data=tmp, x=col, hue="y", palette=COLORS, ax=ax)
    ax.set(title=f"ECDF của {col}", xlabel=LABEL[col], ylabel="Tỷ lệ tích lũy")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
fig.suptitle("ECDF: so sánh toàn bộ phân phối giữa y=no và y=yes", fontsize=16)
fig.tight_layout()
save(fig, "15_ecdf_numeric_by_y.png")

# 16. Hexbin reveals density hidden by overplotting.
hex_df = df[df["balance"].between(df["balance"].quantile(.01), df["balance"].quantile(.99)) &
            df["duration"].between(df["duration"].quantile(.01), df["duration"].quantile(.99))]
fig, ax = plt.subplots(figsize=(10, 7))
hb = ax.hexbin(hex_df["balance"], hex_df["duration"], gridsize=45, bins="log", cmap="viridis", mincnt=1)
fig.colorbar(hb, ax=ax, label="Mật độ quan sát (log)")
ax.set(title="Hexbin: mật độ đồng thời của balance và duration (miền 1%–99%)",
       xlabel=LABEL["balance"], ylabel=LABEL["duration"])
fig.tight_layout()
save(fig, "16_hexbin_balance_duration.png")

# 17–19. Binned response curves: count + yes rate, with equal-frequency bins.
for idx, col in enumerate(NUM, start=17):
    if col == "age":
        bins = pd.cut(df[col], bins=[17, 29, 39, 49, 59, 69, 100], right=True)
    else:
        bins = pd.qcut(df[col], q=10, duplicates="drop")
    stats = df.assign(bin=bins).groupby("bin", observed=True)["y_num"].agg(["mean", "count"])
    labels = [str(x) for x in stats.index]
    x = np.arange(len(stats))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(x, stats["count"], color="#B9D7EA", label="Số quan sát")
    ax1.set(xlabel=f"Nhóm {col}", ylabel="Số quan sát", xticks=x)
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, stats["mean"], color="#E45756", marker="o", linewidth=2, label="Tỷ lệ yes")
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.set_ylabel("Tỷ lệ y = yes")
    ax1.set_title(f"Số lượng và tỷ lệ yes theo nhóm {col}")
    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labs+labs2, loc="upper left")
    fig.tight_layout()
    save(fig, f"{idx:02d}_{col}_bins_yes_rate.png")

# 20. Pairplot on a balanced, capped sample.
pair_sample = (df.groupby("y", group_keys=False)
                 .apply(lambda g: g.sample(min(len(g), 1500), random_state=7), include_groups=False)
                 .reset_index(drop=True))
pair_sample["y"] = np.where(pair_sample["y_num"] == 1, "yes", "no")
for col in ["balance", "duration"]:
    pair_sample[col] = pair_sample[col].clip(df[col].quantile(.01), df[col].quantile(.99))
g = sns.pairplot(pair_sample, vars=NUM, hue="y", palette=COLORS,
                 corner=True, diag_kind="hist", plot_kws={"alpha": .3, "s": 14, "linewidth": 0})
g.fig.suptitle("Pairplot tổng hợp ba biến định lượng theo y", y=1.02, fontsize=16)
g.savefig(OUT / "20_pairplot_numeric_by_y.png", dpi=180, bbox_inches="tight")
plt.close(g.fig)


# Written report: each figure gets explicit axes and a data-backed interpretation.
yes_rate = df["y_num"].mean()
pearson = df[NUM + ["y_num"]].corr()
spearman = df[NUM + ["y_num"]].corr(method="spearman")
med_y = df.groupby("y")[NUM].median()
top_jobs = df["job"].value_counts()
rates = df.groupby("job")["y_num"].mean().sort_values(ascending=False)

outlier_lines = []
for col in NUM:
    row = desc.loc[col]
    outlier_lines.append(
        f"- **{col}**: Q1={row['25%']:,.0f}, trung vị={row['50%']:,.0f}, Q3={row['75%']:,.0f}; "
        f"hàng rào IQR [{row['lower_fence']:,.0f}, {row['upper_fence']:,.0f}], "
        f"{int(row['outlier_count']):,} điểm ngoài hàng rào ({row['outlier_count']/len(df):.1%}), độ lệch={row['skewness']:.2f}."
    )

report = f"""# Báo cáo trực quan hóa: age, balance, duration, job và y

**Nguồn:** `bank/bank-full.csv`  
**Quy mô:** {len(df):,} dòng, 5 biến phân tích; không có giá trị thiếu trong 5 biến.  
**Mục tiêu:** `y=yes` nghĩa là khách hàng đăng ký tiền gửi kỳ hạn.

Tên file PNG bắt đầu bằng số thứ tự tương ứng với từng mục giải thích dưới đây.

## Thống kê nền để đối chiếu

{chr(10).join(outlier_lines)}

- `y=yes`: {y_count['yes']:,}/{len(df):,} = **{yes_rate:.1%}**; dữ liệu mất cân bằng mạnh về lớp `no` ({1-yes_rate:.1%}).
- Trung vị theo `y`: age no/yes = {med_y.loc['no','age']:.0f}/{med_y.loc['yes','age']:.0f}; balance = {med_y.loc['no','balance']:.0f}/{med_y.loc['yes','balance']:.0f}; duration = {med_y.loc['no','duration']:.0f}/{med_y.loc['yes','duration']:.0f}.
- Pearson với `y`: age={pearson.loc['age','y_num']:.3f}, balance={pearson.loc['balance','y_num']:.3f}, duration={pearson.loc['duration','y_num']:.3f}. Duration nổi bật nhất, nhưng lưu ý đây là thông tin chỉ biết sau khi cuộc gọi kết thúc.

## Giải thích từng biểu đồ

### 01 — Histogram và KDE
- **Trục X:** giá trị age, balance, duration. **Trục Y:** số quan sát trong mỗi khoảng; đường KDE biểu diễn mật độ làm trơn.
- **Nhận xét:** age tập trung quanh 30–50 và lệch phải nhẹ. Balance và duration lệch phải rất mạnh: trung bình cao hơn trung vị rõ rệt (balance {df.balance.mean():,.0f} so với {df.balance.median():,.0f}; duration {df.duration.mean():,.0f} so với {df.duration.median():,.0f}). Co biên 1%–99% chỉ phục vụ khả năng đọc, không xóa dữ liệu khi tính thống kê.

### 02 — Histogram toàn miền với trục Y log
- **Trục X:** giá trị thật trên toàn miền. **Trục Y:** số quan sát theo thang log.
- **Nhận xét:** nhìn thấy các đuôi hiếm mà histogram thường che khuất: balance từ {df.balance.min():,} đến {df.balance.max():,}; duration tới {df.duration.max():,} giây; age tới {df.age.max():,}. Đây là bằng chứng hình học cho độ lệch và outlier trong thống kê.

### 03 — Boxplot toàn miền
- **Trục X:** giá trị biến; hộp từ Q1 đến Q3, đường giữa là trung vị, điểm ngoài râu là dị biệt theo quy tắc 1.5×IQR. **Trục Y:** chỉ là vị trí biến.
- **Nhận xét:** balance có nhiều điểm cực trị nhất về biên độ; duration có đuôi dài; age có nhóm tuổi cao hiếm. Boxplot bị nén ở balance chính là dấu hiệu cực trị rất lớn, không phải lỗi vẽ.

### 04 — Boxplot theo y
- **Trục X:** lớp `no/yes`. **Trục Y:** giá trị biến.
- **Nhận xét:** chênh lệch nổi bật nhất là duration: trung vị `yes`={med_y.loc['yes','duration']:.0f}s so với `no`={med_y.loc['no','duration']:.0f}s. Balance của nhóm yes cũng cao hơn, còn age chồng lấn mạnh nên khó tách lớp một mình.

### 05 — Scatter age–balance
- **Trục X:** tuổi. **Trục Y:** balance; màu là y.
- **Nhận xét:** đám mây phân tán rộng, không có quan hệ tuyến tính mạnh (Pearson={pearson.loc['age','balance']:.3f}); hai lớp y chồng lấn đáng kể.

### 06 — Scatter age–duration
- **Trục X:** tuổi. **Trục Y:** thời lượng; màu là y.
- **Nhận xét:** không thấy tuổi quyết định thời lượng; các điểm yes xuất hiện dày hơn ở vùng thời lượng cao. Quan hệ của duration với y mạnh hơn age.

### 07 — Scatter balance–duration
- **Trục X:** balance. **Trục Y:** duration; màu là y.
- **Nhận xét:** không có đường xu hướng tuyến tính rõ giữa hai biến (Pearson={pearson.loc['balance','duration']:.3f}); y=yes tập trung tương đối nhiều ở nửa trên của trục duration, bất kể balance.

### 08 — Heatmap tương quan
- **Hai trục:** cùng liệt kê age, balance, duration và y mã hóa 0/1; màu/số là hệ số tương quan.
- **Nhận xét:** Pearson đo tuyến tính, Spearman đo đơn điệu theo thứ hạng. Duration–y cao nhất ({pearson.loc['duration','y_num']:.3f}/{spearman.loc['duration','y_num']:.3f}); các cặp biến đầu vào có tương quan thấp, nên không có dấu hiệu đa cộng tuyến mạnh trong 3 biến này.

### 09 — Bar chart số lượng job
- **Trục X:** số khách hàng. **Trục Y:** nhóm nghề nghiệp, sắp tăng dần.
- **Nhận xét:** blue-collar ({top_jobs['blue-collar']:,}) và management ({top_jobs['management']:,}) chiếm nhiều nhất; unknown chỉ {top_jobs['unknown']:,}. So sánh tỷ lệ yes của nhóm nhỏ cần thận trọng vì bất định cao hơn.

### 10 — Doughnut tỷ trọng job
- **Góc/diện tích lát:** tỷ trọng số quan sát; chú giải ghi job và phần trăm. Biểu đồ tròn không có trục tọa độ.
- **Nhận xét:** cơ cấu tập trung vào blue-collar, management và technician; nhiều nhóm nhỏ khiến bar chart 09 chính xác hơn để so hạng, doughnut hữu ích cho cái nhìn cơ cấu.

### 11 — Bar và pie cho y
- **Bar:** X là lớp y, Y là số khách hàng. **Pie:** diện tích lát là tỷ trọng lớp.
- **Nhận xét:** `no`={y_count['no']:,} ({y_count['no']/len(df):.1%}), `yes`={y_count['yes']:,} ({yes_rate:.1%}). Khi xây mô hình, accuracy đơn thuần dễ gây hiểu lầm; nên xem recall, precision, F1/PR-AUC.

### 12 — 100% stacked bar job × y
- **Trục X:** tỷ trọng trong từng job (mỗi thanh = 100%). **Trục Y:** job; màu phân rã no/yes.
- **Nhận xét:** tỷ trọng yes khác đáng kể theo nghề; student và retired có phần màu yes lớn hơn, blue-collar nhỏ hơn. Biểu đồ chuẩn hóa giúp không bị số lượng nhóm chi phối.

### 13 — Tỷ lệ yes theo job và khoảng tin cậy
- **Trục X:** tỷ lệ y=yes. **Trục Y:** job. Điểm là tỷ lệ mẫu, thanh ngang là Wilson 95%.
- **Nhận xét:** cao nhất là {rates.index[0]} ({rates.iloc[0]:.1%}), tiếp theo {rates.index[1]} ({rates.iloc[1]:.1%}); thấp nhất là {rates.index[-1]} ({rates.iloc[-1]:.1%}). Khoảng rộng hơn ở nhóm nhỏ thể hiện đúng mức bất định.

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
"""

(OUT / "BAO_CAO_NHAN_XET.md").write_text(report, encoding="utf-8")

index = pd.DataFrame({
    "file": sorted(p.name for p in OUT.glob("*.png")),
})
index["mo_ta"] = [
    "Histogram và KDE", "Histogram toàn miền, trục đếm log", "Boxplot toàn miền",
    "Boxplot theo y", "Scatter age-balance", "Scatter age-duration",
    "Scatter balance-duration", "Heatmap tương quan", "Bar số lượng job",
    "Doughnut tỷ trọng job", "Bar và pie của y", "100% stacked job-y",
    "Tỷ lệ yes theo job + CI", "Violin theo y", "ECDF theo y",
    "Hexbin balance-duration", "Nhóm tuổi + tỷ lệ yes", "Decile balance + tỷ lệ yes",
    "Decile duration + tỷ lệ yes", "Pairplot tổng hợp"
]
index.to_csv(OUT / "DANH_MUC_BIEU_DO.csv", index=False, encoding="utf-8-sig")

print(f"Created {len(index)} images in {OUT.resolve()}")
print(f"Report: {(OUT / 'BAO_CAO_NHAN_XET.md').resolve()}")
