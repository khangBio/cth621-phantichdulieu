from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


INPUT_PATH = Path("all_stocks_5yr.csv")
OUTPUT_DIR = Path("outputs/gom_cum_all_stocks")
RANDOM_SEED = 42

TARGET_COLUMNS = ["close", "target_close_next", "target_date", "cv_fold", "split"]
NUMERIC_BASE_COLUMNS = ["open", "high", "low", "volume"]
FEATURE_COLUMNS = [
    "log_open",
    "log_high",
    "log_low",
    "log_volume",
    "intraday_range_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
]


COLORS = [
    (58, 123, 213),
    (222, 98, 98),
    (74, 163, 123),
    (217, 160, 65),
    (141, 102, 198),
    (71, 169, 191),
    (192, 99, 158),
    (134, 134, 134),
    (120, 175, 70),
    (225, 125, 49),
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT = load_font(22)
FONT_SMALL = load_font(17)
FONT_TINY = load_font(14)
FONT_BOLD = load_font(24, bold=True)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font=None, fill=(30, 30, 30)) -> None:
    draw.text(xy, text, font=font or FONT, fill=fill)


def make_canvas(width: int = 1600, height: int = 1000) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def axis_ticks(vmin: float, vmax: float, n: int = 6) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
        return np.linspace(0, 1, n)
    return np.linspace(vmin, vmax, n)


def safe_minmax(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if math.isclose(vmin, vmax):
        pad = abs(vmin) * 0.05 + 1.0
        return vmin - pad, vmax + pad
    pad = (vmax - vmin) * 0.06
    return vmin - pad, vmax + pad


def plot_line(
    path: Path,
    x: list[int],
    series: dict[str, list[float]],
    title: str,
    x_label: str,
    y_label: str,
    subtitle: str = "",
) -> None:
    width, height = 1500, 900
    margin_l, margin_r, margin_t, margin_b = 120, 70, 110, 120
    image, draw = make_canvas(width, height)
    draw_text(draw, (margin_l, 30), title, FONT_BOLD)
    if subtitle:
        draw_text(draw, (margin_l, 65), subtitle, FONT_SMALL, (90, 90, 90))
    all_y = np.array([v for vals in series.values() for v in vals], dtype=float)
    ymin, ymax = safe_minmax(all_y)
    xmin, xmax = min(x), max(x)

    def sx(v):
        return margin_l + (v - xmin) / (xmax - xmin) * (width - margin_l - margin_r)

    def sy(v):
        return height - margin_b - (v - ymin) / (ymax - ymin) * (height - margin_t - margin_b)

    draw.line((margin_l, margin_t, margin_l, height - margin_b), fill=(70, 70, 70), width=2)
    draw.line((margin_l, height - margin_b, width - margin_r, height - margin_b), fill=(70, 70, 70), width=2)

    for tick in axis_ticks(ymin, ymax):
        ypix = sy(tick)
        draw.line((margin_l, ypix, width - margin_r, ypix), fill=(230, 230, 230), width=1)
        draw_text(draw, (20, int(ypix) - 10), f"{tick:,.2f}", FONT_TINY, (90, 90, 90))
    for tick in x:
        xpix = sx(tick)
        draw.line((xpix, height - margin_b, xpix, height - margin_b + 8), fill=(90, 90, 90), width=1)
        draw_text(draw, (int(xpix) - 8, height - margin_b + 15), str(tick), FONT_TINY, (90, 90, 90))

    for idx, (name, vals) in enumerate(series.items()):
        pts = [(sx(xv), sy(yv)) for xv, yv in zip(x, vals)]
        color = COLORS[idx % len(COLORS)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4)
        for p in pts:
            draw.ellipse((p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5), fill=color)
        lx = margin_l + idx * 260
        ly = height - 60
        draw.rectangle((lx, ly, lx + 20, ly + 12), fill=color)
        draw_text(draw, (lx + 28, ly - 6), name, FONT_SMALL)

    draw_text(draw, (width // 2 - 80, height - 35), x_label, FONT_SMALL)
    draw_text(draw, (25, margin_t - 35), y_label, FONT_SMALL)
    image.save(path)


def plot_scatter(
    path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    title: str,
    x_label: str = "PC1",
    y_label: str = "PC2",
    subtitle: str = "",
    noise_label: int | None = None,
) -> None:
    width, height = 1500, 1000
    margin_l, margin_r, margin_t, margin_b = 115, 220, 110, 115
    image, draw = make_canvas(width, height)
    draw_text(draw, (margin_l, 30), title, FONT_BOLD)
    if subtitle:
        draw_text(draw, (margin_l, 65), subtitle, FONT_SMALL, (90, 90, 90))
    x = points[:, 0]
    y = points[:, 1]
    xmin, xmax = safe_minmax(x)
    ymin, ymax = safe_minmax(y)

    def sx(v):
        return margin_l + (v - xmin) / (xmax - xmin) * (width - margin_l - margin_r)

    def sy(v):
        return height - margin_b - (v - ymin) / (ymax - ymin) * (height - margin_t - margin_b)

    draw.line((margin_l, margin_t, margin_l, height - margin_b), fill=(70, 70, 70), width=2)
    draw.line((margin_l, height - margin_b, width - margin_r, height - margin_b), fill=(70, 70, 70), width=2)
    for tick in axis_ticks(xmin, xmax):
        xpix = sx(tick)
        draw.line((xpix, margin_t, xpix, height - margin_b), fill=(235, 235, 235), width=1)
        draw_text(draw, (int(xpix) - 28, height - margin_b + 15), f"{tick:.1f}", FONT_TINY, (90, 90, 90))
    for tick in axis_ticks(ymin, ymax):
        ypix = sy(tick)
        draw.line((margin_l, ypix, width - margin_r, ypix), fill=(235, 235, 235), width=1)
        draw_text(draw, (25, int(ypix) - 10), f"{tick:.1f}", FONT_TINY, (90, 90, 90))

    order = np.argsort(labels)
    for i in order:
        lab = int(labels[i])
        color = (60, 60, 60) if noise_label is not None and lab == noise_label else COLORS[lab % len(COLORS)]
        px, py = sx(points[i, 0]), sy(points[i, 1])
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)

    unique_labels = sorted(Counter(labels).keys())
    legend_y = margin_t
    for lab in unique_labels[:18]:
        color = (60, 60, 60) if noise_label is not None and lab == noise_label else COLORS[int(lab) % len(COLORS)]
        label_text = "Noise (-1)" if noise_label is not None and int(lab) == noise_label else f"Cụm {lab}"
        draw.rectangle((width - margin_r + 25, legend_y + 4, width - margin_r + 45, legend_y + 18), fill=color)
        draw_text(draw, (width - margin_r + 55, legend_y - 2), label_text, FONT_SMALL)
        legend_y += 30
    draw_text(draw, (width // 2 - 40, height - 35), x_label, FONT_SMALL)
    draw_text(draw, (25, margin_t - 35), y_label, FONT_SMALL)
    image.save(path)


def draw_centered_multiline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    lines: list[str],
    fonts: list[ImageFont.ImageFont],
    fills: list[tuple[int, int, int]],
    line_gap: int = 4,
) -> None:
    x0, y0, x1, y1 = box
    heights = []
    widths = []
    for line, font in zip(lines, fonts):
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = y0 + ((y1 - y0) - total_h) / 2
    for line, font, fill, h, w in zip(lines, fonts, fills, heights, widths):
        x = x0 + ((x1 - x0) - w) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += h + line_gap


def blend_color(value: float, vmax: float) -> tuple[int, int, int]:
    intensity = min(1.0, abs(value) / max(vmax, 1e-9))
    if value >= 0:
        base = np.array([42, 111, 187])
    else:
        base = np.array([203, 73, 73])
    return tuple((255 - intensity * (255 - base)).astype(int))


def plot_heatmap(
    path: Path,
    matrix: pd.DataFrame,
    title: str,
    subtitle: str = "",
    annotations: pd.DataFrame | None = None,
) -> None:
    n_rows, n_cols = matrix.shape
    cell_w = 215
    cell_h = 105
    margin_l = 190
    margin_t = 185
    margin_r = 70
    margin_b = 190
    width = margin_l + n_cols * cell_w + margin_r
    height = margin_t + n_rows * cell_h + margin_b
    image, draw = make_canvas(width, height)
    draw_text(draw, (40, 30), title, FONT_BOLD)
    if subtitle:
        draw_text(draw, (40, 65), subtitle, FONT_SMALL, (90, 90, 90))
    values = matrix.to_numpy(dtype=float)
    vmax = max(abs(float(np.nanmin(values))), abs(float(np.nanmax(values))), 1e-9)

    draw_text(draw, (30, margin_t - 45), "Cụm", FONT_SMALL, (70, 70, 70))
    for j, col in enumerate(matrix.columns):
        x0 = margin_l + j * cell_w
        draw_centered_multiline(
            draw,
            (x0 + 5, margin_t - 82, x0 + cell_w - 5, margin_t - 18),
            str(col).split("\n"),
            [FONT_SMALL] * len(str(col).split("\n")),
            [(45, 45, 45)] * len(str(col).split("\n")),
            line_gap=2,
        )

    for i, row in enumerate(matrix.index):
        y0 = margin_t + i * cell_h
        draw.rectangle((25, y0, margin_l - 18, y0 + cell_h - 5), fill=(245, 247, 250), outline=(220, 225, 232))
        draw_centered_multiline(
            draw,
            (25, y0, margin_l - 18, y0 + cell_h - 5),
            [str(row)],
            [FONT_BOLD],
            [(35, 35, 35)],
        )
        for j, col in enumerate(matrix.columns):
            val = float(matrix.loc[row, col])
            rgb = blend_color(val, vmax)
            x0 = margin_l + j * cell_w
            draw.rectangle((x0, y0, x0 + cell_w - 4, y0 + cell_h - 5), fill=rgb, outline=(235, 235, 235))
            z_text = f"Z = {val:+.2f}"
            detail = ""
            if annotations is not None:
                detail = str(annotations.loc[row, col])
            draw_centered_multiline(
                draw,
                (x0 + 6, y0 + 6, x0 + cell_w - 10, y0 + cell_h - 11),
                [z_text, detail],
                [FONT_SMALL, FONT_TINY],
                [(20, 20, 20), (45, 45, 45)],
                line_gap=7,
            )

    legend_x = margin_l
    legend_y = height - 115
    draw_text(draw, (legend_x, legend_y - 38), "Chú giải màu: Z-score so với trung bình các cụm", FONT_SMALL)
    legend_w = min(720, n_cols * cell_w)
    for step in range(legend_w):
        z = -vmax + 2 * vmax * step / max(1, legend_w - 1)
        draw.line((legend_x + step, legend_y, legend_x + step, legend_y + 24), fill=blend_color(z, vmax))
    draw.rectangle((legend_x, legend_y, legend_x + legend_w, legend_y + 24), outline=(170, 170, 170))
    draw_text(draw, (legend_x, legend_y + 34), f"Thấp hơn TB ({-vmax:.1f})", FONT_TINY, (80, 80, 80))
    draw_text(draw, (legend_x + legend_w // 2 - 35, legend_y + 34), "Trung bình", FONT_TINY, (80, 80, 80))
    draw_text(draw, (legend_x + legend_w - 110, legend_y + 34), f"Cao hơn TB (+{vmax:.1f})", FONT_TINY, (80, 80, 80))

    note = "Mỗi ô hiển thị Z-score và giá trị trung bình thực tế của đặc trưng trong cụm."
    draw_text(draw, (40, height - 45), note, FONT_SMALL, (90, 90, 90))
    image.save(path)


def plot_dendrogram(path: Path, merges: list[tuple[int, int, float, int]], n_leaves: int, title: str) -> None:
    width, height = 1800, 950
    margin_l, margin_r, margin_t, margin_b = 90, 50, 100, 120
    image, draw = make_canvas(width, height)
    draw_text(draw, (margin_l, 30), title, FONT_BOLD)
    draw_text(draw, (margin_l, 65), "Trục X: các quan sát mẫu; Trục Y: khoảng cách gộp cụm (average linkage).", FONT_SMALL, (90, 90, 90))
    max_dist = max([m[2] for m in merges] or [1.0])
    leaf_x = {i: margin_l + i / max(1, n_leaves - 1) * (width - margin_l - margin_r) for i in range(n_leaves)}
    cluster_x = dict(leaf_x)
    cluster_y = {i: height - margin_b for i in range(n_leaves)}

    def sy(dist):
        return height - margin_b - dist / max_dist * (height - margin_t - margin_b)

    draw.line((margin_l, margin_t, margin_l, height - margin_b), fill=(70, 70, 70), width=2)
    draw.line((margin_l, height - margin_b, width - margin_r, height - margin_b), fill=(70, 70, 70), width=2)
    for tick in axis_ticks(0, max_dist):
        y = sy(tick)
        draw.line((margin_l, y, width - margin_r, y), fill=(235, 235, 235), width=1)
        draw_text(draw, (10, int(y) - 10), f"{tick:.1f}", FONT_TINY, (90, 90, 90))

    for merge_idx, (a, b, dist, _count) in enumerate(merges):
        new_id = n_leaves + merge_idx
        x1, x2 = cluster_x[a], cluster_x[b]
        y1, y2 = cluster_y[a], cluster_y[b]
        y = sy(dist)
        color = COLORS[merge_idx % len(COLORS)] if dist > max_dist * 0.55 else (90, 90, 90)
        draw.line((x1, y1, x1, y), fill=color, width=2)
        draw.line((x2, y2, x2, y), fill=color, width=2)
        draw.line((x1, y, x2, y), fill=color, width=2)
        cluster_x[new_id] = (x1 + x2) / 2
        cluster_y[new_id] = y
    draw_text(draw, (width // 2 - 60, height - 35), "Quan sát mẫu", FONT_SMALL)
    draw_text(draw, (10, margin_t + 8), "Khoảng cách", FONT_SMALL)
    image.save(path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["date"] + NUMERIC_BASE_COLUMNS)
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["volume"] > 0)].copy()
    df["log_open"] = np.log(df["open"])
    df["log_high"] = np.log(df["high"])
    df["log_low"] = np.log(df["low"])
    df["log_volume"] = np.log1p(df["volume"])
    df["intraday_range_pct"] = (df["high"] - df["low"]) / df["open"]
    df["upper_shadow_pct"] = (df["high"] - df[["open", "low"]].max(axis=1)) / df["open"]
    df["lower_shadow_pct"] = (df[["open", "high"]].min(axis=1) - df["low"]) / df["open"]
    month_angle = 2 * np.pi * df["date"].dt.month / 12
    dow_angle = 2 * np.pi * df["date"].dt.dayofweek / 5
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)
    df["dow_sin"] = np.sin(dow_angle)
    df["dow_cos"] = np.cos(dow_angle)
    feature_df = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna()
    df = df.loc[feature_df.index].copy()
    metadata = {
        "input_rows": int(before),
        "usable_rows": int(len(df)),
        "removed_rows": int(before - len(df)),
        "target_columns_removed": [c for c in TARGET_COLUMNS if c in df.columns or c == "close"],
        "features_used": FEATURE_COLUMNS,
        "base_columns_kept_out_of_target": NUMERIC_BASE_COLUMNS,
    }
    return df.reset_index(drop=True), feature_df.reset_index(drop=True), metadata


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    components = vt[:2].T
    coords = (x - x.mean(axis=0)) @ components
    variance = np.var(coords, axis=0)
    ratio = variance / np.var(x, axis=0).sum()
    return coords, components, ratio


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 80, n_init: int = 5) -> tuple[np.ndarray, np.ndarray, float]:
    best_labels, best_centers, best_inertia = None, None, np.inf
    n = len(x)
    for _ in range(n_init):
        centers = np.empty((k, x.shape[1]), dtype=float)
        centers[0] = x[rng.integers(n)]
        closest = np.sum((x - centers[0]) ** 2, axis=1)
        for ci in range(1, k):
            probs = closest / closest.sum()
            idx = rng.choice(n, p=probs)
            centers[ci] = x[idx]
            closest = np.minimum(closest, np.sum((x - centers[ci]) ** 2, axis=1))
        labels = np.zeros(n, dtype=int)
        for _iter in range(max_iter):
            dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(labels, new_labels) and _iter > 0:
                break
            labels = new_labels
            for ci in range(k):
                mask = labels == ci
                if mask.any():
                    centers[ci] = x[mask].mean(axis=0)
                else:
                    centers[ci] = x[rng.integers(n)]
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_labels, best_centers, best_inertia = labels.copy(), centers.copy(), inertia
    return best_labels, best_centers, best_inertia


def assign_to_centers(x: np.ndarray, centers: np.ndarray, chunk: int = 100_000) -> tuple[np.ndarray, float]:
    labels = np.empty(len(x), dtype=int)
    inertia = 0.0
    for start in range(0, len(x), chunk):
        end = min(start + chunk, len(x))
        dists = ((x[start:end, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels[start:end] = dists.argmin(axis=1)
        inertia += float(dists[np.arange(end - start), labels[start:end]].sum())
    return labels, inertia


def silhouette_sample(x: np.ndarray, labels: np.ndarray, max_points: int, rng: np.random.Generator) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    idx = rng.choice(len(x), size=min(max_points, len(x)), replace=False)
    xs = x[idx]
    labs = labels[idx]
    dmat = np.sqrt(((xs[:, None, :] - xs[None, :, :]) ** 2).sum(axis=2))
    scores = []
    for i, lab in enumerate(labs):
        same = labs == lab
        same[i] = False
        a = dmat[i, same].mean() if same.any() else 0.0
        b = np.inf
        for other in np.unique(labs):
            if other == lab:
                continue
            mask = labs == other
            if mask.any():
                b = min(b, dmat[i, mask].mean())
        denom = max(a, b)
        scores.append(0.0 if denom == 0 or not np.isfinite(denom) else (b - a) / denom)
    return float(np.mean(scores))


def choose_k(summary: pd.DataFrame) -> int:
    k_values = summary["k"].to_numpy()
    inertia = summary["inertia"].to_numpy(dtype=float)
    if len(k_values) < 3:
        return int(k_values[np.argmin(inertia)])
    x = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    y = (inertia - inertia.min()) / (inertia.max() - inertia.min())
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line = p2 - p1
    distances = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi])
        distances.append(abs(line[0] * (p1[1] - p[1]) - line[1] * (p1[0] - p[0])) / np.linalg.norm(line))
    elbow_k = int(k_values[int(np.argmax(distances))])
    top_sil = summary.sort_values(["silhouette_sample", "k"], ascending=[False, True]).iloc[0]
    # Ưu tiên silhouette nếu nó cao hơn rõ rệt; nếu không lấy điểm gãy elbow.
    if float(top_sil["silhouette_sample"]) >= float(summary.loc[summary["k"] == elbow_k, "silhouette_sample"].iloc[0]) + 0.03:
        return int(top_sil["k"])
    return elbow_k


def agglomerative_average_linkage(x: np.ndarray) -> list[tuple[int, int, float, int]]:
    n = len(x)
    dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    clusters = {i: [i] for i in range(n)}
    active = set(range(n))
    next_id = n
    merges: list[tuple[int, int, float, int]] = []
    while len(active) > 1:
        active_list = sorted(active)
        best_pair = None
        best_dist = np.inf
        for pos, a in enumerate(active_list[:-1]):
            da = dist[a, active_list[pos + 1 :]]
            j = int(np.argmin(da))
            if da[j] < best_dist:
                best_dist = float(da[j])
                best_pair = (a, active_list[pos + 1 + j])
        assert best_pair is not None
        a, b = best_pair
        members = clusters[a] + clusters[b]
        merges.append((a, b, best_dist, len(members)))
        clusters[next_id] = members
        dist = np.pad(dist, ((0, 1), (0, 1)), constant_values=np.inf)
        for c in active:
            if c in (a, b):
                continue
            new_dist = np.mean([np.linalg.norm(x[i] - x[j]) for i in members for j in clusters[c]])
            dist[next_id, c] = new_dist
            dist[c, next_id] = new_dist
        active.remove(a)
        active.remove(b)
        active.add(next_id)
        next_id += 1
    return merges


def estimate_dbscan_eps(coords: np.ndarray, k: int = 12) -> float:
    sample = coords
    dists = np.sqrt(((sample[:, None, :] - sample[None, :, :]) ** 2).sum(axis=2))
    kth = np.partition(dists, k, axis=1)[:, k]
    return float(np.quantile(kth, 0.90))


def dbscan_2d(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = len(points)
    labels = np.full(n, -99, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cell_size = eps
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    cells = np.floor(points / cell_size).astype(int)
    for i, cell in enumerate(cells):
        grid[(int(cell[0]), int(cell[1]))].append(i)

    def region_query(i: int) -> list[int]:
        cx, cy = cells[i]
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(grid.get((int(cx + dx), int(cy + dy)), []))
        if not candidates:
            return []
        cand = np.array(candidates, dtype=int)
        d = np.sqrt(((points[cand] - points[i]) ** 2).sum(axis=1))
        return cand[d <= eps].tolist()

    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = list(neighbors)
        in_seeds = np.zeros(n, dtype=bool)
        in_seeds[seeds] = True
        cursor = 0
        while cursor < len(seeds):
            j = seeds[cursor]
            if not visited[j]:
                visited[j] = True
                neighbors_j = region_query(j)
                if len(neighbors_j) >= min_samples:
                    for nb in neighbors_j:
                        if not in_seeds[nb]:
                            seeds.append(nb)
                            in_seeds[nb] = True
            if labels[j] in (-99, -1):
                labels[j] = cluster_id
            cursor += 1
        cluster_id += 1
    labels[labels == -99] = -1
    return labels


def cluster_profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    work = df.copy()
    work["cluster"] = labels
    aggregations = {
        "open": ["count", "mean", "median"],
        "high": ["mean"],
        "low": ["mean"],
        "volume": ["mean", "median"],
        "intraday_range_pct": ["mean"],
        "upper_shadow_pct": ["mean"],
        "lower_shadow_pct": ["mean"],
    }
    profile = work.groupby("cluster").agg(aggregations)
    profile.columns = ["_".join(col).strip("_") for col in profile.columns.to_flat_index()]
    profile = profile.rename(columns={"open_count": "count"})
    profile.insert(1, "pct_rows", profile["count"] / len(work))
    return profile.reset_index()


def markdown_report(
    metadata: dict[str, object],
    k_summary: pd.DataFrame,
    selected_k: int,
    profile: pd.DataFrame,
    dbscan_info: dict[str, object],
    pca_ratio: np.ndarray,
) -> str:
    best = k_summary.loc[k_summary["k"] == selected_k].iloc[0]
    largest = profile.sort_values("count", ascending=False).iloc[0]
    high_price_cluster = profile.sort_values("open_mean", ascending=False).iloc[0]
    high_volume_cluster = profile.sort_values("volume_mean", ascending=False).iloc[0]
    high_range_cluster = profile.sort_values("intraday_range_pct_mean", ascending=False).iloc[0]
    db_clusters = dbscan_info["n_clusters"]
    db_noise_pct = dbscan_info["noise_pct"]
    lines = [
        "# Bài toán Gom cụm trên all_stocks_5yr",
        "",
        "## 1. Thiết lập bài toán",
        "",
        "- Mục tiêu là học không giám sát: không dự đoán `close`, mà để thuật toán tự tìm cấu trúc nhóm trong dữ liệu giao dịch.",
        f"- Các cột nhãn/target bị loại khỏi ma trận đặc trưng: `{', '.join(metadata['target_columns_removed'])}`. Đặc biệt, `close` không được đưa vào clustering.",
        f"- Số dòng ban đầu: {metadata['input_rows']:,}; số dòng dùng được sau làm sạch: {metadata['usable_rows']:,}; số dòng loại do thiếu/sai giá trị: {metadata['removed_rows']:,}.",
        "- Đặc trưng sử dụng gồm log-thang đo của `open`, `high`, `low`, `volume`, các tỷ lệ dao động trong ngày và mã hóa chu kỳ tháng/thứ trong tuần.",
        "- Tất cả đặc trưng được chuẩn hóa Z-score trước khi chạy thuật toán để `volume` hoặc giá cổ phiếu lớn không lấn át các biến còn lại.",
        "",
        "## 2. K-Means Clustering",
        "",
        f"- Đã thử K từ {int(k_summary['k'].min())} đến {int(k_summary['k'].max())}. Số cụm được chọn: **K = {selected_k}**.",
        f"- Lý do chọn: K={selected_k} nằm tại vùng cân bằng giữa độ giảm inertia và silhouette mẫu. Với K này, inertia = {best['inertia']:,.0f}, silhouette mẫu = {best['silhouette_sample']:.3f}.",
        "- Diễn giải: inertia càng nhỏ thì cụm càng chặt; silhouette càng gần 1 thì các điểm càng nằm đúng cụm và tách biệt cụm khác. K tối ưu không nhất thiết là K có inertia nhỏ nhất, vì inertia luôn giảm khi tăng K.",
        "",
        "### Hồ sơ cụm K-Means",
        "",
        "| Cụm | Số dòng | Tỷ trọng | Open TB | Volume TB | Range trong ngày TB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in profile.iterrows():
        lines.append(
            f"| {int(row['cluster'])} | {int(row['count']):,} | {row['pct_rows']:.2%} | "
            f"{row['open_mean']:.2f} | {row['volume_mean']:,.0f} | {row['intraday_range_pct_mean']:.2%} |"
        )
    lines.extend(
        [
            "",
            f"- Cụm lớn nhất là cụm {int(largest['cluster'])}, chiếm {largest['pct_rows']:.2%}. Đây là trạng thái giao dịch phổ biến nhất trong dữ liệu.",
            f"- Cụm {int(high_price_cluster['cluster'])} có mức `open` trung bình cao hơn ({high_price_cluster['open_mean']:.2f}) nhưng volume trung bình thấp hơn, nên có thể xem là nhóm cổ phiếu/phiên giao dịch giá cao và thanh khoản vừa phải hơn.",
            f"- Cụm {int(high_volume_cluster['cluster'])} có volume trung bình cao hơn ({high_volume_cluster['volume_mean']:,.0f}) và cụm {int(high_range_cluster['cluster'])} có biên dao động trong ngày cao hơn ({high_range_cluster['intraday_range_pct_mean']:.2%}); đây là vùng cần chú ý khi tìm trạng thái giao dịch sôi động hoặc biến động mạnh.",
            "",
            "## 3. Hierarchical Clustering",
            "",
            "- Gom cụm phân cấp được trực quan hóa bằng dendrogram trên mẫu con, vì dendrogram cần tính khoảng cách cặp điểm và không phù hợp để vẽ trực tiếp toàn bộ 619 nghìn quan sát.",
            "- Trục X là các quan sát mẫu; trục Y là khoảng cách khi hai nhánh/cụm được gộp lại. Nhánh nào nhập vào nhau ở độ cao thấp nghĩa là hai nhóm quan sát gần nhau; nhánh tách xa và chỉ nhập ở độ cao lớn thể hiện cấu trúc khác biệt rõ.",
            "- Dendrogram giúp kiểm chứng trực quan số cụm K-Means: nếu có vài nhánh lớn chỉ nhập ở mức khoảng cách cao, dữ liệu thật sự có phân tầng thay vì chỉ là một đám mây liên tục.",
            "",
            "## 4. DBSCAN",
            "",
            f"- DBSCAN được chạy trên không gian PCA 2 chiều của mẫu lớn để phát hiện cấu trúc mật độ và noise. Tham số dùng: eps = {dbscan_info['eps']:.3f}, min_samples = {dbscan_info['min_samples']}.",
            f"- Kết quả: phát hiện {db_clusters} cụm mật độ và {dbscan_info['noise_count']:,} điểm noise, tương đương {db_noise_pct:.2%} trong mẫu DBSCAN.",
            "- Điểm noise không nhất thiết là lỗi dữ liệu; trong ngữ cảnh cổ phiếu, đó có thể là phiên giao dịch khối lượng đột biến, biên độ ngày rất rộng, hoặc cổ phiếu có mức giá/thanh khoản khác xa phần đông.",
            "",
            "## 5. Nhận xét tổng hợp",
            "",
            f"- Hai thành phần PCA đầu tiên dùng để vẽ scatter giải thích khoảng {pca_ratio[:2].sum():.2%} biến thiên chuẩn hóa. Vì vậy scatter 2D là bản đồ trực quan, còn phân cụm K-Means vẫn dựa trên toàn bộ đặc trưng chuẩn hóa.",
            "- K-Means phù hợp để chia dữ liệu thành các trạng thái giao dịch tương đối đều và dễ mô tả bằng trung bình cụm.",
            "- Hierarchical Clustering phù hợp để nhìn cấu trúc phân tách nhiều tầng, nhưng chỉ nên dùng mẫu con khi dữ liệu rất lớn.",
            "- DBSCAN bổ sung góc nhìn về điểm dị biệt/noise, hữu ích hơn K-Means khi muốn cô lập những phiên giao dịch bất thường.",
            "",
            "## 6. Danh sách biểu đồ",
            "",
            "1. `01_kmeans_elbow_silhouette.png`: Line chart chọn K tối ưu.",
            "2. `02_kmeans_pca_scatter.png`: Scatter PCA theo nhãn cụm K-Means.",
            "3. `03_hierarchical_dendrogram.png`: Dendrogram gom cụm phân cấp trên mẫu con.",
            "4. `04_dbscan_pca_scatter.png`: Scatter PCA theo cụm/noise DBSCAN.",
            "5. `05_kmeans_cluster_profile_heatmap.png`: Heatmap hồ sơ trung bình theo cụm.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(INPUT_PATH)
    df, feature_df, metadata = prepare_features(df_raw)
    x = feature_df.to_numpy(dtype=float)
    xz, mean, std = standardize(x)
    metadata["standardization"] = {"method": "z_score", "mean": mean.tolist(), "std": std.tolist()}

    # K-Means: fit/đánh giá K trên mẫu ổn định, sau đó chỉ gán cụm cho toàn bộ dữ liệu với K được chọn.
    k_fit_size = min(30_000, len(xz))
    k_fit_idx = rng.choice(len(xz), size=k_fit_size, replace=False)
    x_fit = xz[k_fit_idx]
    k_rows = []
    centers_by_k: dict[int, np.ndarray] = {}
    for k in range(2, 11):
        labels_fit, centers, inertia_fit = kmeans(x_fit, k, rng, max_iter=55, n_init=2)
        sil = silhouette_sample(x_fit, labels_fit, max_points=1000, rng=rng)
        k_rows.append({"k": k, "inertia": inertia_fit, "silhouette_sample": sil})
        centers_by_k[k] = centers
    k_summary = pd.DataFrame(k_rows)
    selected_k = choose_k(k_summary)
    kmeans_labels, kmeans_inertia = assign_to_centers(xz, centers_by_k[selected_k])
    k_summary.loc[k_summary["k"] == selected_k, "selected"] = True
    k_summary["selected"] = k_summary["selected"].fillna(False)
    k_summary.to_csv(OUTPUT_DIR / "kmeans_k_selection.csv", index=False)

    profile = cluster_profile(df, kmeans_labels)
    profile.to_csv(OUTPUT_DIR / "kmeans_cluster_profile.csv", index=False)

    clustered = df[["date", "Name", "open", "high", "low", "volume"]].copy()
    clustered["kmeans_cluster"] = kmeans_labels
    clustered.to_csv(OUTPUT_DIR / "all_stocks_kmeans_clusters.csv", index=False)

    # PCA/scatter samples.
    pca_fit_size = min(40_000, len(xz))
    pca_fit_idx = rng.choice(len(xz), size=pca_fit_size, replace=False)
    _, components, pca_ratio = pca_2d(xz[pca_fit_idx])
    coords_full = (xz - xz.mean(axis=0)) @ components
    scatter_size = min(12_000, len(coords_full))
    scatter_idx = rng.choice(len(coords_full), size=scatter_size, replace=False)

    plot_line(
        OUTPUT_DIR / "01_kmeans_elbow_silhouette.png",
        k_summary["k"].astype(int).tolist(),
        {
            "Inertia chuẩn hóa": (k_summary["inertia"] / k_summary["inertia"].max()).tolist(),
            "Silhouette mẫu": k_summary["silhouette_sample"].tolist(),
        },
        "K-Means: Elbow và Silhouette để chọn số cụm K",
        "Số cụm K",
        "Giá trị chuẩn hóa / điểm silhouette",
        "Trục X: K thử nghiệm; Trục Y: inertia chuẩn hóa càng thấp càng tốt, silhouette càng cao càng tốt.",
    )
    plot_scatter(
        OUTPUT_DIR / "02_kmeans_pca_scatter.png",
        coords_full[scatter_idx],
        kmeans_labels[scatter_idx],
        f"K-Means PCA Scatter: K = {selected_k}",
        "PC1",
        "PC2",
        "Mỗi điểm là một phiên giao dịch; màu thể hiện cụm K-Means. Trục là 2 thành phần PCA đầu tiên.",
    )

    dendro_size = min(100, len(xz))
    dendro_idx = rng.choice(len(xz), size=dendro_size, replace=False)
    merges = agglomerative_average_linkage(xz[dendro_idx])
    plot_dendrogram(
        OUTPUT_DIR / "03_hierarchical_dendrogram.png",
        merges,
        dendro_size,
        "Hierarchical Clustering: Dendrogram trên mẫu con",
    )

    db_size = min(7_000, len(coords_full))
    db_idx = rng.choice(len(coords_full), size=db_size, replace=False)
    db_points = coords_full[db_idx]
    eps_sample_idx = rng.choice(len(db_points), size=min(1000, len(db_points)), replace=False)
    eps = estimate_dbscan_eps(db_points[eps_sample_idx], k=12)
    db_labels = dbscan_2d(db_points, eps=eps, min_samples=12)
    db_counts = Counter(db_labels)
    db_info = {
        "sample_size": int(db_size),
        "eps": float(eps),
        "min_samples": 12,
        "n_clusters": int(len([lab for lab in db_counts if lab != -1])),
        "noise_count": int(db_counts.get(-1, 0)),
        "noise_pct": float(db_counts.get(-1, 0) / db_size),
        "cluster_counts": {str(int(k)): int(v) for k, v in sorted(db_counts.items())},
    }
    pd.DataFrame(
        [{"dbscan_cluster": int(k), "count": int(v), "pct": float(v / db_size)} for k, v in sorted(db_counts.items())]
    ).to_csv(OUTPUT_DIR / "dbscan_cluster_counts.csv", index=False)
    plot_scatter(
        OUTPUT_DIR / "04_dbscan_pca_scatter.png",
        db_points,
        db_labels,
        "DBSCAN PCA Scatter: cụm mật độ và noise",
        "PC1",
        "PC2",
        "Màu đen là noise/outlier (-1); các màu khác là cụm mật độ DBSCAN.",
        noise_label=-1,
    )

    heat_cols = ["log_open", "log_volume", "intraday_range_pct", "upper_shadow_pct", "lower_shadow_pct"]
    heat_labels = {
        "log_open": "Giá mở cửa\n(Open TB)",
        "log_volume": "Khối lượng\n(Volume TB)",
        "intraday_range_pct": "Biên dao động\ntrong ngày",
        "upper_shadow_pct": "Dao động\nphía trên",
        "lower_shadow_pct": "Dao động\nphía dưới",
    }
    profile_z = pd.DataFrame(index=[f"Cụm {int(c)}" for c in profile["cluster"]])
    annotations = pd.DataFrame(index=profile_z.index)
    for col in heat_cols:
        original_col = {
            "log_open": "open_mean",
            "log_volume": "volume_mean",
            "intraday_range_pct": "intraday_range_pct_mean",
            "upper_shadow_pct": "upper_shadow_pct_mean",
            "lower_shadow_pct": "lower_shadow_pct_mean",
        }[col]
        vals = np.log1p(profile[original_col].to_numpy()) if original_col == "volume_mean" else profile[original_col].to_numpy()
        label = heat_labels[col]
        profile_z[label] = (vals - vals.mean()) / (vals.std(ddof=0) if vals.std(ddof=0) else 1)
        if original_col == "volume_mean":
            annotations[label] = [f"{v:,.0f}" for v in profile[original_col]]
        elif original_col == "open_mean":
            annotations[label] = [f"{v:,.2f}" for v in profile[original_col]]
        else:
            annotations[label] = [f"{v:.2%}" for v in profile[original_col]]
    plot_heatmap(
        OUTPUT_DIR / "05_kmeans_cluster_profile_heatmap.png",
        profile_z,
        "Heatmap hồ sơ trung bình theo cụm K-Means",
        "Trục X: đặc trưng mô tả cụm; Trục Y: cụm K-Means. Màu xanh = cao hơn trung bình, đỏ = thấp hơn trung bình.",
        annotations=annotations,
    )

    metadata.update(
        {
            "kmeans_selected_k": int(selected_k),
            "kmeans_inertia": float(kmeans_inertia),
            "pca_explained_ratio_first_two": pca_ratio[:2].tolist(),
            "dbscan": db_info,
            "sample_notes": {
                "kmeans_fit_sample": int(k_fit_size),
                "kmeans_final_assignment_rows": int(len(xz)),
                "hierarchical_sample": int(dendro_size),
                "dbscan_sample": int(db_size),
                "scatter_sample": int(scatter_size),
            },
        }
    )
    (OUTPUT_DIR / "clustering_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "nhan_xet_gom_cum.md").write_text(
        markdown_report(metadata, k_summary, selected_k, profile, db_info, pca_ratio),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(OUTPUT_DIR.resolve()), "selected_k": int(selected_k), "dbscan": db_info}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
