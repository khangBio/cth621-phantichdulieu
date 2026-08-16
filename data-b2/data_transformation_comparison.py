from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


INPUT_PATH = Path("all_stocks_5yr.csv")
OUTPUT_DIR = Path("outputs/data_transformation_comparison")
NUMERIC_COLS = ["open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]
PNG_SCALE_FACTOR = 1.6


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = font(28, True)
FONT = font(20)
FONT_SMALL = font(16)
FONT_TINY = font(13)
BLUE = (48, 112, 188)
RED = (207, 73, 73)
GREEN = (65, 150, 95)
ORANGE = (222, 145, 48)
GRAY = (90, 90, 90)
GRID = (230, 230, 230)


def canvas(w=1600, h=950):
    img = Image.new("RGB", (w, h), "white")
    return img, ImageDraw.Draw(img)


def text(draw, xy, s, fnt=FONT, fill=(30, 30, 30)):
    draw.text(xy, str(s), font=fnt, fill=fill)


def upscale_png(path: Path, scale: float = PNG_SCALE_FACTOR) -> None:
    """Phóng to PNG cuối để dễ đọc khi đưa vào báo cáo Word/PDF."""
    img = Image.open(path)
    new_size = (int(img.width * scale), int(img.height * scale))
    img.resize(new_size, Image.Resampling.LANCZOS).save(path)


def upscale_output_charts(output_dir: Path) -> None:
    for path in sorted(output_dir.glob("*.png")):
        upscale_png(path)


def minmax(values, pad=0.06):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0, 1
    lo, hi = float(arr.min()), float(arr.max())
    if math.isclose(lo, hi):
        p = abs(lo) * 0.05 + 1
        return lo - p, hi + p
    p = (hi - lo) * pad
    return lo - p, hi + p


def draw_axes(draw, box, xlab, ylab, yticks, yfmt="{:.1f}"):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(80, 80, 80), width=1)
    for v, y in yticks:
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        text(draw, (max(4, x0 - 88), y - 9), yfmt.format(v), FONT_TINY, GRAY)
    text(draw, ((x0 + x1) // 2 - 55, y1 + 38), xlab, FONT_SMALL, GRAY)
    text(draw, (max(4, x0 - 88), y0 - 35), ylab, FONT_SMALL, GRAY)


def plot_missing_bar(path: Path, missing_cells: pd.Series, missing_timestamps: int, full_rows: int):
    img, d = canvas(1500, 850)
    text(d, (55, 35), "Trạng thái dữ liệu thô: số ô khuyết thiếu và missing timestamps", FONT_TITLE)
    text(d, (55, 75), "Trục X: cột dữ liệu; Trục Y: số ô Null/NaN. Ô cuối là số cặp (Name, date) bị thiếu khi đối chiếu lịch giao dịch chung.", FONT_SMALL, GRAY)
    labels = list(missing_cells.index) + ["missing\ntimestamps"]
    vals = list(missing_cells.astype(int).values) + [int(missing_timestamps)]
    x0, y0, x1, y1 = 105, 145, 1420, 690
    ymax = max(vals) * 1.18 if max(vals) else 1
    draw_axes(d, (x0, y0, x1, y1), "Cột / loại thiếu", "Số lượng", [(v, y1 - v / ymax * (y1 - y0)) for v in np.linspace(0, ymax, 6)], "{:,.0f}")
    bw = (x1 - x0) / len(vals) * 0.58
    for i, (lab, val) in enumerate(zip(labels, vals)):
        cx = x0 + (i + 0.5) * (x1 - x0) / len(vals)
        top = y1 - val / ymax * (y1 - y0)
        color = RED if val > 0 else GREEN
        d.rectangle((cx - bw / 2, top, cx + bw / 2, y1), fill=color)
        text(d, (int(cx - bw / 2), int(top - 26)), f"{val:,}", FONT_SMALL, color)
        parts = str(lab).split("\n")
        for j, part in enumerate(parts):
            text(d, (int(cx - 42), y1 + 18 + j * 18), part, FONT_TINY, GRAY)
    text(d, (105, 765), f"Tổng số dòng nếu mỗi mã có đủ lịch giao dịch chung: {full_rows:,}. Missing timestamp là dòng bị thiếu theo cặp mã cổ phiếu-ngày giao dịch.", FONT_SMALL, GRAY)
    img.save(path)


def plot_line_gap(path: Path, raw: pd.DataFrame, filled: pd.DataFrame, ticker: str, col: str):
    raw_t = raw[raw["Name"] == ticker].sort_values("date").copy()
    filled_t = filled[filled["Name"] == ticker].sort_values("date").copy()
    # zoom quanh missing trực tiếp nếu có, nếu không lấy toàn chuỗi.
    miss_dates = raw_t.loc[raw_t[col].isna(), "date"].sort_values()
    if len(miss_dates):
        center = miss_dates.iloc[0]
        start, end = center - pd.Timedelta(days=25), center + pd.Timedelta(days=25)
        raw_t = raw_t[(raw_t["date"] >= start) & (raw_t["date"] <= end)]
        filled_t = filled_t[(filled_t["date"] >= start) & (filled_t["date"] <= end)]
    else:
        raw_t = raw_t.tail(80)
        filled_t = filled_t[filled_t["date"].isin(raw_t["date"])]

    img, d = canvas(1600, 900)
    text(d, (70, 35), f"Line Chart: dữ liệu thô có đứt gãy và dữ liệu sau điền khuyết ({ticker} - {col})", FONT_TITLE)
    text(d, (70, 75), "Trục X: thời gian; Trục Y: giá trị. Đường đỏ là dữ liệu thô, đường xanh là dữ liệu sau interpolation + ffill/bfill.", FONT_SMALL, GRAY)
    box = (110, 145, 1500, 710)
    all_vals = pd.concat([raw_t[col], filled_t[col]]).dropna().to_numpy()
    ymin, ymax = minmax(all_vals)
    dates = sorted(set(filled_t["date"]))
    pos = {dt: i for i, dt in enumerate(dates)}

    def sx(dt):
        return box[0] + pos[dt] / max(1, len(dates) - 1) * (box[2] - box[0])

    def sy(v):
        return box[3] - (v - ymin) / (ymax - ymin) * (box[3] - box[1])

    draw_axes(d, box, "Thời gian", col, [(v, sy(v)) for v in np.linspace(ymin, ymax, 6)], "{:.2f}")
    for i in np.linspace(0, len(dates) - 1, 6).astype(int):
        dt = dates[i]
        text(d, (int(sx(dt)) - 42, box[3] + 18), str(pd.Timestamp(dt).date()), FONT_TINY, GRAY)
    for df_line, color, width in [(filled_t, BLUE, 3), (raw_t, RED, 4)]:
        pts = []
        last = None
        for _, r in df_line.iterrows():
            if pd.isna(r[col]):
                if len(pts) > 1:
                    d.line(pts, fill=color, width=width)
                pts = []
                last = r["date"]
                continue
            p = (sx(r["date"]), sy(float(r[col])))
            pts.append(p)
            last = r["date"]
        if len(pts) > 1:
            d.line(pts, fill=color, width=width)
    for md in miss_dates:
        if md in pos:
            x = sx(md)
            d.line((x, box[1], x, box[3]), fill=ORANGE, width=2)
            text(d, (int(x) + 5, box[1] + 10), "NaN", FONT_TINY, ORANGE)
    d.rectangle((110, 770, 130, 784), fill=RED)
    text(d, (140, 764), "Raw", FONT_SMALL)
    d.rectangle((230, 770, 250, 784), fill=BLUE)
    text(d, (260, 764), "Sau xử lý", FONT_SMALL)
    img.save(path)


def iqr_bounds(s: pd.Series):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1, q3, q1 - 1.5 * iqr, q3 + 1.5 * iqr


def plot_boxplots(path: Path, df: pd.DataFrame, cols: list[str], title: str):
    img, d = canvas(1700, 1250)
    text(d, (60, 35), title, FONT_TITLE)
    text(
        d,
        (60, 75),
        "Tách 2 panel để boxplot rõ hơn: nhóm giá OHLC dùng thang Y riêng, volume dùng thang Y riêng. Điểm đỏ là outlier IQR tiêu biểu.",
        FONT_SMALL,
        GRAY,
    )

    transformed = {c: np.log10(df[c].dropna().astype(float).clip(lower=1e-9)) for c in cols}

    def draw_panel(panel_box, panel_cols, panel_title, x_axis_title=False):
        x0, y0, x1, y1 = panel_box
        values = np.concatenate([transformed[c].to_numpy() for c in panel_cols])
        # Loại các cực trị ngoài 0.1%-99.9% khỏi miền hiển thị để phần box không bị nén quá nhỏ;
        # outlier cực trị vẫn được đếm và vẽ ở sát biên.
        ylo, yhi = np.quantile(values[np.isfinite(values)], [0.001, 0.999])
        ymin, ymax = minmax(np.array([ylo, yhi]), pad=0.08)

        def sy(v):
            vv = min(max(float(v), ymin), ymax)
            return y1 - (vv - ymin) / (ymax - ymin) * (y1 - y0)

        d.rectangle(panel_box, outline=(80, 80, 80), width=1)
        text(d, (x0, y0 - 42), panel_title, FONT, (35, 35, 35))
        for tick in np.linspace(ymin, ymax, 5):
            y = sy(tick)
            d.line((x0, y, x1, y), fill=GRID, width=1)
            text(d, (x0 - 88, y - 9), f"{tick:.2f}", FONT_TINY, GRAY)
        text(d, (x0 - 88, y0 - 22), "log10(value)", FONT_SMALL, GRAY)

        step = (x1 - x0) / len(panel_cols)
        box_w = min(145, step * 0.46)
        for i, c in enumerate(panel_cols):
            x = x0 + (i + 0.5) * step
            s = transformed[c]
            q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
            iqr = q3 - q1
            low_bound, high_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            lo = max(s[s >= low_bound].min() if (s >= low_bound).any() else s.min(), ymin)
            hi = min(s[s <= high_bound].max() if (s <= high_bound).any() else s.max(), ymax)
            d.line((x, sy(lo), x, sy(hi)), fill=(50, 50, 50), width=3)
            d.line((x - box_w * 0.28, sy(lo), x + box_w * 0.28, sy(lo)), fill=(50, 50, 50), width=3)
            d.line((x - box_w * 0.28, sy(hi), x + box_w * 0.28, sy(hi)), fill=(50, 50, 50), width=3)
            d.rectangle((x - box_w / 2, sy(q3), x + box_w / 2, sy(q1)), fill=(210, 228, 248), outline=BLUE, width=3)
            d.line((x - box_w / 2, sy(med), x + box_w / 2, sy(med)), fill=(30, 30, 30), width=4)
            out = s[(s < low_bound) | (s > high_bound)]
            sample = pd.concat([out.nsmallest(10), out.nlargest(10)]).drop_duplicates()
            for v in sample:
                y = sy(v)
                d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=RED)
            text(d, (int(x - 32), y1 + 24), c, FONT, GRAY)
            text(d, (int(x - 58), y1 + 56), f"{len(out):,} outlier", FONT_SMALL, RED)
        if x_axis_title:
            text(d, ((x0 + x1) // 2 - 65, y1 + 95), "Biến dữ liệu", FONT, GRAY)

    draw_panel((130, 170, 1600, 560), [c for c in cols if c != "volume"], "Panel A - Nhóm giá OHLC")
    draw_panel((130, 705, 1600, 940), ["volume"], "Panel B - Khối lượng giao dịch", x_axis_title=True)
    text(d, (130, 1115), "Ghi chú: các điểm outlier rất cực trị được ghim sát biên hiển thị để giữ boxplot đủ lớn và dễ đọc.", FONT_SMALL, GRAY)
    img.save(path)


def plot_hist_compare(path: Path, raw: pd.DataFrame, transformed: pd.DataFrame):
    img, d = canvas(1600, 900)
    text(d, (60, 35), "Histogram so sánh: volume thô và log_volume sau biến đổi", FONT_TITLE)
    text(d, (60, 75), "Trục X: giá trị; Trục Y: tần suất. Log-transform làm phân phối bớt lệch phải và dễ học hơn.", FONT_SMALL, GRAY)
    panels = [((90, 150, 730, 710), raw["volume"].dropna().astype(float), "Raw volume", BLUE), ((870, 150, 1510, 710), transformed["log_volume"].dropna().astype(float), "log1p(volume)", GREEN)]
    for box, series, title, color in panels:
        vals = series.to_numpy()
        lo, hi = np.quantile(vals, [0.001, 0.999])
        vals = vals[(vals >= lo) & (vals <= hi)]
        counts, edges = np.histogram(vals, bins=38)
        ymax = counts.max() * 1.12
        draw_axes(d, box, title, "Tần suất", [(v, box[3] - v / ymax * (box[3] - box[1])) for v in np.linspace(0, ymax, 5)], "{:,.0f}")
        for i, cnt in enumerate(counts):
            x0 = box[0] + i / len(counts) * (box[2] - box[0])
            x1 = box[0] + (i + 1) / len(counts) * (box[2] - box[0])
            y = box[3] - cnt / ymax * (box[3] - box[1])
            d.rectangle((x0 + 1, y, x1 - 1, box[3]), fill=color)
        for j, v in enumerate(np.linspace(lo, hi, 4)):
            x = box[0] + (v - lo) / (hi - lo) * (box[2] - box[0])
            label = f"{v/1e6:.1f}M" if title == "Raw volume" else f"{v:.1f}"
            text(d, (int(x) - 25, box[3] + 18), label, FONT_TINY, GRAY)
    img.save(path)


def plot_feature_engineering(path: Path, df: pd.DataFrame, ticker: str):
    t = df[df["Name"] == ticker].sort_values("date").tail(260)
    img, d = canvas(1650, 1050)
    text(d, (60, 35), f"Chuỗi sau biến đổi: differencing, lag và rolling features ({ticker})", FONT_TITLE)
    text(d, (60, 75), "Ô trên: close và rolling mean 20 phiên. Ô dưới: daily log return và rolling volatility 20 phiên.", FONT_SMALL, GRAY)
    panels = [((110, 140, 1540, 490), "level"), ((110, 610, 1540, 925), "return")]
    dates = t["date"].tolist()

    def sx(dt, box):
        return box[0] + dates.index(dt) / max(1, len(dates) - 1) * (box[2] - box[0])

    # level panel
    box, _ = panels[0]
    vals = pd.concat([t["close_filled"], t["rolling_close_mean_20"]]).dropna()
    ymin, ymax = minmax(vals)
    sy = lambda v: box[3] - (v - ymin) / (ymax - ymin) * (box[3] - box[1])
    draw_axes(d, box, "Thời gian", "Giá close", [(v, sy(v)) for v in np.linspace(ymin, ymax, 5)], "{:.1f}")
    for col, color, width in [("close_filled", BLUE, 2), ("rolling_close_mean_20", ORANGE, 4)]:
        pts = [(box[0] + i / max(1, len(t) - 1) * (box[2] - box[0]), sy(v)) for i, v in enumerate(t[col]) if pd.notna(v)]
        if len(pts) > 1:
            d.line(pts, fill=color, width=width)
    d.rectangle((115, 510, 135, 524), fill=BLUE)
    text(d, (145, 504), "close sau fill", FONT_SMALL)
    d.rectangle((315, 510, 335, 524), fill=ORANGE)
    text(d, (345, 504), "rolling mean 20", FONT_SMALL)

    # return panel
    box, _ = panels[1]
    vals = pd.concat([t["log_return_close"], t["rolling_volatility_20"]]).dropna()
    ymin, ymax = minmax(vals)
    sy = lambda v: box[3] - (v - ymin) / (ymax - ymin) * (box[3] - box[1])
    draw_axes(d, box, "Thời gian", "Log-return / volatility", [(v, sy(v)) for v in np.linspace(ymin, ymax, 5)], "{:.3f}")
    zero = sy(0)
    d.line((box[0], zero, box[2], zero), fill=(120, 120, 120), width=1)
    bar_w = max(2, (box[2] - box[0]) / len(t) * 0.7)
    for i, v in enumerate(t["log_return_close"]):
        if pd.notna(v):
            x = box[0] + i / max(1, len(t) - 1) * (box[2] - box[0])
            color = GREEN if v >= 0 else RED
            d.rectangle((x - bar_w / 2, min(zero, sy(v)), x + bar_w / 2, max(zero, sy(v))), fill=color)
    pts = [(box[0] + i / max(1, len(t) - 1) * (box[2] - box[0]), sy(v)) for i, v in enumerate(t["rolling_volatility_20"]) if pd.notna(v)]
    if len(pts) > 1:
        d.line(pts, fill=ORANGE, width=3)
    d.rectangle((115, 945, 135, 959), fill=GREEN)
    text(d, (145, 939), "log-return dương", FONT_SMALL)
    d.rectangle((330, 945, 350, 959), fill=RED)
    text(d, (360, 939), "log-return âm", FONT_SMALL)
    d.rectangle((525, 945, 545, 959), fill=ORANGE)
    text(d, (555, 939), "rolling volatility 20", FONT_SMALL)
    img.save(path)


def plot_value_domain_dispersion(path: Path, df: pd.DataFrame, cols: list[str]):
    """Biểu đồ miền trị/phân tán của từng thuộc tính bằng min-Q1-median-Q3-max.

    Dùng log10 để 5 thuộc tính khác thang đo vẫn có thể so sánh trên cùng trục.
    """
    img, d = canvas(1800, 1050)
    text(d, (70, 35), "Phân tán miền trị của 5 thuộc tính được chọn", FONT_TITLE)
    text(
        d,
        (70, 78),
        "Trục X: log10(value); Trục Y: thuộc tính. Đường xám = min–max, hộp xanh = Q1–Q3, chấm đen = median.",
        FONT_SMALL,
        GRAY,
    )
    plot = (230, 165, 1640, 770)
    stats = []
    transformed = {}
    for col in cols:
        raw_original = df[col].dropna().astype(float)
        raw = raw_original.clip(lower=1e-9)
        logv = np.log10(raw)
        transformed[col] = logv
        q = logv.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
        rawq = raw_original.quantile([0, 0.25, 0.5, 0.75, 1.0])
        stats.append(
            {
                "variable": col,
                "min": rawq.loc[0.0],
                "q1": rawq.loc[0.25],
                "median": rawq.loc[0.5],
                "q3": rawq.loc[0.75],
                "max": rawq.loc[1.0],
                "log_min": q.loc[0.0],
                "log_p01": q.loc[0.01],
                "log_q1": q.loc[0.25],
                "log_median": q.loc[0.5],
                "log_q3": q.loc[0.75],
                "log_p99": q.loc[0.99],
                "log_max": q.loc[1.0],
            }
        )
    stat_df = pd.DataFrame(stats)
    xmin, xmax = minmax(stat_df[["log_min", "log_max"]].to_numpy().ravel(), pad=0.08)

    def sx(v):
        return plot[0] + (float(v) - xmin) / (xmax - xmin) * (plot[2] - plot[0])

    d.rectangle(plot, outline=(80, 80, 80), width=1)
    for tick in np.linspace(math.floor(xmin), math.ceil(xmax), 8):
        x = sx(tick)
        d.line((x, plot[1], x, plot[3]), fill=GRID, width=1)
        text(d, (int(x) - 18, plot[3] + 22), f"{tick:.1f}", FONT_TINY, GRAY)
    text(d, ((plot[0] + plot[2]) // 2 - 70, plot[3] + 66), "log10(value)", FONT, GRAY)
    text(d, (25, plot[1] - 32), "Thuộc tính", FONT, GRAY)

    row_gap = (plot[3] - plot[1]) / len(cols)
    for i, col in enumerate(cols):
        r = stat_df.loc[stat_df["variable"] == col].iloc[0]
        y = plot[1] + (i + 0.5) * row_gap
        # Min–max line and capped robust 1%–99% line.
        d.line((sx(r["log_min"]), y, sx(r["log_max"]), y), fill=(185, 185, 185), width=4)
        d.line((sx(r["log_p01"]), y, sx(r["log_p99"]), y), fill=(80, 80, 80), width=6)
        # IQR box.
        d.rectangle((sx(r["log_q1"]), y - 28, sx(r["log_q3"]), y + 28), fill=(210, 228, 248), outline=BLUE, width=3)
        # Median marker.
        d.ellipse((sx(r["log_median"]) - 9, y - 9, sx(r["log_median"]) + 9, y + 9), fill=(25, 25, 25))
        # End markers.
        d.ellipse((sx(r["log_min"]) - 6, y - 6, sx(r["log_min"]) + 6, y + 6), fill=RED)
        d.ellipse((sx(r["log_max"]) - 6, y - 6, sx(r["log_max"]) + 6, y + 6), fill=RED)
        text(d, (55, int(y) - 14), col, FONT, (35, 35, 35))
        min_label = "min 0" if float(r["min"]) == 0 else f"min {r['min']:,.2g}"
        text(d, (int(sx(r["log_min"])) - 20, int(y) + 38), min_label, FONT_TINY, RED)
        text(d, (int(sx(r["log_median"])) - 42, int(y) - 58), f"median {r['median']:,.2f}", FONT_TINY, (30, 30, 30))
        text(d, (min(int(sx(r["log_max"])) - 55, plot[2] - 130), int(y) + 38), f"max {r['max']:,.2g}", FONT_TINY, RED)

    # Legend.
    legend_y = 910
    d.line((230, legend_y, 330, legend_y), fill=(185, 185, 185), width=4)
    text(d, (345, legend_y - 12), "Min–Max", FONT_SMALL, GRAY)
    d.line((500, legend_y, 600, legend_y), fill=(80, 80, 80), width=6)
    text(d, (615, legend_y - 12), "P1–P99", FONT_SMALL, GRAY)
    d.rectangle((770, legend_y - 18, 870, legend_y + 18), fill=(210, 228, 248), outline=BLUE, width=3)
    text(d, (890, legend_y - 12), "Q1–Q3 (IQR)", FONT_SMALL, GRAY)
    d.ellipse((1085, legend_y - 9, 1103, legend_y + 9), fill=(25, 25, 25))
    text(d, (1120, legend_y - 12), "Median", FONT_SMALL, GRAY)
    d.ellipse((1295, legend_y - 7, 1309, legend_y + 7), fill=RED)
    text(d, (1325, legend_y - 12), "Min/Max cực trị", FONT_SMALL, GRAY)
    text(
        d,
        (230, 990),
        "Nhận xét nhanh: volume có miền trị rộng nhất; nhóm giá OHLC có IQR khá giống nhau do cùng phản ánh mặt bằng giá trong phiên.",
        FONT_SMALL,
        GRAY,
    )
    img.save(path)
    return stat_df


def direct_missing_timestamp_summary(df: pd.DataFrame):
    dates = pd.Index(sorted(df["date"].dropna().unique()))
    names = pd.Index(sorted(df["Name"].dropna().unique()))
    full_index = pd.MultiIndex.from_product([names, dates], names=["Name", "date"])
    current_index = pd.MultiIndex.from_frame(df[["Name", "date"]].drop_duplicates())
    missing_pairs = full_index.difference(current_index)
    return dates, names, full_index, missing_pairs


def transform_data(raw: pd.DataFrame, full_index: pd.MultiIndex) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "Name"])
    df = df.drop_duplicates(["Name", "date"], keep="last")
    panel = df.set_index(["Name", "date"]).reindex(full_index).reset_index()
    panel = panel.sort_values(["Name", "date"], kind="stable")
    panel["was_missing_timestamp"] = panel[NUMERIC_COLS].isna().all(axis=1)
    for col in NUMERIC_COLS:
        panel[f"{col}_raw"] = panel[col]
        panel[col] = (
            panel.groupby("Name", group_keys=False)[col]
            .apply(lambda s: s.interpolate(method="linear", limit_direction="both").ffill().bfill())
        )
    panel = panel[(panel[PRICE_COLS] > 0).all(axis=1) & (panel["volume"] >= 0)].copy()
    for col in NUMERIC_COLS:
        target = f"{col}_filled"
        panel[target] = panel[col]
    for col in PRICE_COLS:
        panel[f"log_{col}"] = np.log(panel[f"{col}_filled"])
    panel["log_volume"] = np.log1p(panel["volume_filled"])
    for col in PRICE_COLS:
        panel[f"log_return_{col}"] = panel.groupby("Name")[f"log_{col}"].diff()
        panel[f"{col}_lag1"] = panel.groupby("Name")[f"{col}_filled"].shift(1)
    panel["volume_lag1"] = panel.groupby("Name")["volume_filled"].shift(1)
    panel["rolling_close_mean_20"] = panel.groupby("Name")["close_filled"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    panel["rolling_volatility_20"] = panel.groupby("Name")["log_return_close"].transform(lambda s: s.rolling(20, min_periods=5).std())
    panel["rolling_volume_mean_20"] = panel.groupby("Name")["volume_filled"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    return panel


def stats_table(raw: pd.DataFrame, transformed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_COLS:
        r = raw[col].dropna().astype(float)
        f = transformed[f"{col}_filled"].dropna().astype(float)
        log_col = f"log_{col}" if col != "volume" else "log_volume"
        lg = transformed[log_col].dropna().astype(float)
        rows.append(
            {
                "variable": col,
                "raw_count": len(r),
                "raw_missing": int(raw[col].isna().sum()),
                "raw_mean": r.mean(),
                "raw_std": r.std(ddof=1),
                "raw_skew": r.skew(),
                "filled_count": len(f),
                "filled_missing": int(transformed[f"{col}_filled"].isna().sum()),
                "filled_mean": f.mean(),
                "filled_std": f.std(ddof=1),
                "log_std": lg.std(ddof=1),
                "log_skew": lg.skew(),
            }
        )
    return pd.DataFrame(rows)


def outlier_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = []
    examples = []
    for col in NUMERIC_COLS:
        s = raw[col].dropna().astype(float)
        q1, q3, lo, hi = iqr_bounds(s)
        mask = raw[col].notna() & ((raw[col] < lo) | (raw[col] > hi))
        out = raw.loc[mask, ["date", "Name", col]].copy()
        out["variable"] = col
        out["iqr_lower"] = lo
        out["iqr_upper"] = hi
        summary.append({"variable": col, "q1": q1, "q3": q3, "iqr_lower": lo, "iqr_upper": hi, "outlier_count": int(mask.sum()), "outlier_pct": float(mask.mean())})
        if len(out):
            out["distance_to_bound"] = np.where(out[col] > hi, out[col] - hi, lo - out[col])
            top = out.sort_values("distance_to_bound", ascending=False).head(10)
            top = top.rename(columns={col: "value"})
            examples.append(top[["variable", "date", "Name", "value", "iqr_lower", "iqr_upper", "distance_to_bound"]])
    return pd.DataFrame(summary), pd.concat(examples, ignore_index=True)


def markdown_report(missing_cells, missing_pairs, out_sum, out_examples, stats, transformed):
    n_missing_cells = int(missing_cells.sum())
    rows = [
        "# So sánh dữ liệu thô và dữ liệu sau tiền xử lý / biến đổi",
        "",
        "## 1. Trạng thái dữ liệu thô",
        "",
        f"- Tổng số dòng gốc: **619,040**.",
        f"- Tổng số ô khuyết thiếu trực tiếp ở các cột dữ liệu: **{n_missing_cells:,}**.",
        f"- Nếu đối chiếu theo lịch giao dịch chung của toàn bộ mã cổ phiếu, có **{len(missing_pairs):,}** cặp `(Name, date)` bị thiếu, gọi là missing timestamps.",
        "- Các ngày cuối tuần/ngày nghỉ thị trường không được tạo thêm; lịch đối chiếu chỉ lấy từ các ngày giao dịch đã xuất hiện trong dataset.",
        "",
        "### Số ô Null/NaN trực tiếp",
        "",
        "| Cột | Số ô thiếu |",
        "|---|---:|",
    ]
    for col, val in missing_cells.items():
        rows.append(f"| {col} | {int(val):,} |")
    rows += [
        "",
        "### Ngoại lai theo quy tắc IQR",
        "",
        "| Biến | Số outlier | Tỷ lệ | Cận dưới IQR | Cận trên IQR |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in out_sum.iterrows():
        rows.append(f"| {r['variable']} | {int(r['outlier_count']):,} | {r['outlier_pct']:.2%} | {r['iqr_lower']:,.2f} | {r['iqr_upper']:,.2f} |")
    rows += [
        "",
        "Một số điểm ngoại lai cụ thể mạnh nhất:",
        "",
        "| Biến | Date | Name | Giá trị | Cận trên IQR |",
        "|---|---|---|---:|---:|",
    ]
    for _, r in out_examples.head(15).iterrows():
        rows.append(f"| {r['variable']} | {pd.Timestamp(r['date']).date()} | {r['Name']} | {r['value']:,.2f} | {r['iqr_upper']:,.2f} |")
    rows += [
        "",
        "## 2. Trạng thái dữ liệu sau xử lý khuyết và biến đổi",
        "",
        "Quy trình xử lý gồm:",
        "",
        "1. Chuyển `date` sang kiểu thời gian và sắp xếp theo `Name`, `date`.",
        "2. Reindex từng mã cổ phiếu theo lịch ngày giao dịch chung để nhận diện missing timestamps.",
        "3. Điền khuyết theo chuỗi thời gian trong từng mã cổ phiếu bằng `linear interpolation`, sau đó dùng `forward fill` và `backward fill` cho các đoạn biên.",
        "4. Tạo log-transform: `log_open`, `log_high`, `log_low`, `log_close`, `log_volume`.",
        "5. Tạo differencing/log-return: `log_return_open`, `log_return_high`, `log_return_low`, `log_return_close`.",
        "6. Tạo lag features: `open_lag1`, `high_lag1`, `low_lag1`, `close_lag1`, `volume_lag1`.",
        "7. Tạo rolling features: `rolling_close_mean_20`, `rolling_volatility_20`, `rolling_volume_mean_20`.",
        "",
        f"- Sau xử lý, số dòng dạng panel là **{len(transformed):,}**.",
        f"- Số dòng được tạo từ missing timestamp: **{int(transformed['was_missing_timestamp'].sum()):,}**.",
        "",
        "## 3. So sánh chỉ số thống kê trước và sau xử lý",
        "",
        "| Biến | Missing raw | Missing sau fill | Std raw | Std sau fill | Std sau log | Skew raw | Skew sau log |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in stats.iterrows():
        rows.append(
            f"| {r['variable']} | {int(r['raw_missing']):,} | {int(r['filled_missing']):,} | "
            f"{r['raw_std']:,.4f} | {r['filled_std']:,.4f} | {r['log_std']:,.4f} | {r['raw_skew']:,.4f} | {r['log_skew']:,.4f} |"
        )
    rows += [
        "",
        "## 4. Nhận xét so sánh",
        "",
        "- Dữ liệu thô có ít ô Null trực tiếp, nhưng vẫn tồn tại missing timestamps khi xét theo cấu trúc panel `(Name, date)`.",
        "- Boxplot của dữ liệu thô cho thấy nhiều ngoại lai, đặc biệt ở `volume` và các biến giá. Đây là đặc trưng bình thường của dữ liệu chứng khoán nhưng có thể làm thuật toán dựa trên khoảng cách bị lệch.",
        "- Sau khi điền khuyết, các chuỗi không còn đoạn trống trong lịch giao dịch chung, giúp dữ liệu sẵn sàng hơn cho mô hình học máy.",
        "- Log-transform làm phân phối `volume` và giá bớt lệch phải; histogram sau log mượt và dễ đọc hơn histogram raw.",
        "- Differencing/log-return giúp chuyển chuỗi giá từ dạng có xu hướng sang dạng biến động quanh 0, phù hợp hơn cho phân tích volatility và các mô hình nhạy với tính dừng.",
        "- Lag và rolling features bổ sung thông tin quá khứ gần, giúp mô hình học được động lượng, thanh khoản gần đây và mức rủi ro biến động.",
        "",
        "## 5. Biểu đồ đầu ra",
        "",
        "1. `01_raw_missing_values.png`: thống kê Null/NaN và missing timestamps.",
        "2. `02_raw_line_breaks_vs_filled.png`: đoạn đứt gãy trên line chart và chuỗi sau điền khuyết.",
        "3. `03_raw_boxplot_outliers.png`: boxplot dữ liệu thô và số outlier IQR.",
        "4. `04_hist_volume_raw_vs_log.png`: histogram volume thô và log-volume.",
        "5. `05_transformed_features_timeseries.png`: differencing/log-return, lag/rolling features.",
        "6. `06_value_domain_dispersion.png`: phân tán miền trị của 5 thuộc tính `open`, `high`, `low`, `close`, `volume`.",
    ]
    return "\n".join(rows) + "\n"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    raw = raw.sort_values(["Name", "date"], kind="stable")
    missing_cells = raw.isna().sum()
    dates, names, full_index, missing_pairs = direct_missing_timestamp_summary(raw)
    transformed = transform_data(raw, full_index)
    stats = stats_table(raw, transformed)
    out_sum, out_examples = outlier_table(raw)

    missing_cells.to_csv(OUTPUT_DIR / "raw_missing_cells.csv", header=["missing_count"])
    pd.DataFrame({"missing_timestamps": [len(missing_pairs)], "full_panel_rows": [len(full_index)], "raw_rows": [len(raw)]}).to_csv(OUTPUT_DIR / "raw_missing_timestamps.csv", index=False)
    out_sum.to_csv(OUTPUT_DIR / "raw_outlier_summary_iqr.csv", index=False)
    out_examples.to_csv(OUTPUT_DIR / "raw_outlier_examples_iqr.csv", index=False)
    stats.to_csv(OUTPUT_DIR / "statistics_raw_vs_transformed.csv", index=False)
    transformed.to_csv(OUTPUT_DIR / "sp500_transformed_ml_ready.csv", index=False)

    # Chọn mã/cột có NaN trực tiếp để biểu đồ line có đoạn đứt gãy thật.
    miss_rows = raw[raw[NUMERIC_COLS].isna().any(axis=1)]
    if len(miss_rows):
        first = miss_rows.iloc[0]
        ticker = first["Name"]
        col = next(c for c in NUMERIC_COLS if pd.isna(first[c]))
    else:
        counts = pd.Series(missing_pairs.get_level_values("Name")).value_counts()
        ticker = counts.index[0]
        col = "close"

    filled_view = transformed[["date", "Name"] + [f"{c}_filled" for c in NUMERIC_COLS]].rename(
        columns={f"{c}_filled": c for c in NUMERIC_COLS}
    )
    plot_missing_bar(OUTPUT_DIR / "01_raw_missing_values.png", missing_cells, len(missing_pairs), len(full_index))
    plot_line_gap(OUTPUT_DIR / "02_raw_line_breaks_vs_filled.png", raw, filled_view, ticker, col)
    plot_boxplots(OUTPUT_DIR / "03_raw_boxplot_outliers.png", raw, NUMERIC_COLS, "Boxplot dữ liệu thô: nhận diện outlier theo IQR")
    plot_hist_compare(OUTPUT_DIR / "04_hist_volume_raw_vs_log.png", raw, transformed)
    ticker_roll = "AAPL" if "AAPL" in set(transformed["Name"]) else transformed["Name"].iloc[0]
    plot_feature_engineering(OUTPUT_DIR / "05_transformed_features_timeseries.png", transformed, ticker_roll)
    domain_stats = plot_value_domain_dispersion(OUTPUT_DIR / "06_value_domain_dispersion.png", raw, NUMERIC_COLS)
    domain_stats.to_csv(OUTPUT_DIR / "value_domain_dispersion_stats.csv", index=False)
    upscale_output_charts(OUTPUT_DIR)

    metadata = {
        "input_path": str(INPUT_PATH.resolve()),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "raw_rows": int(len(raw)),
        "full_panel_rows": int(len(full_index)),
        "transformed_rows": int(len(transformed)),
        "direct_missing_cells": {k: int(v) for k, v in missing_cells.items()},
        "missing_timestamps": int(len(missing_pairs)),
        "line_gap_example": {"Name": str(ticker), "column": str(col)},
        "transformation_methods": ["linear interpolation by Name", "forward fill", "backward fill", "log-transform", "log differencing", "lag1 features", "rolling 20 features"],
        "png_scale_factor": PNG_SCALE_FACTOR,
        "value_domain_chart": "06_value_domain_dispersion.png",
    }
    (OUTPUT_DIR / "preprocessing_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "nhan_xet_so_sanh_tien_xu_ly.md").write_text(markdown_report(missing_cells, missing_pairs, out_sum, out_examples, stats, transformed), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
