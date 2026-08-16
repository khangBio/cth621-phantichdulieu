"""Trực quan hóa 5 biến AAPL bằng Python (Pandas, NumPy, Pillow).

Tạo line chart, histogram, biểu đồ mùa vụ đã khử xu hướng, boxplot,
heatmap tương quan, scatter và rolling volatility trong thư mục
outputs/truc_quan_aapl/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


INPUT_FILE = Path("all_stocks_5yr.csv")
OUTPUT_DIR = Path("outputs/truc_quan_aapl")
STOCK_CODE = "AAPL"
VARIABLES = ["open", "high", "low", "volume", "close"]
PRICE_VARIABLES = ["open", "high", "low", "close"]

BG = "#F4F7FB"
PANEL = "#FFFFFF"
INK = "#183153"
MUTED = "#667085"
GRID = "#DDE5EE"
ACCENT = "#0F6CBD"
ORANGE = "#E67E22"
RED = "#C0392B"
GREEN = "#14866D"
PURPLE = "#7A5AF8"
COLORS = {
    "open": "#0F6CBD",
    "high": "#E67E22",
    "low": "#14866D",
    "volume": "#7A5AF8",
    "close": "#C0392B",
}


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


F10 = load_font(18)
F9 = load_font(16)
F8 = load_font(14)
FB = load_font(18, bold=True)
FT = load_font(30, bold=True)
FS = load_font(21, bold=True)


def fmt_number(value: float, volume: bool = False) -> str:
    if volume:
        if abs(value) >= 1e9:
            return f"{value / 1e9:.1f}B"
        if abs(value) >= 1e6:
            return f"{value / 1e6:.0f}M"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.1f}"


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=INK) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text, font=font, fill=fill)


def canvas(title: str, subtitle: str, width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((60, 35), title, font=FT, fill=INK)
    draw.text((60, 78), subtitle, font=F9, fill=MUTED)
    return image, draw


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=16, fill=PANEL, outline="#E3EAF2", width=2)
    draw.text((x0 + 22, y0 + 16), title, font=FS, fill=INK)
    return x0 + 74, y0 + 62, x1 - 28, y1 - 54


def axes(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    ymin: float,
    ymax: float,
    y_volume: bool = False,
    y_percent: bool = False,
    y_ticks: int = 5,
) -> None:
    x0, y0, x1, y1 = plot
    for i in range(y_ticks + 1):
        y = y1 - (y1 - y0) * i / y_ticks
        value = ymin + (ymax - ymin) * i / y_ticks
        draw.line((x0, y, x1, y), fill=GRID, width=1)
        label = f"{value:.1f}%" if y_percent else fmt_number(value, y_volume)
        box = draw.textbbox((0, 0), label, font=F8)
        draw.text((x0 - 10 - (box[2] - box[0]), y - 8), label, font=F8, fill=MUTED)
    draw.line((x0, y0, x0, y1), fill="#AAB7C4", width=2)
    draw.line((x0, y1, x1, y1), fill="#AAB7C4", width=2)


def line_chart(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    dates: pd.Series,
    values: pd.Series,
    color: str,
    moving_average: int | None = 60,
    volume: bool = False,
) -> None:
    clean = values.astype(float)
    ymin, ymax = float(clean.min()), float(clean.max())
    padding = (ymax - ymin) * 0.06 or 1.0
    ymin, ymax = (max(0.0, ymin - padding) if volume else ymin - padding), ymax + padding
    axes(draw, plot, ymin, ymax, y_volume=volume)
    x0, y0, x1, y1 = plot

    def points(series: pd.Series) -> list[tuple[float, float]]:
        result = []
        n = len(series) - 1
        for i, value in enumerate(series):
            if pd.isna(value):
                continue
            x = x0 + (x1 - x0) * i / n
            y = y1 - (float(value) - ymin) / (ymax - ymin) * (y1 - y0)
            result.append((x, y))
        return result

    draw.line(points(clean), fill=color, width=3, joint="curve")
    if moving_average:
        ma = clean.rolling(moving_average, min_periods=max(5, moving_average // 4)).mean()
        draw.line(points(ma), fill="#172B4D", width=4, joint="curve")
        draw.line((x1 - 245, y0 + 10, x1 - 210, y0 + 10), fill=color, width=4)
        draw.text((x1 - 202, y0 + 1), "Daily", font=F8, fill=MUTED)
        draw.line((x1 - 132, y0 + 10, x1 - 97, y0 + 10), fill="#172B4D", width=4)
        draw.text((x1 - 89, y0 + 1), f"MA{moving_average}", font=F8, fill=MUTED)
    tick_count = 6
    for i in range(tick_count):
        idx = round(i * (len(dates) - 1) / (tick_count - 1))
        x = x0 + (x1 - x0) * idx / (len(dates) - 1)
        draw.line((x, y1, x, y1 + 6), fill="#AAB7C4", width=2)
        text_center(draw, (int(x), y1 + 10), pd.Timestamp(dates.iloc[idx]).strftime("%Y-%m"), F8, MUTED)


def histogram(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    values: pd.Series,
    color: str,
    bins: int = 28,
    percent_x: bool = False,
    volume: bool = False,
    clip_quantiles: tuple[float, float] | None = None,
) -> None:
    x = values.dropna().astype(float).to_numpy()
    if clip_quantiles:
        lo, hi = np.quantile(x, clip_quantiles)
        x = x[(x >= lo) & (x <= hi)]
    counts, edges = np.histogram(x, bins=bins)
    x0, y0, x1, y1 = plot
    ymax = max(counts) * 1.1
    axes(draw, plot, 0, ymax, y_ticks=4)
    bar_width = (x1 - x0) / len(counts)
    for i, count in enumerate(counts):
        left = x0 + i * bar_width + 1
        right = x0 + (i + 1) * bar_width - 1
        top = y1 - count / ymax * (y1 - y0)
        draw.rectangle((left, top, right, y1), fill=color)
    xmin, xmax = edges[0], edges[-1]
    for statistic, stat_color, name in [(np.mean(x), RED, "Mean"), (np.median(x), GREEN, "Median")]:
        xx = x0 + (statistic - xmin) / (xmax - xmin) * (x1 - x0)
        draw.line((xx, y0, xx, y1), fill=stat_color, width=3)
        draw.text((xx + 5, y0 + (4 if name == "Mean" else 26)), name, font=F8, fill=stat_color)
    for i in range(5):
        xx = x0 + (x1 - x0) * i / 4
        value = xmin + (xmax - xmin) * i / 4
        label = f"{value:.1f}%" if percent_x else fmt_number(value, volume)
        text_center(draw, (int(xx), y1 + 10), label, F8, MUTED)


def boxplot_one(
    draw: ImageDraw.ImageDraw,
    x: int,
    y0: int,
    y1: int,
    values: pd.Series,
    global_min: float,
    global_max: float,
    color: str,
    label: str,
    width: int = 46,
) -> None:
    s = values.dropna().astype(float)
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    whisker_low = s[s >= lower_fence].min()
    whisker_high = s[s <= upper_fence].max()

    def yy(v: float) -> float:
        return y1 - (v - global_min) / (global_max - global_min) * (y1 - y0)

    draw.line((x, yy(whisker_low), x, yy(whisker_high)), fill=INK, width=3)
    draw.line((x - width // 3, yy(whisker_low), x + width // 3, yy(whisker_low)), fill=INK, width=3)
    draw.line((x - width // 3, yy(whisker_high), x + width // 3, yy(whisker_high)), fill=INK, width=3)
    draw.rectangle((x - width // 2, yy(q3), x + width // 2, yy(q1)), fill=color, outline=INK, width=2)
    draw.line((x - width // 2, yy(med), x + width // 2, yy(med)), fill="#FFFFFF", width=4)
    outliers = s[(s < lower_fence) | (s > upper_fence)]
    for value in outliers.iloc[:: max(1, len(outliers) // 80)]:
        draw.ellipse((x - 3, yy(value) - 3, x + 3, yy(value) + 3), fill=RED)
    text_center(draw, (x, y1 + 12), label, F9, INK)


def heat_color(value: float) -> str:
    value = max(-1.0, min(1.0, value))
    if value >= 0:
        start = np.array([247, 250, 252])
        end = np.array([15, 108, 189])
        rgb = start + value * (end - start)
    else:
        start = np.array([247, 250, 252])
        end = np.array([192, 57, 43])
        rgb = start + (-value) * (end - start)
    return "#" + "".join(f"{int(c):02X}" for c in rgb)


def draw_heatmap(draw, box, corr: pd.DataFrame, title: str) -> None:
    x0, y0, x1, y1 = panel(draw, box, title)
    names = list(corr.columns)
    left, top = x0 + 70, y0 + 12
    size = min((x1 - left) // len(names), (y1 - top) // len(names))
    for i, name in enumerate(names):
        text_center(draw, (left + i * size + size // 2, top - 28), name, F8, MUTED)
        draw.text((x0 - 6, top + i * size + size // 2 - 8), name, font=F8, fill=MUTED)
        for j, name2 in enumerate(names):
            value = float(corr.iloc[i, j])
            cell = (left + j * size, top + i * size, left + (j + 1) * size, top + (i + 1) * size)
            draw.rectangle(cell, fill=heat_color(value), outline="#FFFFFF", width=2)
            text_center(draw, (cell[0] + size // 2, cell[1] + size // 2 - 9), f"{value:.2f}", F9, "#FFFFFF" if abs(value) > 0.55 else INK)


def save(image: Image.Image, name: str) -> None:
    image.save(OUTPUT_DIR / name, format="PNG", optimize=True)


def make_time_series(data: pd.DataFrame) -> None:
    image, draw = canvas(
        "AAPL – Time Series Plot cho 5 biến",
        "Đường màu: dữ liệu ngày | Đường đậm: trung bình trượt 60 phiên | 2013-02 đến 2018-02",
        1900,
        2100,
    )
    boxes = [(45, 125 + i * 390, 1855, 485 + i * 390) for i in range(5)]
    for variable, box in zip(VARIABLES, boxes):
        unit = "khối lượng cổ phiếu" if variable == "volume" else "USD"
        plot = panel(draw, box, f"{variable.upper()} ({unit})")
        line_chart(draw, plot, data["date"], data[variable], COLORS[variable], moving_average=60, volume=variable == "volume")
    save(image, "01_time_series_5_bien.png")


def make_raw_histograms(data: pd.DataFrame) -> None:
    image, draw = canvas(
        "Phân phối giá trị của 5 biến",
        "Histogram của mức giá/khối lượng quan sát; đường đỏ = mean, xanh = median",
        1900,
        1380,
    )
    boxes = [
        (45, 125, 935, 515),
        (965, 125, 1855, 515),
        (45, 545, 935, 935),
        (965, 545, 1855, 935),
        (505, 965, 1395, 1355),
    ]
    for variable, box in zip(VARIABLES, boxes):
        plot = panel(draw, box, f"{variable.upper()}")
        histogram(draw, plot, data[variable], COLORS[variable], volume=variable == "volume")
    save(image, "02_histogram_gia_tri.png")


def make_change_histograms(data: pd.DataFrame) -> pd.DataFrame:
    changes = np.log(data[VARIABLES]).diff() * 100
    image, draw = canvas(
        "Phân phối mức biến động ngày",
        "Daily log change (%); trục x cắt tại P1–P99 để phần trung tâm dễ đọc (ngoại lệ vẫn được thống kê trong báo cáo)",
        1900,
        1380,
    )
    boxes = [
        (45, 125, 935, 515),
        (965, 125, 1855, 515),
        (45, 545, 935, 935),
        (965, 545, 1855, 935),
        (505, 965, 1395, 1355),
    ]
    for variable, box in zip(VARIABLES, boxes):
        plot = panel(draw, box, f"Δ log {variable.upper()} mỗi phiên (%)")
        histogram(draw, plot, changes[variable], COLORS[variable], bins=34, percent_x=True, clip_quantiles=(0.01, 0.99))
    save(image, "03_histogram_bien_dong_ngay.png")
    return changes


def make_seasonality(data: pd.DataFrame) -> pd.DataFrame:
    log_values = np.log(data[VARIABLES])
    trend = log_values.rolling(252, center=True, min_periods=60).mean()
    detrended = (np.exp(log_values - trend) - 1) * 100
    detrended["month"] = data["date"].dt.month
    monthly = detrended.groupby("month")[VARIABLES].mean()
    image, draw = canvas(
        "Mẫu hình mùa vụ theo tháng",
        "",
        1900,
        920,
    )
    price_plot = panel(draw, (45, 125, 1855, 535), "OPEN / HIGH / LOW / CLOSE – sai lệch khỏi xu hướng (%)")
    all_price = monthly[PRICE_VARIABLES].to_numpy()
    ymin, ymax = float(all_price.min()), float(all_price.max())
    pad = (ymax - ymin) * 0.15 or 1
    ymin, ymax = ymin - pad, ymax + pad
    axes(draw, price_plot, ymin, ymax, y_percent=True)
    x0, y0, x1, y1 = price_plot
    for variable in PRICE_VARIABLES:
        pts = []
        for month, value in monthly[variable].items():
            x = x0 + (x1 - x0) * (month - 1) / 11
            y = y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)
            pts.append((x, y))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=COLORS[variable])
        draw.line(pts, fill=COLORS[variable], width=4)
    for month in range(1, 13):
        x = x0 + (x1 - x0) * (month - 1) / 11
        text_center(draw, (int(x), y1 + 10), f"T{month}", F8, MUTED)
    lx = x0 + 20
    for variable in PRICE_VARIABLES:
        draw.line((lx, y0 + 8, lx + 30, y0 + 8), fill=COLORS[variable], width=4)
        draw.text((lx + 36, y0), variable, font=F8, fill=MUTED)
        lx += 125

    volume_plot = panel(draw, (45, 565, 1855, 895), "VOLUME – sai lệch khỏi xu hướng (%)")
    vol = monthly["volume"]
    ymin, ymax = float(vol.min()), float(vol.max())
    pad = (ymax - ymin) * 0.15 or 1
    axes(draw, volume_plot, ymin - pad, ymax + pad, y_percent=True)
    x0, y0, x1, y1 = volume_plot
    pts = []
    for month, value in vol.items():
        x = x0 + (x1 - x0) * (month - 1) / 11
        y = y1 - (value - (ymin - pad)) / ((ymax + pad) - (ymin - pad)) * (y1 - y0)
        pts.append((x, y))
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=PURPLE)
        text_center(draw, (int(x), y1 + 10), f"T{month}", F8, MUTED)
    draw.line(pts, fill=PURPLE, width=4)
    save(image, "04_mua_vu_theo_thang.png")
    return monthly


def make_boxplots(data: pd.DataFrame) -> None:
    image, draw = canvas(
        "Boxplot – trung vị, IQR và ngoại lệ",
        "Râu theo quy tắc 1,5×IQR; chấm đỏ là quan sát ngoài hàng rào",
        1900,
        850,
    )
    price_plot = panel(draw, (45, 125, 1190, 825), "Các biến giá (USD)")
    x0, y0, x1, y1 = price_plot
    pmin = min(float(data[c].min()) for c in PRICE_VARIABLES)
    pmax = max(float(data[c].max()) for c in PRICE_VARIABLES)
    pad = (pmax - pmin) * 0.06
    axes(draw, price_plot, pmin - pad, pmax + pad)
    for i, variable in enumerate(PRICE_VARIABLES):
        x = x0 + (i + 1) * (x1 - x0) / 5
        boxplot_one(draw, int(x), y0, y1, data[variable], pmin - pad, pmax + pad, COLORS[variable], variable)

    vol_plot = panel(draw, (1220, 125, 1855, 825), "VOLUME")
    x0, y0, x1, y1 = vol_plot
    vmin, vmax = float(data["volume"].min()), float(data["volume"].max())
    pad = (vmax - vmin) * 0.04
    axes(draw, vol_plot, vmin - pad, vmax + pad, y_volume=True)
    boxplot_one(draw, (x0 + x1) // 2, y0, y1, data["volume"], vmin - pad, vmax + pad, PURPLE, "volume", width=90)
    save(image, "05_boxplot.png")


def make_correlations(data: pd.DataFrame, changes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_levels = data[VARIABLES].corr()
    corr_changes = changes.corr()
    image, draw = canvas(
        "Heatmap tương quan: mức giá so với biến động ngày",
        "Tương quan ở level có thể bị xu hướng chung thổi phồng; tương quan daily log change phản ánh đồng biến ngắn hạn tốt hơn",
        1900,
        900,
    )
    draw_heatmap(draw, (45, 125, 935, 875), corr_levels, "Tương quan giá trị gốc (level)")
    draw_heatmap(draw, (965, 125, 1855, 875), corr_changes, "Tương quan biến động ngày")
    save(image, "06_heatmap_tuong_quan.png")
    return corr_levels, corr_changes


def make_scatter(data: pd.DataFrame) -> float:
    image, draw = canvas(
        "Mối quan hệ giữa CLOSE và VOLUME",
        "Mỗi chấm là một phiên; màu biểu thị năm để thấy cấu trúc theo thời gian",
        1900,
        930,
    )
    plot = panel(draw, (45, 125, 1855, 900), "Close (USD) vs Volume")
    x0, y0, x1, y1 = plot
    x = data["close"].to_numpy(float)
    y = data["volume"].to_numpy(float)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    axes(draw, plot, ymin, ymax, y_volume=True)
    years = sorted(data["date"].dt.year.unique())
    palette = ["#0F6CBD", "#14866D", "#E67E22", "#7A5AF8", "#C0392B", "#34495E"]
    for xv, yv, year in zip(x, y, data["date"].dt.year):
        xx = x0 + (xv - xmin) / (xmax - xmin) * (x1 - x0)
        yy = y1 - (yv - ymin) / (ymax - ymin) * (y1 - y0)
        color = palette[years.index(year) % len(palette)]
        draw.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), fill=color)
    slope, intercept = np.polyfit(x, y, 1)
    y_start, y_end = intercept + slope * xmin, intercept + slope * xmax
    draw.line((x0, y1 - (y_start - ymin) / (ymax - ymin) * (y1 - y0), x1, y1 - (y_end - ymin) / (ymax - ymin) * (y1 - y0)), fill=INK, width=4)
    for i in range(6):
        xx = x0 + (x1 - x0) * i / 5
        text_center(draw, (int(xx), y1 + 10), f"{xmin + (xmax - xmin) * i / 5:.0f}", F8, MUTED)
    lx = x0 + 25
    for year, color in zip(years, palette):
        draw.ellipse((lx, y0 + 2, lx + 14, y0 + 16), fill=color)
        draw.text((lx + 20, y0), str(year), font=F8, fill=MUTED)
        lx += 105
    corr = float(data["close"].corr(data["volume"]))
    draw.text((x1 - 285, y0), f"Pearson r = {corr:.3f}", font=FB, fill=INK)
    save(image, "07_scatter_close_volume.png")
    return corr


def make_target_relationships(data: pd.DataFrame, changes: pd.DataFrame) -> None:
    """Scatter matrix: close là Y, từng biến giải thích là X."""
    predictors = ["open", "high", "low", "volume"]
    image, draw = canvas(
        "Biến mục tiêu CLOSE và các biến giải thích",
        "Mỗi điểm là một phiên; màu theo năm | r(level) đo quan hệ mức giá, r(Δlog) đo quan hệ biến động ngày",
        1900,
        1400,
    )
    boxes = [
        (45, 125, 935, 735),
        (965, 125, 1855, 735),
        (45, 765, 935, 1375),
        (965, 765, 1855, 1375),
    ]
    years = sorted(data["date"].dt.year.unique())
    palette = ["#0F6CBD", "#14866D", "#E67E22", "#7A5AF8", "#C0392B", "#34495E"]
    y_values = data["close"].to_numpy(float)
    ymin, ymax = float(y_values.min()), float(y_values.max())
    ypad = (ymax - ymin) * 0.05
    ymin, ymax = ymin - ypad, ymax + ypad

    for predictor, box in zip(predictors, boxes):
        r_level = float(data[predictor].corr(data["close"]))
        r_change = float(changes[predictor].corr(changes["close"]))
        plot = panel(
            draw,
            box,
            f"X: {predictor.upper()}  →  Y: CLOSE     r(level)={r_level:.3f} | r(Δlog)={r_change:.3f}",
        )
        x0, y0, x1, y1 = plot
        x_values = data[predictor].to_numpy(float)
        xmin, xmax = float(x_values.min()), float(x_values.max())
        xpad = (xmax - xmin) * 0.04 or 1.0
        xmin = max(0.0, xmin - xpad) if predictor == "volume" else xmin - xpad
        xmax += xpad
        axes(draw, plot, ymin, ymax)

        for xv, yv, year in zip(x_values, y_values, data["date"].dt.year):
            xx = x0 + (xv - xmin) / (xmax - xmin) * (x1 - x0)
            yy = y1 - (yv - ymin) / (ymax - ymin) * (y1 - y0)
            color = palette[years.index(year) % len(palette)]
            draw.ellipse((xx - 2, yy - 2, xx + 2, yy + 2), fill=color)

        slope, intercept = np.polyfit(x_values, y_values, 1)
        reg_start = intercept + slope * xmin
        reg_end = intercept + slope * xmax
        ry0 = y1 - (reg_start - ymin) / (ymax - ymin) * (y1 - y0)
        ry1 = y1 - (reg_end - ymin) / (ymax - ymin) * (y1 - y0)
        ry0 = max(y0, min(y1, ry0))
        ry1 = max(y0, min(y1, ry1))
        draw.line((x0, ry0, x1, ry1), fill=INK, width=4)

        for i in range(5):
            xx = x0 + (x1 - x0) * i / 4
            value = xmin + (xmax - xmin) * i / 4
            text_center(
                draw,
                (int(xx), y1 + 10),
                fmt_number(value, volume=predictor == "volume"),
                F8,
                MUTED,
            )
        draw.text((x0 + 4, y0 + 4), "Close (USD)", font=F8, fill=MUTED)
        draw.text(
            (x1 - 165, y1 + 28),
            f"{predictor} ({'cổ phiếu' if predictor == 'volume' else 'USD'})",
            font=F8,
            fill=MUTED,
        )

    lx, ly = 70, 112
    for year, color in zip(years, palette):
        draw.ellipse((lx, ly, lx + 14, ly + 14), fill=color)
        draw.text((lx + 20, ly - 2), str(year), font=F8, fill=MUTED)
        lx += 105
    save(image, "09_close_vs_bien_giai_thich.png")


def make_rolling_volatility(data: pd.DataFrame, changes: pd.DataFrame) -> dict[str, object]:
    returns = changes["close"] / 100
    annualized = returns.rolling(20).std(ddof=1) * math.sqrt(252) * 100
    image, draw = canvas(
        "CLOSE – biến động ngày và rolling volatility",
        "Biểu đồ bổ sung để nhận diện cụm biến động (volatility clustering) và các phiên bất thường",
        1900,
        930,
    )
    plot1 = panel(draw, (45, 125, 1855, 500), "Daily log return của CLOSE (%)")
    x0, y0, x1, y1 = plot1
    values = changes["close"].fillna(0)
    limit = max(abs(float(values.quantile(0.005))), abs(float(values.quantile(0.995))))
    axes(draw, plot1, -limit, limit, y_percent=True, y_ticks=4)
    zero = y1 - (0 + limit) / (2 * limit) * (y1 - y0)
    for i, value in enumerate(values):
        clipped = max(-limit, min(limit, float(value)))
        xx = x0 + (x1 - x0) * i / (len(values) - 1)
        yy = y1 - (clipped + limit) / (2 * limit) * (y1 - y0)
        draw.line((xx, zero, xx, yy), fill=GREEN if value >= 0 else RED, width=2)

    plot2 = panel(draw, (45, 530, 1855, 900), "Độ lệch chuẩn 20 phiên, quy đổi năm (%)")
    line_chart(draw, plot2, data["date"], annualized.bfill(), PURPLE, moving_average=None)
    max_idx = annualized.idxmax()
    result = {
        "max_rolling_volatility_percent": float(annualized.loc[max_idx]),
        "max_rolling_volatility_date": str(data.loc[max_idx, "date"].date()),
        "daily_log_return_std_percent": float(changes["close"].std(ddof=1)),
        "max_daily_log_return_percent": float(changes["close"].max()),
        "max_daily_log_return_date": str(data.loc[changes["close"].idxmax(), "date"].date()),
        "min_daily_log_return_percent": float(changes["close"].min()),
        "min_daily_log_return_date": str(data.loc[changes["close"].idxmin(), "date"].date()),
    }
    save(image, "08_rolling_volatility_close.png")
    return result


def build_report(
    data: pd.DataFrame,
    changes: pd.DataFrame,
    monthly: pd.DataFrame,
    corr_levels: pd.DataFrame,
    corr_changes: pd.DataFrame,
    close_volume_corr: float,
    volatility: dict[str, object],
) -> str:
    stats = data[VARIABLES].describe().T
    skew = data[VARIABLES].skew()
    price_corr_level = corr_levels.loc["open", "close"]
    price_corr_change = corr_changes.loc["open", "close"]
    outlier_counts = {}
    for variable in VARIABLES:
        q1, q3 = data[variable].quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_counts[variable] = int(((data[variable] < q1 - 1.5 * iqr) | (data[variable] > q3 + 1.5 * iqr)).sum())
    season = {}
    for variable in VARIABLES:
        season[variable] = {
            "highest_month": int(monthly[variable].idxmax()),
            "highest_percent": float(monthly[variable].max()),
            "lowest_month": int(monthly[variable].idxmin()),
            "lowest_percent": float(monthly[variable].min()),
            "amplitude_percent": float(monthly[variable].max() - monthly[variable].min()),
        }
    volume_max_idx = data["volume"].idxmax()
    close_change = (data["close"].iloc[-1] / data["close"].iloc[0] - 1) * 100
    cv = data[VARIABLES].std(ddof=1) / data[VARIABLES].mean() * 100

    lines = [
        "# Trực quan hóa 5 biến AAPL",
        "",
        f"Phạm vi gồm {len(data):,} phiên từ {data['date'].min().date()} đến {data['date'].max().date()}. Biến mục tiêu là `close`.",
        "",
        "## 1. Line Chart – xu hướng theo thời gian",
        "",
        "![Time series](01_time_series_5_bien.png)",
        "",
        "**Mô tả biểu đồ:**",
        "",
        "- Biểu đồ gồm 5 ô tương ứng `open`, `high`, `low`, `volume` và biến mục tiêu `close`.",
        "- Trục X: thời gian giao dịch từ 02/2013 đến 02/2018, sắp xếp tăng dần theo ngày.",
        "- Trục Y: giá cổ phiếu (USD) đối với `open`, `high`, `low`, `close`; số cổ phiếu giao dịch đối với `volume` (nhãn M biểu thị triệu cổ phiếu).",
        "- Đường màu mảnh biểu diễn giá trị từng phiên; đường xanh đậm là trung bình trượt 60 phiên (MA60), dùng để làm mượt nhiễu ngắn hạn và nhận diện xu hướng.",
        "- Đường dữ liệu nằm trên MA60 thể hiện giai đoạn mạnh hơn xu hướng trung hạn; khi cắt xuống dưới MA60 thường là giai đoạn điều chỉnh. Các đỉnh nhọn của `volume` là phiên có hoạt động giao dịch bất thường.",
        "",
        f"- `open`, `high`, `low`, `close` đi gần như song song; tương quan level giữa open và close là {price_corr_level:.4f}. Đây là cấu trúc OHLC tự nhiên, không phải năm tín hiệu độc lập.",
        f"- `close` tăng {close_change:.2f}% từ đầu đến cuối kỳ. Đường MA60 cho thấy xu hướng tăng dài hạn nhưng có các đoạn điều chỉnh/đi ngang; vì vậy mean {data['close'].mean():.2f} USD không đại diện cho mọi giai đoạn thời gian.",
        f"- `volume` không tăng đều theo giá mà xuất hiện các đỉnh nhọn. Phiên lớn nhất là {data.loc[volume_max_idx, 'date'].date()} với {data.loc[volume_max_idx, 'volume']:,.0f} cổ phiếu, phù hợp với skewness {skew['volume']:.2f} và CV {cv['volume']:.2f}%.",
        "- Không thấy chu kỳ lặp lại rõ bằng mắt trên đường giá gốc; xu hướng dài hạn chi phối mạnh hơn mùa vụ.",
        "",
        "## 2. Histogram giá trị gốc",
        "",
        "![Histogram values](02_histogram_gia_tri.png)",
        "",
        "**Mô tả biểu đồ:**",
        "",
        "- Mỗi ô là histogram của một biến trong toàn bộ 1.259 phiên.",
        "- Trục X: các khoảng giá trị của biến; đơn vị USD cho bốn biến giá và số cổ phiếu (M = triệu) cho `volume`.",
        "- Trục Y: tần số, tức số phiên rơi vào từng khoảng giá trị. Cột càng cao nghĩa là khoảng đó xuất hiện càng thường xuyên.",
        "- Đường dọc đỏ là Mean; đường dọc xanh là Median. Khoảng cách và thứ tự giữa hai đường giúp nhận biết độ lệch: Mean nằm bên phải Median thường gợi ý lệch phải.",
        "- Độ rộng và số cột được giữ nhất quán trong từng ô để thể hiện hình dáng phân phối; không nên so trực tiếp chiều cao cột giữa các ô nếu thang Y khác nhau.",
        "",
        f"- Bốn biến giá có mean gần median (`close`: {data['close'].mean():.2f} so với {data['close'].median():.2f}) và skewness khoảng {skew['close']:.2f}; hình phân phối khá cân đối nhưng không phải phân phối chuẩn thuần, vì dữ liệu là các mức giá nối tiếp theo thời gian.",
        f"- `volume` lệch phải mạnh: mean {data['volume'].mean():,.0f} cao hơn median {data['volume'].median():,.0f}; đuôi phải dài trên histogram khớp với skewness {skew['volume']:.2f}.",
        "",
        "## 3. Histogram biến động ngày",
        "",
        "![Histogram changes](03_histogram_bien_dong_ngay.png)",
        "",
        "**Mô tả biểu đồ:**",
        "",
        "- Trục X: biến động logarit giữa hai phiên liên tiếp, tính bằng phần trăm: `100 × [ln(X_t) − ln(X_{t-1})]`. Giá trị âm là giảm, dương là tăng và 0 là gần như không đổi.",
        "- Trục Y: số phiên có mức biến động nằm trong từng khoảng phần trăm.",
        "- Để phần trung tâm dễ quan sát, miền trục X chỉ hiển thị từ phân vị P1 đến P99; các giá trị cực đoan ngoài miền này vẫn được giữ trong phần thống kê và nhận xét.",
        "- Đường đỏ và xanh lần lượt là Mean và Median của phần dữ liệu hiển thị. Phân phối càng hẹp quanh 0 thì biến động thông thường càng thấp; đuôi càng dài thì rủi ro xuất hiện phiên sốc càng lớn.",
        "",
        f"- Daily log return của `close` tập trung quanh 0, độ lệch chuẩn {volatility['daily_log_return_std_percent']:.2f}%/phiên. Đuôi phân phối vẫn dày hơn dạng chuông đơn giản, phản ánh các phiên sốc.",
        f"- Phiên tăng mạnh nhất: {volatility['max_daily_log_return_date']} ({volatility['max_daily_log_return_percent']:.2f}%); giảm mạnh nhất: {volatility['min_daily_log_return_date']} ({volatility['min_daily_log_return_percent']:.2f}%).",
        f"- Tương quan daily change open–close chỉ {price_corr_change:.4f}, thấp xa mức 0,9991 ở level; high/low với close cao hơn (xấp xỉ 0,71–0,72). Biến động volume rộng và bất đối xứng hơn.",
        "",
        "## 4. Mùa vụ theo tháng sau khi khử xu hướng",
        "",
        "![Seasonality](04_mua_vu_theo_thang.png)",
        "",
        "**Mô tả biểu đồ:**",
        "",
        "- Biểu đồ trên dành cho `open`, `high`, `low`, `close`; biểu đồ dưới dành riêng cho `volume` vì biên độ khác đáng kể.",
        "- Trục X: tháng trong năm, từ T1 đến T12; mỗi điểm tổng hợp các quan sát của cùng tháng qua các năm trong mẫu.",
        "- Trục Y: sai lệch trung bình (%) so với xu hướng trung tâm 252 phiên sau khi biến đổi log. Mốc 0% nghĩa là xấp xỉ xu hướng; giá trị dương/âm nghĩa là cao/thấp hơn xu hướng.",
        "- Các đường màu cho biết mẫu hình tháng của từng biến. Đường lặp có đỉnh/đáy ổn định qua tháng gợi ý seasonality, nhưng đồ thị này chỉ mô tả trung bình của 5 năm và chưa kèm khoảng tin cậy.",
        "",
        f"- Với `close`, tháng cao nhất tương đối là T{season['close']['highest_month']} ({season['close']['highest_percent']:.2f}% trên xu hướng) và thấp nhất là T{season['close']['lowest_month']} ({season['close']['lowest_percent']:.2f}%); biên độ chỉ {season['close']['amplitude_percent']:.2f} điểm %. Tín hiệu nhỏ và chỉ dựa trên 5 năm nên chưa đủ kết luận seasonality bền vững.",
        f"- `volume` có biên độ mùa vụ mô tả lớn hơn ({season['volume']['amplitude_percent']:.2f} điểm %), cao nhất T{season['volume']['highest_month']} và thấp nhất T{season['volume']['lowest_month']}; cần kiểm định trên chuỗi dài hơn trước khi dùng dự báo.",
        "",
        "## 5. Boxplot",
        "",
        "![Boxplot](05_boxplot.png)",
        "",
        "**Mô tả biểu đồ:**",
        "",
        "- Trục X: tên biến. Bốn biến giá được đặt chung một ô; `volume` nằm ở ô riêng để tránh chênh lệch đơn vị làm bẹt boxplot giá.",
        "- Trục Y: mức giá USD ở ô bên trái và số cổ phiếu giao dịch (M = triệu) ở ô bên phải.",
        "- Cạnh dưới/cạnh trên của hộp là Q1/Q3, đường trắng trong hộp là Median; chiều cao hộp chính là IQR. Hai râu kéo đến giá trị xa nhất còn nằm trong hàng rào 1,5×IQR.",
        "- Chấm đỏ là quan sát ngoài hàng rào. Hộp lệch về một phía, râu không cân xứng hoặc nhiều chấm tập trung ở một đầu là dấu hiệu phân phối lệch và có ngoại lệ.",
        "",
        f"- Các biến giá không có ngoại lệ theo hàng rào 1,5×IQR ({outlier_counts['close']} điểm cho `close`), phù hợp với range rộng nhưng phân phối level tương đối cân đối.",
        f"- `volume` có {outlier_counts['volume']} ngoại lệ: các đỉnh giao dịch là hiện tượng thực tế đáng phân tích, không nên tự động xóa như lỗi dữ liệu.",
        "",
        "## 6. Heatmap tương quan",
        "",
        "![Correlation](06_heatmap_tuong_quan.png)",
        "",
        "**Mô tả biểu đồ:**",
        "",
        "- Cả trục X và trục Y đều liệt kê 5 biến; mỗi ô là hệ số tương quan Pearson của cặp biến tại giao điểm.",
        "- Heatmap bên trái dùng giá trị gốc (level); heatmap bên phải dùng daily log change để giảm tác động của xu hướng.",
        "- Hệ số nằm trong [-1; 1]: gần 1 là đồng biến mạnh, gần -1 là nghịch biến mạnh, gần 0 là ít quan hệ tuyến tính. Đường chéo luôn bằng 1 vì mỗi biến tương quan hoàn hảo với chính nó.",
        "- Màu xanh thể hiện tương quan dương, đỏ thể hiện tương quan âm; màu càng đậm thì độ lớn tuyệt đối càng cao. Cần đọc con số trong ô thay vì chỉ dựa vào màu.",
        "",
        f"- OHLC tương quan gần 1 ở level, nhưng sau khi chuyển sang biến động ngày thì hệ số giảm còn khoảng 0,35–0,76. Điều này cho thấy phần lớn tương quan level đến từ xu hướng chung và cảnh báo đa cộng tuyến nếu đưa đồng thời cả bốn mức giá vào mô hình.",
        f"- Tương quan `close`–`volume` ở level là {close_volume_corr:.3f}, nhưng nên ưu tiên heatmap biến động ngày khi nghiên cứu quan hệ ngắn hạn vì level bị ảnh hưởng bởi trend.",
        "",
        "## 7. Scatter CLOSE–VOLUME",
        "",
        "![Scatter](07_scatter_close_volume.png)",
        "",
        "**Mô tả biểu đồ:**",
        "",
        "- Trục X: giá đóng cửa `close` (USD). Trục Y: `volume`, tức số cổ phiếu giao dịch trong phiên (M = triệu).",
        "- Mỗi chấm là một phiên giao dịch; màu chấm biểu thị năm từ 2013 đến 2018, giúp nhận diện sự dịch chuyển của các cụm quan sát theo thời gian.",
        "- Đường đen là đường hồi quy tuyến tính tổng quát. Độ dốc âm cho thấy trong mẫu, mức giá cao thường đi cùng volume thấp hơn; chỉ số Pearson r ở góc phải định lượng độ chặt của quan hệ.",
        "- Các điểm nằm xa đám mây theo chiều dọc là phiên volume bất thường. Scatter chỉ mô tả sự liên hệ, không chứng minh quan hệ nhân quả.",
        "",
        "- Màu theo năm cho thấy các cụm điểm dịch chuyển cùng thời gian. Quan hệ tuyến tính tổng thể không đồng nghĩa volume gây ra giá; cấu trúc cụm theo năm là dấu hiệu trend và chế độ thị trường.",
        "",
        "## 8. Rolling volatility của CLOSE",
        "",
        "![Rolling volatility](08_rolling_volatility_close.png)",
        "",
        "**Mô tả biểu đồ:**",
        "",
        "- Cả hai ô dùng trục X là thời gian từ 02/2013 đến 02/2018.",
        "- Ô trên – trục Y: daily log return của `close` (%). Cột xanh nằm trên 0 là phiên tăng; cột đỏ dưới 0 là phiên giảm. Các cột chạm biên là biến động cực đoan đã được giới hạn hiển thị để giữ khả năng đọc.",
        "- Ô dưới – trục Y: độ lệch chuẩn trượt 20 phiên của daily log return, quy đổi năm theo `√252`, đơn vị %. Đường càng cao nghĩa là rủi ro/biến động gần đây càng lớn.",
        "- Các cụm cột lớn ở ô trên thường đi cùng các đỉnh ở ô dưới; đây là hình học đặc trưng của volatility clustering.",
        "",
        f"- Volatility không ổn định mà tạo thành cụm. Độ biến động 20 phiên quy đổi năm đạt cao nhất {volatility['max_rolling_volatility_percent']:.2f}% vào {volatility['max_rolling_volatility_date']}.",
        "- Vì variance thay đổi theo thời gian, mô hình giả định phương sai không đổi có thể bỏ sót rủi ro; nên cân nhắc rolling features hoặc mô hình volatility khi dự báo.",
        "",
        "## 9. Scatter matrix: CLOSE và các biến giải thích",
        "",
        "![Target relationships](09_close_vs_bien_giai_thich.png)",
        "",
        "**Mô tả biểu đồ:**",
        "",
        "- Mỗi ô giữ trục Y cố định là biến mục tiêu `close` (USD); trục X lần lượt là `open`, `high`, `low` (USD) và `volume` (số cổ phiếu, M = triệu).",
        "- Mỗi chấm là một phiên và màu biểu thị năm. Đường đen là hồi quy tuyến tính giữa biến giải thích trên trục X và `close` trên trục Y.",
        "- Tiêu đề từng ô hiển thị hai hệ số: `r(level)` cho giá trị gốc và `r(Δlog)` cho biến động log ngày. So sánh hai số giúp phát hiện tương quan cao giả tạo do các chuỗi cùng có trend.",
        "- Đám mây điểm càng sát đường đen thì quan hệ tuyến tính càng chặt; độ phân tán theo chiều dọc thể hiện phần biến thiên của `close` chưa được giải thích bởi biến X.",
        "",
        f"- `open`, `high`, `low` có tương quan level rất cao với `close`: lần lượt {corr_levels.loc['open', 'close']:.4f}, {corr_levels.loc['high', 'close']:.4f}, {corr_levels.loc['low', 'close']:.4f}. Đây vừa là quan hệ cấu trúc OHLC trong cùng phiên, vừa chịu ảnh hưởng của xu hướng giá chung.",
        f"- Sau khi chuyển sang daily log change, tương quan giảm còn {corr_changes.loc['open', 'close']:.4f} với `open`, {corr_changes.loc['high', 'close']:.4f} với `high` và {corr_changes.loc['low', 'close']:.4f} với `low`. Vì vậy level scatter gần đường thẳng không có nghĩa các biến dự báo hoàn hảo biến động `close`.",
        f"- `volume` có tương quan level âm {corr_levels.loc['volume', 'close']:.4f}, nhưng tương quan biến động ngày chỉ {corr_changes.loc['volume', 'close']:.4f}. Quan hệ âm ở level chủ yếu phản ánh các chế độ thời gian: giai đoạn giá thấp đầu mẫu có volume cao hơn.",
        "- Lưu ý mô hình hóa: `high` và `low` cùng ngày chỉ được xác nhận sau khi phiên diễn ra; dùng chúng để dự báo `close` cùng ngày có thể gây rò rỉ thông tin. Với bài toán dự báo thực sự, nên dùng các biến trễ như `open_t`, `volume_{t-1}`, `close_{t-1}` hoặc rolling features.",
        "",
        "## Kết luận",
        "",
        "Dữ liệu nổi bật bởi xu hướng tăng dài hạn của giá, đồng biến rất mạnh giữa OHLC, volume lệch phải với nhiều đỉnh bất thường và volatility clustering. Seasonality theo tháng có tín hiệu mô tả nhưng yếu hơn trend và chưa đủ ổn định để kết luận chỉ với 5 năm dữ liệu.",
        "",
    ]
    metrics = {
        "close_total_change_percent": close_change,
        "close_volume_corr": close_volume_corr,
        "correlation_levels": corr_levels.round(6).to_dict(),
        "correlation_daily_log_changes": corr_changes.round(6).to_dict(),
        "seasonality": season,
        "outlier_counts_1_5_iqr": outlier_counts,
        "volatility": volatility,
        "summary": stats.to_dict(),
    }
    (OUTPUT_DIR / "visualization_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return "\n".join(lines)


def make_contact_sheet() -> None:
    files = [OUTPUT_DIR / f"{i:02d}_{name}" for i, name in enumerate([
        "time_series_5_bien.png",
        "histogram_gia_tri.png",
        "histogram_bien_dong_ngay.png",
        "mua_vu_theo_thang.png",
        "boxplot.png",
        "heatmap_tuong_quan.png",
        "scatter_close_volume.png",
        "rolling_volatility_close.png",
        "close_vs_bien_giai_thich.png",
    ], start=1)]
    thumbs = []
    for file in files:
        image = Image.open(file).convert("RGB")
        image.thumbnail((760, 430))
        thumbs.append((file.name, image.copy()))
    sheet = Image.new("RGB", (1660, 2415), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((50, 30), "AAPL – Contact sheet kiểm tra 9 biểu đồ", font=FT, fill=INK)
    for idx, (name, thumb) in enumerate(thumbs):
        col, row = idx % 2, idx // 2
        x, y = 50 + col * 810, 95 + row * 455
        if idx == len(thumbs) - 1 and len(thumbs) % 2 == 1:
            x = 440
        draw.rounded_rectangle((x, y, x + 780, y + 430), radius=14, fill=PANEL, outline="#E3EAF2", width=2)
        draw.text((x + 18, y + 12), name, font=F9, fill=INK)
        sheet.paste(thumb, (x + 10 + (760 - thumb.width) // 2, y + 48 + (370 - thumb.height) // 2))
    sheet.save(OUTPUT_DIR / "00_contact_sheet.png", optimize=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    data = (
        raw.loc[raw["Name"].eq(STOCK_CODE), ["date", *VARIABLES]]
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    if data.empty:
        raise ValueError(f"Không tìm thấy dữ liệu cho mã {STOCK_CODE}")
    if data[VARIABLES].isna().any().any():
        raise ValueError("Dữ liệu AAPL còn giá trị thiếu trong 5 biến cần vẽ")

    make_time_series(data)
    make_raw_histograms(data)
    changes = make_change_histograms(data)
    monthly = make_seasonality(data)
    make_boxplots(data)
    corr_levels, corr_changes = make_correlations(data, changes)
    close_volume_corr = make_scatter(data)
    volatility = make_rolling_volatility(data, changes)
    make_target_relationships(data, changes)
    report = build_report(data, changes, monthly, corr_levels, corr_changes, close_volume_corr, volatility)
    (OUTPUT_DIR / "nhan_xet_bieu_do.md").write_text(report, encoding="utf-8")
    make_contact_sheet()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report)


if __name__ == "__main__":
    main()
