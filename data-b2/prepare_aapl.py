from pathlib import Path
import json
import pandas as pd


out = Path("outputs/visualizations_aapl")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("all_stocks_5yr.csv", parse_dates=["date"])
d = df.loc[df["Name"].eq("AAPL"), ["date", "open", "high", "low", "close", "volume"]].copy()
d = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)
d.to_csv(out / "aapl_source.csv", index=False)

r = d["close"].pct_change()
stats = {
    "rows": int(len(d)),
    "date_min": str(d["date"].min().date()),
    "date_max": str(d["date"].max().date()),
    "close_start": float(d.iloc[0]["close"]),
    "close_end": float(d.iloc[-1]["close"]),
    "close_total_change_pct": float((d.iloc[-1]["close"] / d.iloc[0]["close"] - 1) * 100),
    "close_mean": float(d["close"].mean()),
    "close_median": float(d["close"].median()),
    "close_std": float(d["close"].std()),
    "volume_mean": float(d["volume"].mean()),
    "volume_median": float(d["volume"].median()),
    "volume_max": int(d["volume"].max()),
    "volume_max_date": str(d.loc[d["volume"].idxmax(), "date"].date()),
    "max_daily_return_pct": float(r.max() * 100),
    "max_daily_return_date": str(d.loc[r.idxmax(), "date"].date()),
    "min_daily_return_pct": float(r.min() * 100),
    "min_daily_return_date": str(d.loc[r.idxmin(), "date"].date()),
    "corr": d[["open", "high", "low", "close", "volume"]].corr().round(4).to_dict(),
}
(out / "aapl_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
