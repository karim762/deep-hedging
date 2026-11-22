"""Download the daily closes used in part 4 into data/ (one CSV per index, 1990 to today).

    python scripts/download_data.py

Source: Yahoo Finance chart endpoint. The notebooks cut the series at 2026-11-19, so later
downloads give the same results as long as the provider does not revise past closes.
"""
import pathlib
import time

import pandas as pd
import requests

TICKERS = {"sp500": "^GSPC", "nasdaq100": "^NDX", "dax": "^GDAXI",
           "ftse100": "^FTSE", "nikkei225": "^N225", "hangseng": "^HSI"}
START = 631152000                                   # 1 January 1990, Unix time
DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"


def download(ticker):
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(ticker)}"
    params = {"period1": START, "period2": int(time.time()), "interval": "1d"}
    r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    closes = pd.Series(res["indicators"]["quote"][0]["close"],
                       index=pd.to_datetime(res["timestamp"], unit="s").normalize(), name="Close").dropna()
    closes.index.name = "Date"
    return closes


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for name, ticker in TICKERS.items():
        closes = download(ticker)
        closes.to_csv(DATA_DIR / f"{name}_daily.csv")
        print(f"{name:10s} {len(closes):6d} days, {closes.index.min().date()} to {closes.index.max().date()}")
        time.sleep(1)
