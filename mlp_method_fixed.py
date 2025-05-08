#!/usr/bin/env python3

import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

#########################################
# 1) Load & preprocess raw news articles
#########################################

def loadCompanyNews(directory="finnhub_news"):
    data = {}
    for fn in os.listdir(directory):
        if fn.endswith("_news.json"):
            ticker = fn[:-10]
            with open(os.path.join(directory, fn), "r") as f:
                data[ticker] = json.load(f)
    return data

def prepare_text_data(news_data):
    texts, meta = {}, {}
    for ticker, articles in news_data.items():
        for art in articles:
            aid = art.get("id")
            dt  = art.get("datetime")
            if aid is None or not isinstance(dt, (int, float)):
                continue
            ts = dt/1000 if dt > 1e12 else dt
            try:
                pub_dt = datetime.datetime.fromtimestamp(ts)
            except (OSError, OverflowError, ValueError):
                continue

            head = art.get("headline", "").strip()
            summ = art.get("summary",  "").strip()
            txt  = head if not summ else f"{head} {summ}"
            if not txt:
                continue

            texts[aid] = txt
            meta[aid]  = {"ticker": ticker, "date": pub_dt}
    return texts, meta

#########################################
# 2) Robust yfinance download with backoff
#########################################

def safe_download(tickers, start, end, **kwargs):
    for attempt in range(5):
        try:
            return yf.download(
                tickers=tickers,
                start=start,
                end=end,
                group_by="ticker",
                threads=False,
                progress=False,
                **kwargs
            )
        except Exception as e:
            wait = 2 ** attempt
            print(f"Download error '{e}', retrying in {wait}s…")
            time.sleep(wait)
    print("Failed to download after retries; exiting.")
    sys.exit(1)

def getStockPriceChange(news_data, lookahead_days=1, buffer_days=30, chunk_size=5):
    # 1) build date ranges per ticker
    ranges = {}
    for ticker, articles in news_data.items():
        dates = []
        for art in articles:
            dt = art.get("datetime")
            if isinstance(dt, (int, float)):
                ts = dt/1000 if dt > 1e12 else dt
                try:
                    dates.append(datetime.datetime.fromtimestamp(ts).date())
                except:
                    pass
        if dates:
            ranges[ticker] = (min(dates), max(dates))

    if not ranges:
        return {}

    price_changes = {}
    tickers = list(ranges.keys())

    # 2) process in small chunks
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        start = min(ranges[t][0] for t in chunk) - datetime.timedelta(days=buffer_days)
        end   = max(ranges[t][1] for t in chunk) + datetime.timedelta(days=buffer_days + lookahead_days + 1)

        df_chunk = safe_download(chunk, start, end)

        # 3) compute returns per ticker
        for ticker in chunk:
            # pick right sub‐DataFrame
            if isinstance(df_chunk.columns, pd.MultiIndex):
                df = df_chunk[ticker].copy()
            else:
                df = df_chunk.copy()

            if "Close" not in df:
                for art in news_data[ticker]:
                    price_changes[art.get("id")] = None
                continue

            df["return"] = df["Close"].pct_change(periods=lookahead_days)

            for art in news_data[ticker]:
                aid = art.get("id")
                dt  = art.get("datetime")
                if aid is None or not isinstance(dt, (int, float)):
                    continue
                ts = dt/1000 if dt > 1e12 else dt
                try:
                    p_date = datetime.datetime.fromtimestamp(ts).date()
                except:
                    price_changes[aid] = None
                    continue

                mask = df.index.date >= p_date
                if not mask.any():
                    price_changes[aid] = None
                else:
                    day0 = df.index[mask].min()
                    ret  = df.at[day0, "return"]
                    price_changes[aid] = None if pd.isna(ret) else float(ret)

    return price_changes

from alpha_vantage.timeseries import TimeSeries
import datetime
import numpy as np

def getStockPriceChange_AV(news_data, api_key, lookahead_days=1):
    """
    Uses Alpha Vantage’s daily adjusted endpoint to compute, for each article:
      return = (AdjClose_{t+lookahead} - AdjClose_t) / AdjClose_t
    Input:  news_data = { ticker: [article_dict,…] }
            api_key    = your AlphaVantage key
    Output: { article_id → float_return or None }
    """
    ts = TimeSeries(key=api_key, output_format="pandas")
    price_changes = {}

    # 1) build a dict of date ranges per ticker
    ranges = {}
    for ticker, articles in news_data.items():
        dates = []
        for art in articles:
            dt = art.get("datetime")
            if isinstance(dt, (int, float)):
                # normalize ms→s
                ts_val = dt/1000 if dt > 1e12 else dt
                try:
                    dates.append(datetime.datetime.fromtimestamp(ts_val).date())
                except:
                    pass
        if dates:
            ranges[ticker] = (min(dates), max(dates))

    # 2) fetch & compute for each ticker
    for ticker, (d0, d1) in ranges.items():
        # Alpha Vantage returns a full series; we’ll slice
        df, _ = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
        # date strings → datetime.date index
        df.index = pd.to_datetime(df.index).date
        # compute lookahead returns
        df["return"] = df["5. adjusted close"].pct_change(periods=lookahead_days)

        for art in news_data[ticker]:
            aid = art.get("id")
            dt  = art.get("datetime")
            if aid is None or not isinstance(dt, (int, float)):
                continue

            ts_val = dt/1000 if dt > 1e12 else dt
            try:
                pub_date = datetime.datetime.fromtimestamp(ts_val).date()
            except:
                price_changes[aid] = None
                continue

            # find first trading day ≥ publication
            future = [date for date in df.index if date >= pub_date]
            if not future:
                price_changes[aid] = None
            else:
                ret = df.at[future[0], "return"]
                price_changes[aid] = float(ret) if not np.isnan(ret) else None

    return price_changes
#########################################
# 3) TF-IDF features + labels
#########################################

def prepare_features_and_labels(texts, meta, price_changes):
    ids = [aid for aid in texts if aid in price_changes and price_changes[aid] is not None]
    if not ids:
        print("No valid articles with price data; exiting.")
        sys.exit(1)

    X_texts  = [texts[aid] for aid in ids]
    y        = np.array([price_changes[aid] for aid in ids])
    meta_list = [meta[aid] for aid in ids]

    tfidf = TfidfVectorizer(max_features=500)
    X     = tfidf.fit_transform(X_texts)
    return X, y, meta_list, tfidf

#########################################
# 4) Train & evaluate MLP
#########################################

def train_mlp(X, y):
    mlp = MLPRegressor(
        hidden_layer_sizes=(64,32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    )
    mlp.fit(X, y)
    return mlp

def evaluate_per_ticker(model, X, y, meta_list):
    df = pd.DataFrame(meta_list)
    df["y_true"] = y
    df["y_pred"] = model.predict(X)

    results = {}
    for tkr in df["ticker"].unique():
        sub = df[df["ticker"] == tkr]
        mse = mean_squared_error(sub["y_true"], sub["y_pred"])
        dir_acc = ((sub["y_true"] > 0) == (sub["y_pred"] > 0)).mean()
        results[tkr] = {"MSE": mse, "Directional Accuracy": dir_acc}
    return results

#########################################
# 5) Main pipeline
#########################################

if __name__ == "__main__":
    news_data      = loadCompanyNews("finnhub_news")
    texts, meta    = prepare_text_data(news_data)
    #price_changes  = getStockPriceChange_AV(news_data, "7UYBJFRACDPEQTEF")
    price_changes  = getStockPriceChange(news_data)

    X, y, meta_list, tfidf = prepare_features_and_labels(texts, meta, price_changes)
    model = train_mlp(X, y)
    stats = evaluate_per_ticker(model, X, y, meta_list)

    print("\n=== MLP News‑Based Stock Prediction Results ===")
    for tkr, vals in stats.items():
        print(f"\n{tkr}:")
        print(f"  MSE: {vals['MSE']:.6f}")
        print(f"  Directional Accuracy: {vals['Directional Accuracy']:.2%}")
