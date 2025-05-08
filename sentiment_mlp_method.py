#!/usr/bin/env python3

import os
import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix

#########################################
# 1) Load & preprocess raw news articles
#########################################

def load_company_news(directory="finnhub_news"):
    """
    Reads all '<TICKER>_news.json' files in `directory` into a dict:
      { ticker: [ article_dict, … ] }
    """
    data = {}
    for fn in os.listdir(directory):
        if fn.endswith("_news.json"):
            ticker = fn.split("_news.json")[0]
            with open(os.path.join(directory, fn), "r") as f:
                data[ticker] = json.load(f)
    return data

def prepare_text_data(news_data):
    """
    Builds:
      - article_texts: { id → "headline summary" }
      - article_meta:  { id → { "ticker":…, "date": datetime } }
    Filters out entries with missing/invalid timestamps.
    """
    texts = {}
    meta  = {}
    for ticker, arts in news_data.items():
        for art in arts:
            aid = art.get("id")
            dt  = art.get("datetime")
            if aid is None or not isinstance(dt, (int, float)) or dt < 946684800:
                continue
            head = art.get("headline", "").strip()
            summ = art.get("summary",  "").strip()
            txt  = head if not summ else f"{head} {summ}"
            if not txt:
                continue
            texts[aid] = txt
            meta[aid]  = {
                "ticker": ticker,
                "date":    datetime.datetime.fromtimestamp(dt)
            }
    return texts, meta

#########################################
# 2) (Placeholder) Stock price change
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

#########################################
# 3) Sentiment features
#########################################

def compute_sentiment_scores(article_texts):
    """
    Returns { id → VADER compound score }.
    """
    analyzer = SentimentIntensityAnalyzer()
    return { aid: analyzer.polarity_scores(txt)["compound"]
             for aid, txt in article_texts.items() }

#########################################
# 4) Build feature matrix + labels
#########################################

def prepare_features_and_labels(article_texts, article_meta, price_changes):
    # filter valid
    valid_ids = [aid for aid in article_texts if aid in price_changes]
    
    texts = [article_texts[aid] for aid in valid_ids]
    y     = np.array([price_changes[aid] for aid in valid_ids])
    meta  = [article_meta[aid]    for aid in valid_ids]
    
    # TF‑IDF
    tfidf   = TfidfVectorizer(max_features=500)
    X_tfidf = tfidf.fit_transform(texts)  # (N,500)
    
    # raw sentiment
    sent_scores = compute_sentiment_scores(article_texts)
    raw_sent    = np.array([sent_scores[aid] for aid in valid_ids]).reshape(-1,1)
    
    # avg previous sentiment per ticker
    # sort article IDs by date per ticker
    by_tkr = {}
    for aid, m in zip(valid_ids, meta):
        by_tkr.setdefault(m["ticker"], []).append((m["date"], aid))
    for lst in by_tkr.values():
        lst.sort()
    
    # rolling average
    prev_sum = { t:(0.0,0) for t in by_tkr }
    avg_prev = []
    for m, aid in zip(meta, valid_ids):
        t = m["ticker"]
        s,c = prev_sum[t]
        avg_prev.append(s/c if c>0 else 0.0)
        sc = sent_scores[aid]
        prev_sum[t] = (s+sc, c+1)
    avg_prev = np.array(avg_prev).reshape(-1,1)
    
    # stack all features
    X_full = hstack([
        X_tfidf,
        csr_matrix(raw_sent),
        csr_matrix(avg_prev)
    ])  # shape (N, 502)
    
    return X_full, y, meta, tfidf

#########################################
# 5) Train & evaluate MLP
#########################################

def train_mlp(X, y):
    """
    Trains a simple MLP on X→y.
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42
    )
    mlp.fit(X, y)
    return mlp

def evaluate_per_ticker(model, X, y, meta):
    """
    Returns per‑ticker dict of { MSE, directional_accuracy }.
    """
    df = pd.DataFrame(meta)
    df["y_true"] = y
    df["y_pred"] = model.predict(X)
    
    results = {}
    for t in df["ticker"].unique():
        sub = df[df["ticker"]==t]
        mse = mean_squared_error(sub["y_true"], sub["y_pred"])
        dir_acc = ((sub["y_true"]>0)==(sub["y_pred"]>0)).mean()
        results[t] = {"MSE": mse, "DirAcc": dir_acc}
    return results

#########################################
# 6) Main pipeline
#########################################

if __name__ == "__main__":
    # load & preprocess
    news_data      = load_company_news("finnhub_news")
    texts, meta    = prepare_text_data(news_data)
    price_changes  = getStockPriceChange(news_data)
    
    # features & labels
    X, y, meta, tfidf = prepare_features_and_labels(texts, meta, price_changes)
    
    # train
    mlp_model = train_mlp(X, y)
    
    # evaluate
    stats = evaluate_per_ticker(mlp_model, X, y, meta)
    
    # print results
    print("\n=== Sentiment‑Augmented MLP Results ===")
    for tkr, v in stats.items():
        print(f"\n{tkr}:")
        print(f"  MSE: {v['MSE']:.6f}")
        print(f"  Directional Accuracy: {v['DirAcc']:.2%}")
