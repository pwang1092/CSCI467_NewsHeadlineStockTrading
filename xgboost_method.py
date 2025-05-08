
import os
import json
import datetime
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

#########################################
# Data Loading and Preprocessing Functions
#########################################

def loadCompanyNews(directory="finnhub_news"):
    """
    Loads company news JSON files from the specified directory.
    Each file is assumed to be named like "<TICKER>_news.json" and contain a list of articles.
    """
    news_data = {}
    for filename in os.listdir(directory):
        if filename.endswith("_news.json"):
            ticker = filename.split("_news.json")[0]
            with open(os.path.join(directory, filename), "r") as file:
                news_data[ticker] = json.load(file)
    return news_data

def prepare_text_data(news_data):
    """
    Processes the raw news_data into article_texts and article_metadata.
    Uses both 'headline' and 'summary' fields and filters out any articles with invalid datetime.
    """
    article_texts = {}
    article_metadata = {}

    for ticker, articles in news_data.items():
        for article in articles:
            aid = article.get("id")
            dt = article.get("datetime")
            if aid is None or not isinstance(dt, (int, float)) or dt < 946684800:
                continue

            headline = article.get("headline", "").strip()
            summary = article.get("summary", "").strip()
            text = headline if not summary else f"{headline} {summary}"
            if not text:
                continue

            date = datetime.datetime.fromtimestamp(dt)
            article_texts[aid] = text
            article_metadata[aid] = {"ticker": ticker, "date": date}

    return article_texts, article_metadata

import yfinance as yf

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

def prepare_features_and_labels(article_texts, article_metadata, price_changes):
    """
    Builds TF‑IDF features and corresponding labels.
    """
    valid_ids = [aid for aid in article_texts if aid in price_changes and price_changes[aid] is not None]
    texts, y, metadata = [], [], []

    for aid in valid_ids:
        texts.append(article_texts[aid])
        y.append(price_changes[aid])
        metadata.append(article_metadata[aid])

    if not texts:
        raise ValueError("No valid articles after filtering.")

    tfidf = TfidfVectorizer(max_features=500)
    X = tfidf.fit_transform(texts)
    y = np.array(y)
    return X, y, metadata, tfidf

#########################################
# Modeling Functions
#########################################

def train_xgboost(X, y):
    """
    Trains an XGBoost regressor on the TF‑IDF features.
    """
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)
    return model

def evaluate_per_ticker(model, X, y, metadata):
    """
    Computes MSE and directional accuracy per ticker.
    """
    df = pd.DataFrame(metadata)
    df["y_true"] = y
    df["y_pred"] = model.predict(X)

    results = {}
    for ticker in df["ticker"].unique():
        sub = df[df["ticker"] == ticker]
        mse = mean_squared_error(sub["y_true"], sub["y_pred"])
        dir_acc = ((sub["y_true"] > 0) == (sub["y_pred"] > 0)).mean()
        results[ticker] = {"MSE": mse, "Directional Accuracy": dir_acc}

    return results

#########################################
# Main Pipeline
#########################################

if __name__ == "__main__":
    # 1. Load and preprocess
    news_data = loadCompanyNews()
    article_texts, article_metadata = prepare_text_data(news_data)
    price_changes = getStockPriceChange(news_data)

    # 2. Build features & labels
    X, y, metadata, tfidf = prepare_features_and_labels(article_texts, article_metadata, price_changes)

    # 3. Train XGBoost
    xgb_model = train_xgboost(X, y)

    # 4. Evaluate and print
    results = evaluate_per_ticker(xgb_model, X, y, metadata)

    print("\n=== XGBoost News-Based Stock Prediction Results ===")
    for ticker, stats in results.items():
        print(f"\n{ticker}:")
        print(f"  MSE: {stats['MSE']:.6f}")
        print(f"  Directional Accuracy: {stats['Directional Accuracy']:.2%}")