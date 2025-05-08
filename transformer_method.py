import os
import json
import datetime
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

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
            ticker = fn[:-10]  # strip "_news.json"
            path = os.path.join(directory, fn)
            with open(path, "r") as f:
                data[ticker] = json.load(f)
    return data

def prepare_text_data(news_data):
    """
    Builds:
      - texts:    { id → "headline summary" }
      - metadata: { id → { "ticker":…, "date": datetime } }
    Filters out entries with missing/invalid timestamps.
    """
    texts, metadata = {}, {}
    for ticker, articles in news_data.items():
        for art in articles:
            aid = art.get("id")
            dt  = art.get("datetime")
            if aid is None or not isinstance(dt, (int, float)) or dt < 946684800:
                continue
            head = art.get("headline", "").strip()
            summ = art.get("summary",  "").strip()
            text = head if not summ else f"{head} {summ}"
            if not text:
                continue
            texts[aid] = text
            metadata[aid] = {
                "ticker": ticker,
                "date":    datetime.datetime.fromtimestamp(dt)
            }
    return texts, metadata

#########################################
# 2) (Placeholder) Stock price change
#########################################

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

#########################################
# 3) Compute Transformer Embeddings
#########################################

def compute_embeddings(texts, model_name="bert-base-uncased", batch_size=16, max_length=128):
    """
    Given a list of texts, returns an array of shape (N, hidden_size)
    using the [CLS] token from the last hidden state.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_ids = list(texts.keys())
    embeddings = []

    for i in tqdm(range(0, len(all_ids), batch_size), desc="Embedding"):
        batch_ids = all_ids[i : i + batch_size]
        batch_texts = [texts[aid] for aid in batch_ids]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model(**encoded)
        # take [CLS] embedding
        cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)

    embeddings = np.vstack(embeddings)
    # map back to dict: { aid → embedding row }
    return { aid: embeddings[idx] for idx, aid in enumerate(all_ids) }

#########################################
# 4) Build feature matrix + labels
#########################################

def prepare_features_and_labels(texts, metadata, price_changes, emb_dict):
    """
    Filters valid IDs, builds X (embeddings) and y (price changes),
    plus a list of per-article metadata for evaluation.
    """
    valid_ids = [aid for aid in texts if aid in price_changes and aid in emb_dict]
    X = np.vstack([emb_dict[aid] for aid in valid_ids])
    y = np.array([price_changes[aid] for aid in valid_ids])
    meta = [metadata[aid] for aid in valid_ids]
    return X, y, meta

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
    for tkr in df["ticker"].unique():
        sub = df[df["ticker"] == tkr]
        mse = mean_squared_error(sub["y_true"], sub["y_pred"])
        dir_acc = ((sub["y_true"] > 0) == (sub["y_pred"] > 0)).mean()
        results[tkr] = {"MSE": mse, "DirAcc": dir_acc}
    return results

#########################################
# 6) Main pipeline
#########################################

if __name__ == "__main__":
    # 1) Load & preprocess
    news_data     = load_company_news("finnhub_news")
    texts, meta   = prepare_text_data(news_data)
    price_changes = get_stock_price_change(news_data)

    # 2) Compute pretrained transformer embeddings
    emb_dict = compute_embeddings(texts, model_name="bert-base-uncased")

    # 3) Prepare features & labels
    X, y, meta_list = prepare_features_and_labels(texts, meta, price_changes, emb_dict)

    # 4) Train MLP
    mlp_model = train_mlp(X, y)

    # 5) Evaluate & print
    stats = evaluate_per_ticker(mlp_model, X, y, meta_list)
    print("\n=== Transformer‑based MLP Results ===")
    for tkr, vals in stats.items():
        print(f"\n{tkr}:")
        print(f"  MSE: {vals['MSE']:.6f}")
        print(f"  Directional Accuracy: {vals['DirAcc']:.2%}")