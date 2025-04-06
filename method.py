import os
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

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
            article_id = article.get("id")
            dt = article.get("datetime")

            # Skip if article id is missing or datetime is invalid (before year 2000)
            if article_id is None or not isinstance(dt, (int, float)) or dt < 946684800:
                continue

            # Explicitly combine headline and summary
            headline = article.get("headline", "").strip()
            summary = article.get("summary", "").strip()
            # If summary is empty, use headline only; otherwise, join them.
            text = headline if not summary else " ".join([headline, summary])
            # If text is still empty, skip it.
            if not text:
                continue

            date = datetime.datetime.fromtimestamp(dt)
            article_texts[article_id] = text
            article_metadata[article_id] = {
                "ticker": ticker,
                "date": date
            }
    return article_texts, article_metadata

def getStockPriceChange(news_data):
    """
    Your original function that computes stock price changes using yfinance.
    (Assume this works as in your baseline.)
    """
    # ... (your baseline code here) ...
    # For example purposes, we provide a dummy implementation.
    price_changes = {}
    for ticker, articles in news_data.items():
        for article in articles:
            article_id = article.get("id")
            # In your real code, this would be the computed price change.
            # Here we use a random normal value as a placeholder.
            price_changes[article_id] = np.random.normal(0, 0.02)
    return price_changes

def prepare_features_and_labels(article_texts, article_metadata, price_changes):
    """
    Prepares TF-IDF features and labels from the article texts.
    Filters out any articles that end up with empty text.
    """
    valid_ids = [
        aid for aid in article_texts
        if aid in price_changes and price_changes[aid] is not None
    ]

    texts = []
    y = []
    metadata = []

    for aid in valid_ids:
        # Use the combined text from prepare_text_data
        text = article_texts[aid].strip()
        # Debug: Uncomment the next line to see what each article's text looks like
        # print(f"Article ID {aid}: {text[:100]}")
        if not text:
            continue
        texts.append(text)
        y.append(price_changes[aid])
        metadata.append(article_metadata[aid])

    if len(texts) == 0:
        raise ValueError("No valid article texts found after filtering.")

    tfidf = TfidfVectorizer(max_features=500)
    X = tfidf.fit_transform(texts)
    y = np.array(y)
    return X, y, metadata, tfidf

#########################################
# Modeling Functions
#########################################

def train_mlp(X, y):
    """
    Trains a shallow feedforward neural network (MLP) using the TF-IDF features.
    """
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    model.fit(X, y)
    return model

def evaluate_per_ticker(model, X, y, metadata):
    """
    Evaluates the model performance per ticker.
    Returns metrics such as Mean Squared Error (MSE) and directional accuracy.
    """
    df = pd.DataFrame(metadata)
    df["y_true"] = y
    df["y_pred"] = model.predict(X)

    results = {}
    for ticker in df["ticker"].unique():
        sub_df = df[df["ticker"] == ticker]
        mse = mean_squared_error(sub_df["y_true"], sub_df["y_pred"])
        directional_accuracy = ((sub_df["y_true"] > 0) == (sub_df["y_pred"] > 0)).mean()
        results[ticker] = {
            "MSE": mse,
            "Directional Accuracy": directional_accuracy
        }
    return results

#########################################
# Main Pipeline
#########################################

if __name__ == "__main__":
    # Assume your baseline preprocessing functions are working correctly.
    # Load news data from your directory
    news_data = loadCompanyNews()  # Ensure your JSON files are in the 'finnhub_news' directory

    # Process the news articles into texts and metadata
    article_texts, article_metadata = prepare_text_data(news_data)
    
    # Compute stock price changes (your original function)
    price_changes = getStockPriceChange(news_data)
    
    # Prepare TF-IDF features and labels
    X, y, metadata, tfidf = prepare_features_and_labels(article_texts, article_metadata, price_changes)
    
    # Train the MLP model
    model = train_mlp(X, y)
    
    # Evaluate the model per ticker and print the results
    results = evaluate_per_ticker(model, X, y, metadata)
    
    print("\n=== MLP News-Based Stock Prediction Results ===")
    for ticker, stats in results.items():
        print(f"\n{ticker}:")
        print(f"  MSE: {stats['MSE']:.6f}")
        print(f"  Directional Accuracy: {stats['Directional Accuracy']:.2%}")
