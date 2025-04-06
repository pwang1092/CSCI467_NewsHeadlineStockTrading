import numpy as np
import pandas as pd
import json
import datetime
import os

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def loadCompanyNews():
    # Directory where JSON files are stored
    output_dir = "finnhub_news"

    # Dictionary to store articles by stock ticker
    news_data = {}

    # Load articles from saved JSON files
    for filename in os.listdir(output_dir):
        if filename.endswith("_news.json"):
            ticker = filename.split("_news.json")[0]  # Extract ticker from filename
            with open(os.path.join(output_dir, filename), "r") as file:
                news_data[ticker] = json.load(file)  # Store list of articles under the ticker
    
    return news_data

def split_news_data(news_data, train_size=0.9):
    '''
    Split news dataset into train and test datasets
    '''
    train_news_data = {}
    test_news_data = {}
    for ticker, articles in news_data.items():
        # Split articles into train and test sets (90% train, 10% test)
        train_articles, test_articles = train_test_split(
            articles, train_size=train_size, random_state=42, shuffle=True
        )
        train_news_data[ticker] = train_articles
        test_news_data[ticker] = test_articles
    return train_news_data, test_news_data


def getSentimentScores(news_data):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {}

    for ticker, articles in news_data.items():
        ticker_scores = {}
        for article in articles:
            article_id = article.get("id")  # Get the article ID
            if article_id is None:  # Skip if no ID is present
                continue
            
            headline = article.get("headline", "")
            summary = article.get("summary", "")

            # Combine headline + summary for sentiment analysis
            text = f"{headline} {summary}".strip()
            sentiment = analyzer.polarity_scores(text)["compound"]

            ticker_scores[article_id] = sentiment  # Map ID to sentiment score
        
        sentiment_scores[ticker] = ticker_scores  # Store dictionary for the ticker

    return sentiment_scores


def getStockPriceChange(news_data): # based on close price of the current day of article and close price of previous day
    price_changes = {}

    for ticker, articles in news_data.items():
        if not articles:
            price_changes[ticker] = {}
            continue
        
        # Build a dictionary of article IDs to publication dates
        publication_dates = {}
        for article in articles:
            article_id = article.get("id")
            if article_id is None or not isinstance(article.get("datetime"), (int, float)):
                continue
            try:
                publication_dates[article_id] = datetime.datetime.fromtimestamp(article["datetime"])
            except (ValueError, OSError):
                continue  # Skip invalid timestamps

        if not publication_dates:
            price_changes[ticker] = {article["id"]: None for article in articles if article.get("id") is not None}
            continue

        # Buffer period to cover all publication dates
        min_date = min(publication_dates.values()) - datetime.timedelta(days=30)
        max_date = max(publication_dates.values()) + datetime.timedelta(days=30)
        stock_data = yf.download(ticker, start=min_date, end=max_date, progress=False)

        if stock_data.empty:
            price_changes[ticker] = {article_id: None for article_id in publication_dates.keys()}
            continue

        stock_data['return'] = stock_data['Close'].pct_change()
        ticker_price_changes = {}
        for article_id, pub_datetime in publication_dates.items():
            pub_date = pub_datetime.date()
            trading_day = stock_data.index[stock_data.index >= pd.Timestamp(pub_date)].min()
            if pd.notna(trading_day):
                price_change = stock_data.loc[trading_day, 'return'].item()  # Extract float value
                ticker_price_changes[article_id] = price_change
            else:
                ticker_price_changes[article_id] = None
        
        price_changes[ticker] = ticker_price_changes

    return price_changes


# New function: Calculate mean squared error for each ticker
def calculate_mse(sentiment_scores, price_changes, models):
    mse_scores = {}
    for ticker in models.keys():
        if models[ticker] is None:
            mse_scores[ticker] = None
            continue
        
        sentiment_dict = sentiment_scores.get(ticker, {})
        price_dict = price_changes.get(ticker, {})
        
        # Align data by article ID, excluding None values
        article_ids = set(sentiment_dict.keys()) & set(price_dict.keys())
        X = [sentiment_dict[article_id] for article_id in article_ids if price_dict[article_id] is not None]
        y_true = [price_dict[article_id] for article_id in article_ids if price_dict[article_id] is not None]
        
        if not X or not y_true:
            mse_scores[ticker] = None
            continue
        
        X = np.array(X).reshape(-1, 1)
        
        model = models[ticker]
        y_pred = model.predict(X)
        
        mse = mean_squared_error(y_true, y_pred)
        mse_scores[ticker] = mse
    
    return mse_scores


def calculate_directional_accuracy(sentiment_scores, price_changes):
    accuracy_scores = {}
    for ticker in sentiment_scores.keys():
        sentiment_dict = sentiment_scores.get(ticker, {})
        price_dict = price_changes.get(ticker, {})
        
        # Align data by article ID, excluding None values
        article_ids = set(sentiment_dict.keys()) & set(price_dict.keys())
        sentiments = [sentiment_dict[article_id] for article_id in article_ids if price_dict[article_id] is not None]
        price_changes_list = [price_dict[article_id] for article_id in article_ids if price_dict[article_id] is not None]
        
        if not sentiments or not price_changes_list:
            accuracy_scores[ticker] = None
            print(f"No valid data for {ticker}")
            continue
        
        correct = 0
        total = 0
        for sentiment, price_change in zip(sentiments, price_changes_list):
            if sentiment == 0:
                continue
            # Positive sentiment predicts up, negative predicts down
            predicted_up = sentiment > 0
            actual_up = price_change > 0
            if predicted_up == actual_up:
                correct += 1
            total += 1
        
        if total > 0:
            accuracy = correct / total
            accuracy_scores[ticker] = accuracy
        else:
            accuracy_scores[ticker] = None
    
    return accuracy_scores

def train_linear_regression(sentiment_scores, price_changes):
    models = {}
    for ticker in sentiment_scores.keys():
        # Get sentiment scores (X) and price changes (y) for this ticker
        sentiment_dict = sentiment_scores.get(ticker, {})
        price_dict = price_changes.get(ticker, {})
        
        # Align data by article ID, excluding None values
        article_ids = set(sentiment_dict.keys()) & set(price_dict.keys())  # Intersection of IDs
        X = [sentiment_dict[article_id] for article_id in article_ids if price_dict[article_id] is not None]
        y = [price_dict[article_id] for article_id in article_ids if price_dict[article_id] is not None]
        
        if len(X) < 2 or len(y) < 2:
            models[ticker] = None
            print(f"Skipping {ticker}: insufficient data ({len(X)} points)")
            continue
        
        # Reshape X for sklearn
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        
        # Train the model
        model = LinearRegression()
        model.fit(X, y)
        models[ticker] = model
    
    return models


if __name__ == "__main__":
    news_data = loadCompanyNews()

    train_news_data, test_news_data = split_news_data(news_data, train_size=0.9)

    train_sentiment_scores = getSentimentScores(train_news_data)
    train_price_changes = getStockPriceChange(train_news_data)
    test_sentiment_scores = getSentimentScores(test_news_data)
    test_price_changes = getStockPriceChange(test_news_data)

    models = train_linear_regression(train_sentiment_scores, train_price_changes)

    # Evaluate on test data
    test_mse_scores = calculate_mse(test_sentiment_scores, test_price_changes, models)
    test_accuracy_scores = calculate_directional_accuracy(test_sentiment_scores, test_price_changes)

    # Example output for UNH (assuming that's the ticker in your data)
    '''
    ticker = "UNH"
    print(f"Sentiment scores for {ticker}:")
    for article_id, score in sentiment_scores.get(ticker, {}).items():
        print(f"Article ID {article_id}: {score}")
    
    print(f"\nStock price changes for {ticker}:")
    for article_id, change in price_changes.get(ticker, {}).items():
        print(f"Article ID {article_id}: {change}")
    
    print(len(sentiment_scores.get("UNH", {})), len(price_changes.get("UNH", {})))
    '''

    # Print results
    for ticker in models.keys():
        if models[ticker] is not None:
            print(f"\n{ticker} Model (trained on {len(train_sentiment_scores.get(ticker, {}))} articles): "
                  f"slope={models[ticker].coef_[0]:.4f}, intercept={models[ticker].intercept_:.4f}")
            print(f"{ticker} Test MSE: {test_mse_scores.get(ticker, 'N/A'):.6f}")
            accuracy = test_accuracy_scores.get(ticker)
            if accuracy is not None:
                print(f"{ticker} Test Directional Accuracy: {accuracy:.2%}")
            else:
                print(f"{ticker} Test Directional Accuracy: N/A (no valid predictions)")
        else:
            print(f"\n{ticker}: No model trained (insufficient training data)")
    news_data = loadCompanyNews()
    sentiment_scores = getSentimentScores(news_data)
    price_changes = getStockPriceChange(news_data)


