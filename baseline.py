import numpy
import pandas as pd
import json
import datetime
import os

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

def getStockPriceChange(news_data):

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


if __name__ == "__main__":
    news_data = loadCompanyNews()
    sentiment_scores = getSentimentScores(news_data)
    price_changes = getStockPriceChange(news_data)

    # Example output for UNH (assuming that's the ticker in your data)
    ticker = "UNH"
    print(f"Sentiment scores for {ticker}:")
    for article_id, score in sentiment_scores.get(ticker, {}).items():
        print(f"Article ID {article_id}: {score}")
    
    print(f"\nStock price changes for {ticker}:")
    for article_id, change in price_changes.get(ticker, {}).items():
        print(f"Article ID {article_id}: {change}")
    
    print(len(sentiment_scores.get("UNH", {})), len(price_changes.get("UNH", {})))
    


