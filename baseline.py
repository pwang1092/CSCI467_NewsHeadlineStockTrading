import numpy
import yfinance as yf

import json
import datetime
import os

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
        ticker_scores = []
        for article in articles:
            headline = article.get("headline", "")
            summary = article.get("summary", "")

            # Combine headline + summary for sentiment analysis
            text = f"{headline} {summary}".strip()
            sentiment = analyzer.polarity_scores(text)["compound"]

            ticker_scores.append(sentiment)  # Store the score
        
        sentiment_scores[ticker] = ticker_scores  # Store for the ticker

    return sentiment_scores


if __name__ == "__main__":
    news_data = loadCompanyNews()
    sentiment_scores = getSentimentScores(news_data)
    print(f"Sentiment scores for AAPL: {sentiment_scores.get('AAPL', [])[:]}")


