import numpy
import yfinance as yf

import json
import datetime
import os

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

print(f"Loaded news for {len(news_data)} companies.")

# Example: Access articles for a specific stock
sample_ticker = "AAPL"
if sample_ticker in news_data:
    print(f"First news article for {sample_ticker}:")
    print(news_data[sample_ticker][0])  # Print the first article for AAPL