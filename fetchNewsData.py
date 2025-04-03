import numpy
import yfinance as yf
import finnhub

import json
import datetime
import os

# finhubb api key: cvmrjp1r01ql90pvr4g0cvmrjp1r01ql90pvr4gg
finnhub_client = finnhub.Client("cvmrjp1r01ql90pvr4g0cvmrjp1r01ql90pvr4gg")

tickers = ["AAPL", "AMZN", "COR", "GOOG", "JPM", "MCK", "MSFT", "UNH", "WMT", "XOM"]

start_date = "2024-01-01"
end_date = "2024-12-31"
start_timestamp = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp())
end_timestamp = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp())

# Create a directory to store articles
output_dir = "finnhub_news"
os.makedirs(output_dir, exist_ok=True)

# Fetch and store news for each ticker
for ticker in tickers:
    try:
        print(f"Fetching news for {ticker}...")
        articles = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
        
        # Save each company's news to a separate JSON file
        file_path = os.path.join(output_dir, f"{ticker}_news.json")
        with open(file_path, "w") as file:
            json.dump(articles, file, indent=4)

        print(f"Saved {len(articles)} articles for {ticker} in {file_path}")

    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")

print("All news articles have been saved.")