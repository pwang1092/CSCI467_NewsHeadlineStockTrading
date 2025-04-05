import numpy
import yfinance as yf
import finnhub

import json
import datetime
import os

# Finnhub API key
finnhub_client = finnhub.Client(api_key="cvmrjp1r01ql90pvr4g0cvmrjp1r01ql90pvr4gg")

#tickers = ["AAPL", "AMZN", "COR", "GOOG", "JPM", "MCK", "MSFT", "UNH", "WMT", "XOM"]
tickers = ["UNH"]

start_date = datetime.date(2024, 1, 1)
end_date = datetime.date(2024, 12, 31)

# Create a directory to store articles
output_dir = "finnhub_news"
os.makedirs(output_dir, exist_ok=True)

# Fetch and store news for each ticker
for ticker in tickers:
    try:
        print(f"Fetching news for {ticker}...")
        articles = []

        # Fetch data week-by-week
        current_date = start_date
        while current_date <= end_date:
            next_week = current_date + datetime.timedelta(days=6)  # Get end of the week

            if next_week > end_date:
                next_week = end_date

            # Format dates as strings
            start_str = current_date.strftime("%Y-%m-%d")
            end_str = next_week.strftime("%Y-%m-%d")

            # Fetch news for this time window
            weekly_articles = finnhub_client.company_news(ticker, _from=start_str, to=end_str)
            articles.extend(weekly_articles)

            print(f"  - {len(weekly_articles)} articles from {start_str} to {end_str}")

            current_date = next_week + datetime.timedelta(days=1)

        # Save each company's news to a separate JSON file
        file_path = os.path.join(output_dir, f"{ticker}_news.json")
        with open(file_path, "w") as file:
            json.dump(articles, file, indent=4)

        print(f"Saved {len(articles)} articles for {ticker} in {file_path}")

    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")

print("All news articles have been saved.")
