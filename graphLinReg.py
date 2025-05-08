
import matplotlib.pyplot as plt
import numpy as np

def plot_regression(ticker, sentiment_scores, price_changes, models):
    """
    Plot the regression line and data points for a given ticker.
    
    Parameters:
    - ticker: Stock ticker symbol (e.g., "UNH")
    - sentiment_scores: Dictionary of sentiment scores by ticker
    - price_changes: Dictionary of price changes by ticker
    - models: Dictionary of trained LinearRegression models by ticker
    """
    if ticker not in models or models[ticker] is None:
        print(f"No model available for {ticker}")
        return
    
    # Extract data for the ticker
    sentiment_dict = sentiment_scores.get(ticker, {})
    price_dict = price_changes.get(ticker, {})
    article_ids = set(sentiment_dict.keys()) & set(price_dict.keys())
    X = [sentiment_dict[article_id] for article_id in article_ids if price_dict[article_id] is not None]
    y = [price_dict[article_id] for article_id in article_ids if price_dict[article_id] is not None]
    
    if not X or not y:
        print(f"No valid data to plot for {ticker}")
        return
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Get model parameters
    model = models[ticker]
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Generate points for the regression line
    x_range = np.linspace(min(X), max(X), 100)  # 100 points across the range of X
    y_pred = slope * x_range + intercept
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5)  # Scatter plot of actual data
    plt.plot(x_range, y_pred, color='red')
    
    # Add labels and title
    plt.xlabel('Sentiment Score', fontsize=16)
    plt.ylabel('Stock Price Change (Daily Return)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.show()


def plot_all_models(models):
    """
    Plot all regression lines and an average line for all tickers.
    
    Parameters:
    - models: Dictionary of trained LinearRegression models by ticker
    """
    plt.figure(figsize=(12, 8))
    
    # Define a common x-range for all lines (sentiment scores are typically -1 to 1)
    x_range = np.linspace(-1, 1, 100)
    
    # Lists to store slopes and intercepts for averaging
    slopes = []
    intercepts = []
    
    # Plot each model's line
    for ticker, model in models.items():
        if model is None:
            continue
        slope = model.coef_[0]
        intercept = model.intercept_
        y_pred = slope * x_range + intercept
        plt.plot(x_range, y_pred, label=f'{ticker}', alpha=0.7)
        slopes.append(slope)
        intercepts.append(intercept)
    
    # Calculate and plot the average model
    if slopes and intercepts:
        avg_slope = np.mean(slopes)
        avg_intercept = np.mean(intercepts)
        avg_y_pred = avg_slope * x_range + avg_intercept
        plt.plot(x_range, avg_y_pred, color='black', linewidth=2, linestyle='--',
                 label=f'Average')
    
    # Add labels, title, and legend
    plt.xlabel('Sentiment Score', fontsize=14)
    plt.ylabel('Stock Price Change (Daily Return)', fontsize=14)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside plot
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent legend overlap
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def plot_articles_per_day_histogram(news_data):
    """
    Plot a histogram of the number of articles published per day across all tickers.
    
    Parameters:
    - news_data: Dictionary of articles by ticker
    """
    # Collect all publication dates
    all_dates = []
    for ticker, articles in news_data.items():
        for article in articles:
            if isinstance(article.get("datetime"), (int, float)):
                try:
                    pub_date = datetime.datetime.fromtimestamp(article["datetime"]).date()
                    all_dates.append(pub_date)
                except (ValueError, OSError):
                    continue
    
    if not all_dates:
        print("No valid publication dates found.")
        return
    
    # Count articles per day
    date_counts = pd.Series(all_dates).value_counts().sort_index()
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(date_counts.values, bins=20, color='skyblue', edgecolor='black')
    
    # Customize
    plt.xlabel('Number of Articles Published on a Single Day', fontsize=14)
    plt.ylabel('Frequency (Number of Days)', fontsize=14)
    plt.title('Histogram of Articles Published Per Day', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()