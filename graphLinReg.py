
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
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.show()