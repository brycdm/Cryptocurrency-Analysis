import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# price data from CoinGecko API
def fetch_crypto_data(crypto_id, vs_currency="usd", days="365"):
  
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    response = requests.get(url, params=params)
    
    # http errors
    if response.status_code != 200:
        print(f"Error: Unable to fetch data for {crypto_id}. HTTP {response.status_code}")
        return pd.DataFrame()  

    data = response.json()
    # checking 'prices' key
    if 'prices' not in data:
        print(f"Error: 'prices' key not found in response for {crypto_id}. Response: {data}")
        return pd.DataFrame() 

    # data prices
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
    prices.set_index('timestamp', inplace=True)
    return prices

# data for cryptocurrencies
crypto_ids = ["bitcoin", "ethereum", "ripple"]  # 'ripple' corresponds to XRP in CoinGecko API
crypto_data = {}

for crypto in crypto_ids:
    print(f"Fetching data for {crypto}...")
    df = fetch_crypto_data(crypto)
    if df.empty:
        print(f"Skipping {crypto} due to missing data.")
        continue
    crypto_data[crypto] = df

if crypto_data:
    # calculating daily changes, volatility, and sharpe ratio
    metrics = []  # metrics for each cryptocurrency
    risk_free_rate = 2  # assumed risk-free rate in percentage

    for crypto, df in crypto_data.items():
        df['daily_change'] = df['price'].pct_change() * 100  # daily percentage change
        volatility = df['daily_change'].std()  # volatility (standard deviation)
        avg_return = df['daily_change'].mean()  # average daily return
        sharpe_ratio = (avg_return - risk_free_rate) / volatility  # sharpe Ratio
        metrics.append({
            "Cryptocurrency": crypto.capitalize(),
            "Volatility (%)": volatility,
            "Sharpe Ratio": sharpe_ratio
        })
        print(f"{crypto.capitalize()} Volatility: {volatility:.2f}%, Sharpe Ratio: {sharpe_ratio:.2f}")

    # summary dataframe
    metrics_df = pd.DataFrame(metrics)
    print("\nSummary Metrics:")
    print(metrics_df)

    # single dataframe
    combined_data = pd.concat(
        {crypto: df[['price', 'daily_change']] for crypto, df in crypto_data.items()},
        axis=1
    )

    # forward-filling missing values
    combined_data = combined_data.ffill()

    # price trends in subplots
    fig, axes = plt.subplots(len(crypto_ids), 1, figsize=(10, 8), sharex=True)
    for i, crypto in enumerate(crypto_ids):
        if crypto in crypto_data:
            axes[i].plot(combined_data.index, combined_data[crypto]['price'], label=crypto.capitalize(), color=f"C{i}")
            axes[i].set_title(f"{crypto.capitalize()} Price Trend")
            axes[i].set_ylabel("Price (USD)")
            axes[i].legend()
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

    # individual graphs for each cryptocurrency
    for crypto in crypto_ids:
        if crypto in crypto_data:
            plt.figure(figsize=(10, 6))
            plt.plot(crypto_data[crypto].index, crypto_data[crypto]['price'], label=f"{crypto.capitalize()} Price", color='blue')
            plt.title(f"{crypto.capitalize()} Price Trend")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            plt.grid(True)
            plt.show()

    # histogram of daily percentage changes for each cryptocurrency
    for crypto in crypto_ids:
        if crypto in crypto_data:
            plt.figure(figsize=(8, 5))
            plt.hist(crypto_data[crypto]['daily_change'].dropna(), bins=50, edgecolor='black', alpha=0.7)
            plt.title(f"{crypto.capitalize()} Daily Percentage Change Distribution")
            plt.xlabel("Daily Change (%)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

    # correlation matrix for daily percentage changes
    daily_changes = pd.concat({crypto: df['daily_change'] for crypto, df in crypto_data.items()}, axis=1)
    correlation_matrix = daily_changes.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Matrix of Daily Percentage Changes")
    plt.show()

    # insights
    most_volatile = metrics_df.loc[metrics_df['Volatility (%)'].idxmax()]
    best_risk_adjusted = metrics_df.loc[metrics_df['Sharpe Ratio'].idxmax()]
    print(f"\nMost Volatile Cryptocurrency: {most_volatile['Cryptocurrency']} ({most_volatile['Volatility (%)']:.2f}%)")
    print(f"Best Risk-Adjusted Cryptocurrency: {best_risk_adjusted['Cryptocurrency']} (Sharpe Ratio: {best_risk_adjusted['Sharpe Ratio']:.2f})")
else:
    print("No data was fetched for analysis.")
