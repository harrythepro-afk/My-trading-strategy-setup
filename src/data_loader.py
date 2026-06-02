import pandas as pd
import requests
import time
from datetime import datetime
import yfinance as yf

def conditional_cache(*cache_args, **cache_kwargs):
    """
    Applies Streamlit's st.cache_data decorator only when running inside a Streamlit runtime environment.
    This prevents warnings and overhead when running external test scripts from the CLI.
    """
    def decorator(func):
        try:
            import streamlit as st
            if st.runtime.exists():
                return st.cache_data(*cache_args, **cache_kwargs)(func)
        except Exception:
            pass
        return func
    return decorator

@conditional_cache(ttl=86400, show_spinner=False)
def fetch_all_usdt_symbols() -> list:
    """
    Fetches all active USDT trading pairs from the public Binance exchangeInfo API.
    Sorts them alphabetically and places BTCUSDT and ETHUSDT at the top.
    """
    url = "https://api.binance.com/api/v3/exchangeInfo"
    print("Loading live trading symbols from Binance...")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        symbols = [
            s["symbol"] for s in data["symbols"]
            if s["symbol"].endswith("USDT") and s["status"] == "TRADING"
        ]
        
        symbols.sort()
        
        if "BTCUSDT" in symbols:
            symbols.remove("BTCUSDT")
            symbols.insert(0, "BTCUSDT")
        if "ETHUSDT" in symbols:
            symbols.remove("ETHUSDT")
            symbols.insert(1, "ETHUSDT")
            
        print(f"Loaded {len(symbols)} active USDT pairs successfully!")
        return symbols
        
    except Exception as e:
        print(f"Error fetching symbols from Binance: {e}. Using fallback list.")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "DOTUSDT"]

@conditional_cache(ttl=3600, show_spinner=False)
def fetch_nse_history(ticker: str, days: int = 45) -> pd.DataFrame:
    """
    Downloads historical 15m intraday candle data for Indian Markets (NSE) using Yahoo Finance.
    Yahoo Finance limits 15-minute interval historical data to the last 60 days.
    """
    print(f"Fetching {days} days of 15m data for {ticker} from Yahoo Finance...")
    
    # Cap Yahoo Finance 15m downloads to 59 days to avoid server-side threshold errors
    days_capped = min(59, days)
    
    try:
        # Download data using yfinance
        stock = yf.Ticker(ticker.upper())
        # period formats: e.g. "60d"
        raw_df = stock.history(period=f"{days_capped}d", interval="15m")
        
        if raw_df.empty:
            print(f"Error: Empty DataFrame returned for {ticker} from yfinance.")
            return pd.DataFrame()
            
        # Reset index to make the DatetimeIndex a standard column
        df = raw_df.reset_index()
        
        # Standardize columns: open, high, low, close, volume, timestamp
        df = df.rename(columns={
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        # Select only the core required columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        
        # Strip timezone info from timestamps to prevent timezone merge mismatch errors
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        
        print(f"Successfully loaded {len(df)} candles for {ticker} (NSE).")
        return df
        
    except Exception as e:
        print(f"Error loading NSE history for {ticker}: {e}")
        return pd.DataFrame()

def fetch_binance_klines(symbol: str, interval: str = "15m", limit: int = 1000) -> pd.DataFrame:
    """
    Fetches historical candlestick (kline) data from Binance public API.
    """
    url = "https://api.binance.com/api/v3/klines"
    
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }
    
    print(f"Fetching {limit} candles of {interval} data for {symbol} from Binance API...")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        raw_data = response.json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
        
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    formatted_data = []
    
    for candle in raw_data:
        formatted_candle = [
            candle[0],
            float(candle[1]),
            float(candle[2]),
            float(candle[3]),
            float(candle[4]),
            float(candle[5])
        ]
        formatted_data.append(formatted_candle)
        
    df = pd.DataFrame(formatted_data, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"] / 1000, unit="s")
    
    return df

@conditional_cache(ttl=3600, show_spinner=False)
def fetch_extended_history(symbol: str, interval: str = "15m", days: int = 30) -> pd.DataFrame:
    """
    Downloads historical candle data from Binance public API using a high-speed parallel ThreadPoolExecutor.
    """
    from concurrent.futures import ThreadPoolExecutor
    
    if interval == "15m":
        candles_per_day = 24 * 4
        candle_duration_ms = 15 * 60 * 1000
    elif interval == "1h":
        candles_per_day = 24
        candle_duration_ms = 60 * 60 * 1000
    elif interval == "4h":
        candles_per_day = 6
        candle_duration_ms = 4 * 60 * 60 * 1000
    elif interval == "1d":
        candles_per_day = 1
        candle_duration_ms = 24 * 60 * 60 * 1000
    else:
        candles_per_day = 24 * 4
        candle_duration_ms = 15 * 60 * 1000
        
    total_candles_needed = days * candles_per_day
    chunk_size = 1000
    chunk_duration_ms = chunk_size * candle_duration_ms
    
    end_time_ms = int(time.time() * 1000)
    start_time_ms = end_time_ms - (total_candles_needed * candle_duration_ms)
    
    # Calculate start times for each chunk
    chunk_starts = []
    t = start_time_ms
    while t < end_time_ms:
        chunk_starts.append(t)
        t += chunk_duration_ms
        
    url = "https://api.binance.com/api/v3/klines"
    
    def fetch_chunk(start_ms):
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ms,
            "limit": chunk_size
        }
        for _ in range(3): # Retry mechanism up to 3 times
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except Exception:
                time.sleep(0.2)
        return []
        
    # Fetch all chunks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(10, len(chunk_starts) or 1)) as executor:
        results = list(executor.map(fetch_chunk, chunk_starts))
        
    all_candles = []
    for r in results:
        all_candles.extend(r)
        
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    formatted_data = []
    for candle in all_candles:
        formatted_candle = [
            pd.to_datetime(candle[0] / 1000, unit="s"),
            float(candle[1]),
            float(candle[2]),
            float(candle[3]),
            float(candle[4]),
            float(candle[5])
        ]
        formatted_data.append(formatted_candle)
        
    df = pd.DataFrame(formatted_data, columns=columns)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    
    # Slice the dataframe to ensure it matches the exact timeline days requested
    if len(df) > total_candles_needed:
        df = df.iloc[-total_candles_needed:].reset_index(drop=True)
        
    print(f"Successfully loaded {len(df)} candles for {symbol} ({days} days).")
    return df

if __name__ == "__main__":
    test_df = fetch_nse_history("RELIANCE.NS", 5)
    print(test_df.head())
