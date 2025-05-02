import os
import pandas as pd
import numpy as np
import datetime
import pytz
import streamlit as st
import yfinance as yf
import traceback

def init_alpaca_client():
    """Initialize Alpaca API client (placeholder)"""
    st.warning("Alpaca API client not initialized. Using sample data instead.")
    return None

def load_alpaca_data(symbol, timeframe, period):
    """
    Load market data from Alpaca API (placeholder)
    
    Args:
        symbol (str): Trading symbol (e.g., 'AAPL')
        timeframe (str): Time interval (e.g., '1D', '1H', '15Min')
        period (str): Time period to fetch (e.g., '1d', '5d', '1mo')
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    st.warning("Alpaca API data source is not available. Using sample data instead.")
    return load_sample_data(symbol, timeframe, period)

def load_tradingview_data(symbol, timeframe, period):
    """
    Load market data from TradingView (placeholder)
    
    Args:
        symbol (str): Trading symbol (e.g., 'AAPL')
        timeframe (str): Time interval (e.g., '1D', '1H', '15Min')
        period (str): Time period to fetch (e.g., '1d', '5d', '1mo')
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    st.warning("TradingView data source is not available. Using sample data instead.")
    return load_sample_data(symbol, timeframe, period)

def load_sample_data(symbol, timeframe, period):
    """
    Generate sample market data for demonstration
    
    Args:
        symbol (str): Symbol to use for sample data
        timeframe (str): Time interval for the data
        period (str): Period to generate
        
    Returns:
        pd.DataFrame: Sample OHLCV data
    """
    # Map period to number of data points
    period_points = {
        '1d': 24*60,      # 1-minute bars for 1 day
        '5d': 5*24*12,    # 5-minute bars for 5 days
        '1mo': 30*24*4,   # 15-minute bars for 1 month
        '3mo': 90*24,     # 1-hour bars for 3 months
        '6mo': 180*6,     # 4-hour bars for 6 months
        '1y': 365,        # Daily bars for 1 year
        '2y': 2*365,      # Daily bars for 2 years
        '5y': 5*365,      # Daily bars for 5 years
        'all': 10*365     # Daily bars for 10 years
    }
    
    # Default to 30 days of hourly data if period not found
    num_points = period_points.get(period, 30*24)
    
    # Generate timestamps based on timeframe
    end_time = datetime.datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    
    if timeframe == '1m':
        start_time = end_time - datetime.timedelta(minutes=num_points)
        freq = 'min'
    elif timeframe == '5m':
        start_time = end_time - datetime.timedelta(minutes=5*num_points)
        freq = '5min'
    elif timeframe == '15m':
        start_time = end_time - datetime.timedelta(minutes=15*num_points)
        freq = '15min'
    elif timeframe == '30m':
        start_time = end_time - datetime.timedelta(minutes=30*num_points)
        freq = '30min'
    elif timeframe == '1h':
        start_time = end_time - datetime.timedelta(hours=num_points)
        freq = 'H'
    elif timeframe == '4h':
        start_time = end_time - datetime.timedelta(hours=4*num_points)
        freq = '4H'
    else:  # Default to daily
        start_time = end_time - datetime.timedelta(days=num_points)
        freq = 'D'
    
    # Create date range
    timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    # Filter to trading hours
    trading_hours = []
    for ts in timestamps:
        # Skip weekends
        if ts.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            continue
        # Only include times during trading hours (9:30 AM to 4:00 PM ET)
        if freq in ['min', '5min', '15min', '30min', 'H', '4H']:
            if 9 <= ts.hour < 16 or (ts.hour == 9 and ts.minute >= 30):
                trading_hours.append(ts)
        else:
            trading_hours.append(ts)
    
    timestamps = pd.DatetimeIndex(trading_hours)
    
    # Generate price data
    np.random.seed(42)  # For reproducible results
    
    # Start price between $10 and $1000
    base_price = 100.0
    if symbol:
        # Use symbol to create different price series for different symbols
        # Simple hash of the symbol name to get a number
        symbol_hash = sum(ord(c) for c in symbol)
        base_price = 50 + (symbol_hash % 20) * 50  # Price between $50 and $1000
    
    # Generate a random walk for prices
    noise = np.random.normal(0, 1, len(timestamps))
    trend = np.linspace(0, 0.2, len(timestamps))  # Slight upward trend
    price_changes = noise * 0.01 * base_price + trend
    
    # Generate the OHLC data
    data = pd.DataFrame(index=timestamps)
    data['close'] = base_price + np.cumsum(price_changes)
    
    # Generate realistic intraday patterns
    high_low_range = base_price * 0.01 * (0.5 + 0.5 * np.random.rand(len(timestamps)))
    data['open'] = data['close'].shift(1).fillna(base_price)
    data['high'] = data[['open', 'close']].max(axis=1) + high_low_range * 0.6
    data['low'] = data[['open', 'close']].min(axis=1) - high_low_range * 0.6
    
    # Generate volume with intraday patterns (higher at open and close)
    base_volume = 10000
    time_of_day_factor = np.ones(len(timestamps))
    
    for i, ts in enumerate(timestamps):
        hour = ts.hour + ts.minute/60.0
        if hour < 10.5:  # Higher volume at open
            time_of_day_factor[i] = 2.0
        elif hour > 15.0:  # Higher volume at close
            time_of_day_factor[i] = 1.8
        else:
            # Lower volume mid-day
            mid_day_dip = 1.0 - 0.5 * np.sin(np.pi * (hour - 10.5) / 4.5)
            time_of_day_factor[i] = mid_day_dip
    
    volume = base_volume * time_of_day_factor * (0.5 + 1 * np.random.rand(len(timestamps)))
    volume = volume * np.sqrt(abs(price_changes) + 0.001)  # Higher volume on larger price moves
    data['volume'] = volume.astype(int)
    
    # Add some gaps in the data
    gap_indices = np.random.choice(len(data), int(len(data) * 0.01), replace=False)
    data = data.drop(data.index[gap_indices])
    
    return data

def load_market_data(symbol, timeframe="1h", date_range="1 Week"):
    """
    Load market data for the given symbol and timeframe
    
    Args:
        symbol (str or tuple): Symbol name or (symbol, info) tuple from symbol selection
        timeframe (str): Timeframe for data (e.g., "1m", "5m", "15m", "30m", "1h", "4h", "1d")
        date_range (str): Date range to fetch (e.g., "1 Day", "1 Week", "1 Month", "3 Months")
        
    Returns:
        pd.DataFrame: OHLCV data for the symbol
    """
    try:
        # Extract symbol from tuple if needed
        if isinstance(symbol, tuple):
            symbol = symbol[0]
        
        # Map date range string to Yahoo Finance period
        if date_range == "Last 1 Day":
            period = "1d"
        elif date_range == "Last 3 Days":
            period = "3d"
        elif date_range == "Last 5 Days":
            period = "5d"
        elif date_range == "Last Week":
            period = "7d"
        elif date_range == "Last Month":
            period = "1mo"
        elif date_range == "All Data":
            period = "max"
        else:
            period = "7d"  # Default to 1 week
        
        # Map timeframe to Yahoo Finance interval
        if timeframe == "1m":
            interval = "1m"
            # For 1m data, Yahoo only allows 7 days max
            if period not in ["1d", "3d", "5d", "7d"]:
                st.warning("For 1-minute data, Yahoo Finance only provides up to 7 days. Using 7 days.")
                period = "7d"
        elif timeframe == "5m":
            interval = "5m"
        elif timeframe == "15m":
            interval = "15m"
        elif timeframe == "30m":
            interval = "30m"
        elif timeframe == "1h":
            interval = "60m"
        elif timeframe == "4h":
            # Yahoo doesn't have 4h, use 1h and resample later
            interval = "60m"
        elif timeframe == "1d":
            interval = "1d"
        else:
            interval = "60m"  # Default to 1h
        
        # Download data from Yahoo Finance
        with st.spinner(f"Loading {symbol} data..."):
            data = yf.download(symbol, period=period, interval=interval)
        
        # Check if we got data
        if data.empty:
            st.error(f"No data available for {symbol}")
            return None
        
        # Resample to 4h if needed
        if timeframe == "4h" and interval == "60m":
            data = data.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        # Normalize column names
        data.columns = [col.lower() for col in data.columns]
        
        # Add timezone info if missing
        if isinstance(data.index[0], pd.Timestamp) and data.index[0].tzinfo is None:
            eastern = pytz.timezone('US/Eastern')
            data.index = data.index.tz_localize('UTC').tz_convert(eastern)
        
        # Store in session state for reuse
        st.session_state.data = data
        st.session_state.symbol = symbol
        
        return data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        traceback.print_exc()
        return None

def load_days_data(symbol, days=1, timeframe='5m'):
    """
    Load market data for a specific number of days
    
    Args:
        symbol (str): Trading symbol
        days (int): Number of days of data to load
        timeframe (str): Time interval (default: '5m')
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    if days <= 0:
        days = 1
    
    # Map days to an appropriate period
    if days <= 1:
        period = '1d'
    elif days <= 5:
        period = '5d'
    elif days <= 30:
        period = '1mo'
    elif days <= 90:
        period = '3mo'
    else:
        period = '1y'
    
    data = load_sample_data(symbol, timeframe, period)
    
    # Limit to the exact number of days if needed
    if data is not None and len(data) > 0:
        end_time = data.index[-1]
        start_time = end_time - datetime.timedelta(days=days)
        data = data[data.index >= start_time]
    
    return data

def load_symbol_data(symbol, days=1, timeframe='5m'):
    """
    Load market data for a specific symbol
    
    Args:
        symbol (str): Trading symbol
        days (int): Number of days of data to load
        timeframe (str): Time interval (default: '5m')
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    try:
        # For SPY, we might want to load from a special source
        if symbol.upper() == "SPY":
            # Use our special SPY optimizer
            data = load_days_data(symbol, days, timeframe)
        else:
            # Use standard data loading
            data = load_days_data(symbol, days, timeframe)
        
        return data
    except Exception as e:
        print(f"Error loading data for {symbol}: {str(e)}")
        return None 