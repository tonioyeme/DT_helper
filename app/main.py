import streamlit as st

# Set page configuration - this must be the first Streamlit command
st.set_page_config(
    page_title="Day Trading Helper",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from io import StringIO
import os
import ccxt
from alpaca.data import StockHistoricalDataClient, TimeFrame as AlpacaTimeFrame, StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
import pytz
from dotenv import load_dotenv

from app.components import (
    render_dashboard,
    render_signal_table,
    render_risk_analysis,
    render_risk_calculator,
    render_patterns_section,
    render_backtest,
    render_signals,
    render_signal_chart
)
from app.tradingview import (
    generate_tradingview_chart_url,
    generate_indicator_script
)
from app.signals import generate_signals
from app.data.loader import (
    load_alpaca_data,
    load_tradingview_data,
    load_sample_data
)

# Load environment variables for API keys
load_dotenv()

# Initialize session state for cached data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'signals' not in st.session_state:
    st.session_state.signals = None
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'AAPL'
if 'exchange' not in st.session_state:
    st.session_state.exchange = 'NASDAQ'

def configure_api_keys():
    """Configure Alpaca API keys"""
    st.title("API Key Configuration")
    st.markdown("You need to set up your Alpaca API keys to access market data.")
    
    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    
    with st.form("api_key_form"):
        new_api_key = st.text_input("Alpaca API Key", value=api_key, type="password")
        new_api_secret = st.text_input("Alpaca API Secret", value=api_secret, type="password")
        submit = st.form_submit_button("Save Keys")
        
        if submit:
            if new_api_key and new_api_secret:
                # Save keys to environment variables for current session
                os.environ["ALPACA_API_KEY"] = new_api_key
                os.environ["ALPACA_API_SECRET"] = new_api_secret
                
                # Create or update .env file
                with open(".env", "w") as f:
                    f.write(f"ALPACA_API_KEY={new_api_key}\n")
                    f.write(f"ALPACA_API_SECRET={new_api_secret}\n")
                
                st.success("API keys saved successfully!")
                st.experimental_rerun()
            else:
                st.error("Both API Key and Secret are required")
    
    # Provide info about how to get API keys
    with st.expander("How to get Alpaca API keys"):
        st.markdown("""
        1. Create an account at [Alpaca](https://app.alpaca.markets/signup)
        2. Navigate to Paper Trading in your dashboard
        3. Click on "Generate New Key"
        4. Copy the API Key and Secret Key
        
        You can use the Paper Trading API for free to access market data and practice trading.
        """)

def init_alpaca_client():
    """Initialize Alpaca API client"""
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        st.sidebar.error("Alpaca API keys not found. Please set ALPACA_API_KEY and ALPACA_API_SECRET in .env file")
        # Create expandable section to configure API keys
        with st.sidebar.expander("Configure Alpaca API Keys"):
            api_key = st.text_input("Alpaca API Key", type="password", 
                                     value=api_key if api_key else "", key="sidebar_api_key")
            api_secret = st.text_input("Alpaca API Secret", type="password", 
                                        value=api_secret if api_secret else "", key="sidebar_api_secret")
            if st.button("Save API Keys"):
                if api_key and api_secret:
                    os.environ["ALPACA_API_KEY"] = api_key
                    os.environ["ALPACA_API_SECRET"] = api_secret
                    st.success("API keys saved for this session")
                else:
                    st.error("Both API Key and Secret are required")
    
    try:
        data_client = StockHistoricalDataClient(api_key, api_secret)
        trading_client = TradingClient(api_key, api_secret, paper=True)
        return data_client, trading_client
    except Exception as e:
        st.sidebar.error(f"Error initializing Alpaca client: {str(e)}")
        return None, None

def fetch_data(symbol, exchange, timeframe, period='3mo'):
    """
    Fetch market data from various sources
    
    Args:
        symbol (str): Trading symbol
        exchange (str): Exchange name
        timeframe (str): Data timeframe
        period (str): Period to fetch
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    try:
        # For stock market data, use Alpaca
        if exchange in ['NASDAQ', 'NYSE', 'AMEX']:
            # Initialize Alpaca client
            data_client, _ = init_alpaca_client()
            if not data_client:
                st.error("Alpaca client not initialized. Please check API keys.")
                return None
            
            # Convert timeframe string to Alpaca TimeFrame format
            original_timeframe = timeframe
            
            # Parse the timeframe string to extract number and unit
            if timeframe.endswith('m'):
                if timeframe == '1m':
                    alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Minute)
                elif timeframe == '5m':
                    # Will resample 1-minute data to 5m later for some period lengths
                    alpaca_timeframe = AlpacaTimeFrame(5, AlpacaTimeFrame.Minute)
                elif timeframe == '15m':
                    alpaca_timeframe = AlpacaTimeFrame(15, AlpacaTimeFrame.Minute)
                elif timeframe == '30m':
                    alpaca_timeframe = AlpacaTimeFrame(30, AlpacaTimeFrame.Minute)
                else:
                    # Default to minute for unknown minute-based timeframes
                    alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Minute)
            elif timeframe.endswith('h'):
                if timeframe == '1h':
                    alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Hour)
                elif timeframe == '4h':
                    alpaca_timeframe = AlpacaTimeFrame(4, AlpacaTimeFrame.Hour)
                else:
                    alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Hour)
            elif timeframe == '1d':
                alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Day)
            else:
                # Default to 1 Hour for unknown timeframes
                alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Hour)
                st.warning(f"Unknown timeframe format: {timeframe}. Using 1 Hour as default.")
            
            # Calculate start and end times based on period
            # For free accounts, we need to use data that's at least 15 minutes old
            # Use Eastern Time (ET) instead of UTC
            eastern_tz = pytz.timezone('US/Eastern')
            current_time = datetime.now(eastern_tz)
            
            # Create end time 16 minutes in the past to comply with Alpaca free tier
            end = current_time - timedelta(minutes=16)
            
            if period == '1d':
                start = end - timedelta(days=1)
            elif period == '5d':
                start = end - timedelta(days=5)
            elif period == '1wk':
                start = end - timedelta(weeks=1)
            elif period == '1mo':
                start = end - timedelta(days=30)
            elif period == '3mo':
                start = end - timedelta(days=90)
            elif period == '6mo':
                start = end - timedelta(days=180)
            elif period == '1y':
                start = end - timedelta(days=365)
            else:
                start = end - timedelta(days=90)  # Default to 3 months
            
            # Adjust timeframe based on period length to avoid hitting limits
            if (end - start).days > 30 and alpaca_timeframe.value == AlpacaTimeFrame.Minute.value:
                # Use hourly data for periods > 30 days if minute data was requested
                alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Hour)
                st.info(f"Switched to hourly data for {timeframe} due to long period ({period})")
            elif (end - start).days > 90 and alpaca_timeframe.value == AlpacaTimeFrame.Hour.value:
                # Use daily data for periods > 90 days
                alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Day)
                st.info(f"Switched to daily data for {timeframe} due to long period ({period})")
            
            # Create request with proper timeframe
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=alpaca_timeframe,
                start=start,
                end=end,
                adjustment='all'
            )
            
            try:
                # Fetch bars data
                bars = data_client.get_stock_bars(request_params)
                
                # Convert to dataframe
                if bars and bars.data and symbol in bars.data:
                    df = bars.df.loc[symbol].copy()
                    
                    # Rename columns to match expected format
                    df = df.rename(columns={
                        'open': 'open',
                        'high': 'high', 
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume',
                        'trade_count': 'trade_count',
                        'vwap': 'vwap'
                    })
                    
                    # Resample to the original requested timeframe if needed
                    if original_timeframe == '5m' and alpaca_timeframe.value == AlpacaTimeFrame.Minute.value and alpaca_timeframe.amount == 1:
                        df = df.resample('5T').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    elif original_timeframe == '15m' and alpaca_timeframe.value == AlpacaTimeFrame.Minute.value and alpaca_timeframe.amount == 1:
                        df = df.resample('15T').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    elif original_timeframe == '30m' and alpaca_timeframe.value == AlpacaTimeFrame.Minute.value and alpaca_timeframe.amount == 1:
                        df = df.resample('30T').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    elif original_timeframe == '1h' and alpaca_timeframe.value == AlpacaTimeFrame.Minute.value:
                        df = df.resample('1H').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    elif original_timeframe == '2h' and alpaca_timeframe.value == AlpacaTimeFrame.Hour.value and alpaca_timeframe.amount == 1:
                        df = df.resample('2H').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    elif original_timeframe == '4h' and alpaca_timeframe.value == AlpacaTimeFrame.Hour.value and alpaca_timeframe.amount == 1:
                        df = df.resample('4H').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    
                    return df
                else:
                    st.error(f"No data returned for {symbol}")
                    return None
            except Exception as e:
                error_msg = str(e)
                if "subscription does not permit" in error_msg:
                    st.error("Your Alpaca account subscription doesn't permit access to this data.")
                    st.info("Trying with a longer delay or older data...")
                    
                    # Try with a longer delay (15 hours ago)
                    eastern_tz = pytz.timezone('US/Eastern')
                    current_time = datetime.now(eastern_tz)
                    end_older = current_time - timedelta(hours=15)
                    start_older = end_older - timedelta(days=30)
                    
                    # Use daily timeframe for older data
                    request_params = StockBarsRequest(
                        symbol_or_symbols=[symbol],
                        timeframe=AlpacaTimeFrame(1, AlpacaTimeFrame.Day),
                        start=start_older,
                        end=end_older,
                        adjustment='all'
                    )
                    
                    try:
                        bars = data_client.get_stock_bars(request_params)
                        if bars and bars.data and symbol in bars.data:
                            df = bars.df.loc[symbol].copy()
                            st.success(f"Successfully retrieved historical data for {symbol}")
                            return df
                        else:
                            st.error(f"No historical data available for {symbol}")
                            return None
                    except Exception as e2:
                        st.error(f"Error accessing historical data: {str(e2)}")
                        st.info("Loading sample data instead")
                        return load_sample_data(symbol, timeframe, period)
                else:
                    st.error(f"Error fetching data: {error_msg}")
                    return None
        
        # For crypto exchanges
        elif exchange in ['BINANCE', 'COINBASE', 'FTX']:
            # Initialize the exchange
            if exchange == 'BINANCE':
                ex = ccxt.binance()
            elif exchange == 'COINBASE':
                ex = ccxt.coinbasepro()
            elif exchange == 'FTX':
                ex = ccxt.ftx()
            
            # Convert timeframe to exchange format
            if timeframe == '1':
                tf = '1m'
            elif timeframe == '5':
                tf = '5m'
            elif timeframe == '15':
                tf = '15m'
            elif timeframe == '30':
                tf = '30m'
            elif timeframe == '60':
                tf = '1h'
            elif timeframe == 'D':
                tf = '1d'
            elif timeframe == 'W':
                tf = '1w'
            else:
                tf = '1h'
            
            # Calculate start time based on period
            if period == '1d':
                start_time = datetime.now() - timedelta(days=1)
            elif period == '1wk':
                start_time = datetime.now() - timedelta(weeks=1)
            elif period == '1mo':
                start_time = datetime.now() - timedelta(days=30)
            elif period == '3mo':
                start_time = datetime.now() - timedelta(days=90)
            else:
                start_time = datetime.now() - timedelta(days=30)
            
            # Convert to millisecond timestamp
            since = int(start_time.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = ex.fetch_ohlcv(symbol + '/USDT', tf, since=since)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        else:
            st.error(f"Unsupported exchange: {exchange}")
            return None
            
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def load_sample_data(symbol="AAPL", timeframe="1d", period="3mo"):
    """
    Load sample data for demonstration
    
    Args:
        symbol (str): Trading symbol (defaults to AAPL)
        timeframe (str): Data timeframe (defaults to 1d)
        period (str): Period to fetch (defaults to 3mo)
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    try:
        # Initialize Alpaca client
        data_client, _ = init_alpaca_client()
        if not data_client:
            st.error("Alpaca client not initialized. Please check API keys.")
            # Fall back to a basic sample dataset if Alpaca is not available
            eastern_tz = pytz.timezone('US/Eastern')
            current_time = datetime.now(eastern_tz)
            dates = pd.date_range(end=current_time, periods=90)
            data = pd.DataFrame({
                'open': np.random.normal(100, 5, 90),
                'high': np.random.normal(105, 5, 90),
                'low': np.random.normal(95, 5, 90),
                'close': np.random.normal(100, 5, 90),
                'volume': np.random.normal(1000000, 200000, 90),
            }, index=dates)
            for i in range(1, len(data)):
                data.loc[data.index[i], 'open'] = data.loc[data.index[i-1], 'close']
                data.loc[data.index[i], 'high'] = max(data.loc[data.index[i], 'open'], data.loc[data.index[i], 'close']) + np.random.normal(2, 0.5)
                data.loc[data.index[i], 'low'] = min(data.loc[data.index[i], 'open'], data.loc[data.index[i], 'close']) - np.random.normal(2, 0.5)
            return data
        
        # Request data based on symbol - use data that's 16 minutes old for free accounts
        eastern_tz = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern_tz)
        end = current_time - timedelta(minutes=16)
        
        # Calculate start time based on period
        if period == '1d':
            start = end - timedelta(days=1)
        elif period == '5d':
            start = end - timedelta(days=5)
        elif period == '1mo':
            start = end - timedelta(days=30)
        elif period == '3mo':
            start = end - timedelta(days=90)
        elif period == '6mo':
            start = end - timedelta(days=180)
        elif period == '1y':
            start = end - timedelta(days=365)
        else:
            start = end - timedelta(days=90)  # Default to 3 months
        
        # Convert timeframe to Alpaca format
        if timeframe.endswith('m'):
            if timeframe == '1m':
                alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Minute)
            elif timeframe == '5m':
                alpaca_timeframe = AlpacaTimeFrame(5, AlpacaTimeFrame.Minute)
            elif timeframe == '15m':
                alpaca_timeframe = AlpacaTimeFrame(15, AlpacaTimeFrame.Minute)
            elif timeframe == '30m':
                alpaca_timeframe = AlpacaTimeFrame(30, AlpacaTimeFrame.Minute)
            else:
                alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Minute)
        elif timeframe.endswith('h'):
            if timeframe == '1h':
                alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Hour)
            elif timeframe == '4h':
                alpaca_timeframe = AlpacaTimeFrame(4, AlpacaTimeFrame.Hour)
            else:
                alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Hour)
        elif timeframe == '1d':
            alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Day)
        else:
            # Default to daily data
            alpaca_timeframe = AlpacaTimeFrame(1, AlpacaTimeFrame.Day)
        
        # Request params based on the provided symbol
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=alpaca_timeframe,
            start=start,
            end=end,
            adjustment='all'
        )
        
        # Fetch bars data
        bars = data_client.get_stock_bars(request_params)
        
        # Convert to dataframe
        if bars and bars.data and symbol in bars.data:
            df = bars.df.loc[symbol].copy()
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'trade_count': 'trade_count',
                'vwap': 'vwap'
            })
            
            return df
        else:
            raise Exception(f"No sample data available from Alpaca for {symbol}")
            
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        # Create synthetic sample data as fallback
        eastern_tz = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern_tz)
        dates = pd.date_range(end=current_time, periods=90)
        data = pd.DataFrame({
            'open': np.random.normal(100, 5, 90),
            'high': np.random.normal(105, 5, 90),
            'low': np.random.normal(95, 5, 90),
            'close': np.random.normal(100, 5, 90),
            'volume': np.random.normal(1000000, 200000, 90),
        }, index=dates)
        for i in range(1, len(data)):
            data.loc[data.index[i], 'open'] = data.loc[data.index[i-1], 'close']
            data.loc[data.index[i], 'high'] = max(data.loc[data.index[i], 'open'], data.loc[data.index[i], 'close']) + np.random.normal(2, 0.5)
            data.loc[data.index[i], 'low'] = min(data.loc[data.index[i], 'open'], data.loc[data.index[i], 'close']) - np.random.normal(2, 0.5)
        return data

def main():
    """Main function to run the Day Trading Helper application"""
    
    # Check if API keys are configured
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        # Show API configuration screen if keys are missing
        configure_api_keys()
        return
    
    # Create a sidebar for configuration options
    with st.sidebar:
        st.title("Day Trading Helper")
        st.subheader("Configuration")
        
        # Create tabs for different configuration sections
        config_tabs = st.tabs(["Data Source", "Analysis", "Tools"])
        
        with config_tabs[0]:
            # Data source selection
            data_source = st.selectbox(
                "Select Data Source",
                ["Alpaca API", "Sample Data"],  # Add back real data sources
                index=0
            )
            
            # Symbol selection
            symbol = st.text_input("Symbol", "AAPL").upper()
            
            # Exchange selection (only show if using real data)
            if data_source == "Alpaca API":
                exchange = st.selectbox(
                    "Exchange",
                    ["NASDAQ", "NYSE", "AMEX"],
                    index=0
                )
            else:
                exchange = "NASDAQ"  # Default for sample data
            
            # Timeframe selection
            timeframe = st.selectbox(
                "Select Timeframe",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=4
            )
            
            # Period selection
            period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "all"]
            period = st.selectbox("Select Period", period_options, index=3)
            
            # Fetch data button
            fetch_data_button = st.button("Fetch Data", use_container_width=True)
            
            # Add Alpaca API configuration button
            if st.button("Configure API Keys"):
                configure_api_keys()
                st.experimental_rerun()
        
        with config_tabs[1]:
            # Analysis options
            show_signals = st.checkbox("Show Trading Signals", value=True)
            show_patterns = st.checkbox("Show Chart Patterns", value=True)
            
            # Add market hours filter for signals
            market_hours_filter = st.checkbox("Filter Signals to Market Hours", value=True)
            
            # Add volatility threshold
            volatility_threshold = st.slider(
                "Volatility Threshold",
                min_value=0.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                help="Minimum ATR percentage for signal generation"
            )
        
        with config_tabs[2]:
            # Trading tools
            st.subheader("Trading Tools")
            risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)
            stop_loss_atr = st.slider("Stop Loss (ATR Multiple)", 1.0, 5.0, 2.0, 0.5)
            take_profit_ratio = st.slider("Risk:Reward Ratio", 1.0, 5.0, 2.0, 0.5)
            
            # Backtest settings
            st.subheader("Backtesting")
            strategy = st.selectbox(
                "Backtest Strategy",
                ["Signal-based", "Moving Average Crossover", "RSI Oscillator"]
            )
            
            # Notification settings
            st.subheader("Notifications")
            enable_notifications = st.checkbox("Enable Alert Notifications")
        
        # Navigation links
        st.sidebar.markdown("---")
        st.sidebar.subheader("Navigation")
        st.sidebar.markdown("[📊 Dashboard](http://localhost:8501/)")
        st.sidebar.markdown("[🔍 Signal Analysis](/Signal_Analysis)")
        st.sidebar.markdown("[⚖️ Risk Calculator](/Risk_Calculator)")
        
        # Add app information
        st.sidebar.markdown("---")
        st.sidebar.info(
            "This application helps day traders analyze stocks, "
            "generate trading signals, and manage risk effectively."
        )
        st.sidebar.markdown("v1.0.1 | [Source Code](https://github.com/yourusername/day-trade-helper)")

    # Initialize or get session state for data
    if 'data' not in st.session_state or fetch_data_button:
        try:
            # Use real or sample data based on selection
            if data_source == "Alpaca API":
                st.session_state.data = fetch_data(symbol, exchange, timeframe, period)
                if st.session_state.data is None:
                    st.error(f"Failed to fetch data for {symbol} from Alpaca API. Falling back to sample data.")
                    st.session_state.data = load_sample_data(symbol, timeframe, period)
            else:
                st.session_state.data = load_sample_data(symbol, timeframe, period)
                
            st.session_state.symbol = symbol
            st.session_state.timeframe = timeframe
            st.session_state.signals = None  # Reset signals when new data is loaded
            
            # Show success message for data loading
            st.success(f"Successfully loaded {symbol} data for {timeframe} timeframe")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state.data = None
    
    # Create tabs for different content areas
    if 'data' in st.session_state and st.session_state.data is not None:
        # Fix for 'bool' object has no attribute 'empty'
        if isinstance(st.session_state.data, bool):
            st.error("Invalid data format. Please try again.")
            return
            
        tabs = st.tabs(["Dashboard", "Signals", "Position Sizing", "Backtest"])
        
        # Tab 1: Dashboard - Consolidated view, no duplicate subsections
        with tabs[0]:
            st.header("Trading Dashboard")
            
            # Main metrics and overview
            col1, col2, col3 = st.columns(3)
            with col1:
                # Current price and change
                latest_price = st.session_state.data['close'].iloc[-1]
                prev_price = st.session_state.data['close'].iloc[-2] if len(st.session_state.data) > 1 else latest_price
                price_change = latest_price - prev_price
                percent_change = (price_change / prev_price) * 100 if prev_price > 0 else 0
                st.metric(
                    f"{st.session_state.symbol} Price", 
                    f"${latest_price:.2f}", 
                    f"{percent_change:.2f}%"
                )
            
            with col2:
                # Volume
                latest_volume = st.session_state.data['volume'].iloc[-1]
                avg_volume = st.session_state.data['volume'].mean()
                volume_ratio = (latest_volume / avg_volume) - 1 if avg_volume > 0 else 0
                st.metric(
                    "Volume", 
                    f"{int(latest_volume):,}", 
                    f"{volume_ratio:.2f}%"
                )
            
            with col3:
                # Price range
                day_high = st.session_state.data['high'].iloc[-1]
                day_low = st.session_state.data['low'].iloc[-1]
                st.metric("Day Range", f"${day_low:.2f} - ${day_high:.2f}")
            
            # Brief signal summary
            try:
                signals = generate_signals(st.session_state.data)
                if signals is not None and not isinstance(signals, bool):
                    latest_signal = signals.iloc[-1] if not signals.empty else None
                    if latest_signal is not None:
                        signal_col1, signal_col2 = st.columns(2)
                        with signal_col1:
                            if latest_signal.get('buy_signal', False):
                                st.success("🔼 BUY SIGNAL")
                            elif latest_signal.get('sell_signal', False):
                                st.error("🔽 SELL SIGNAL")
                            else:
                                st.info("➡️ NEUTRAL")
                                
                        with signal_col2:
                            # Display signal strength if available
                            if 'signal_strength' in latest_signal:
                                strength = int(latest_signal['signal_strength'])
                                labels = {1: "Weak", 2: "Moderate", 3: "Strong", 4: "Very Strong"}
                                st.info(f"Signal Strength: {labels.get(strength, 'Unknown')}")
            except Exception as e:
                st.warning(f"Could not generate signals: {str(e)}")
        
        # Tab 2: Signals - Detailed signal information
        with tabs[1]:
            st.header("Signal Analysis")
            
            # Display detailed signals information
            try:
                render_signals(st.session_state.data)
            except Exception as e:
                st.error(f"Error rendering signals: {str(e)}")
                st.info("Try adjusting your data timeframe or checking signal generation settings.")
        
        # Tab 3: Position Sizing
        with tabs[2]:
            # Risk management calculator
            render_risk_calculator(
                st.session_state.data.iloc[-1]['close'],
                risk_per_trade,
                stop_loss_atr,
                take_profit_ratio
            )
        
        # Tab 4: Backtest
        with tabs[3]:
            # Backtesting functionality - use a wrapper to provide a temporary UI
            st.header("Backtesting")
            st.info("Choose a strategy and timeframe to backtest your trading signals.")
            
            # Instead of using the removed function, create a basic placeholder
            strategy_options = ["Moving Average Crossover", "RSI Oscillator", "MACD Signal", "Multi-Indicator"]
            selected_strategy = st.selectbox("Select Strategy", strategy_options)
            
            period = st.selectbox("Backtest Period", ["1 Month", "3 Months", "6 Months", "1 Year"])
            
            if st.button("Run Backtest"):
                st.info("Backtesting functionality is under development. Coming soon!")
                
                # Use the available render_backtest function if it's implemented
                try:
                    render_backtest(st.session_state.data, selected_strategy, st.session_state.symbol)
                except Exception as e:
                    st.error(f"Backtest error: {str(e)}")
                    st.info("Detailed backtesting functionality will be available in a future update.")
            
            # Display a sample metrics card
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Win Rate", "0%")
            with col2:
                st.metric("Profit Factor", "0.0")
            with col3:
                st.metric("Max Drawdown", "0%")
            with col4:
                st.metric("Net Profit", "$0")
    else:
        # Landing page content when no data is loaded
        st.write("# Welcome to Day Trading Helper!")
        st.write("👈 Configure your settings in the sidebar and click 'Fetch Data' to get started.")
        
        # Create empty tabs for navigation consistency
        tabs = st.tabs(["Dashboard", "Signals", "Position Sizing", "Backtest"])

if __name__ == "__main__":
    try:
        import pytz  # Make sure pytz is available for timezone handling
    except ImportError:
        st.error("Missing required package: pytz. Install with 'pip install pytz'")
        st.stop()
        
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}") 