import streamlit as st
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
    render_chart,
    render_indicator_chart,
    render_signal_table,
    render_risk_analysis,
    render_patterns_section,
    render_indicator_combinations,
    render_advanced_concepts,
    render_backtest_ui,
    render_trends,
    render_backtest,
    render_risk,
    render_signals,
    render_signal_chart
)
from app.tradingview import (
    generate_tradingview_chart_url,
    generate_indicator_script,
    render_tradingview_widget
)
from app.signals import generate_signals

# Load environment variables for API keys
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Day Trading Helper",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for cached data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'signals' not in st.session_state:
    st.session_state.signals = None
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'AAPL'
if 'exchange' not in st.session_state:
    st.session_state.exchange = 'NASDAQ'

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
            # Alpaca API requires period number to be 1 (e.g., 1 Minute, 1 Hour, 1 Day)
            original_timeframe = timeframe
            
            # Parse the timeframe string to extract number and unit
            if timeframe.endswith('m'):
                if timeframe == '1m':
                    alpaca_timeframe = AlpacaTimeFrame.Minute
                elif timeframe == '5m':
                    # Will resample 1-minute data to 5m later
                    alpaca_timeframe = AlpacaTimeFrame.Minute
                elif timeframe == '15m':
                    # Will resample 1-minute data to 15m later
                    alpaca_timeframe = AlpacaTimeFrame.Minute
                elif timeframe == '30m':
                    # Will resample 1-minute data to 30m later
                    alpaca_timeframe = AlpacaTimeFrame.Minute
                else:
                    # Default to minute for unknown minute-based timeframes
                    alpaca_timeframe = AlpacaTimeFrame.Minute
            elif timeframe.endswith('h'):
                alpaca_timeframe = AlpacaTimeFrame.Hour
            elif timeframe == '1d':
                alpaca_timeframe = AlpacaTimeFrame.Day
            else:
                # Default to 1 Hour for unknown timeframes
                alpaca_timeframe = AlpacaTimeFrame.Hour
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
            if (end - start).days > 30 and alpaca_timeframe == AlpacaTimeFrame.Minute:
                # Use hourly data for periods > 30 days if minute data was requested
                alpaca_timeframe = AlpacaTimeFrame.Hour
                st.info(f"Switched to hourly data for {timeframe} due to long period ({period})")
            elif (end - start).days > 90 and alpaca_timeframe == AlpacaTimeFrame.Hour:
                # Use daily data for periods > 90 days
                alpaca_timeframe = AlpacaTimeFrame.Day
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
                    if original_timeframe == '5m' and alpaca_timeframe == AlpacaTimeFrame.Minute:
                        df = df.resample('5T').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    elif original_timeframe == '15m' and alpaca_timeframe == AlpacaTimeFrame.Minute:
                        df = df.resample('15T').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    elif original_timeframe == '30m' and alpaca_timeframe == AlpacaTimeFrame.Minute:
                        df = df.resample('30T').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    elif original_timeframe == '1h' and alpaca_timeframe == AlpacaTimeFrame.Minute:
                        df = df.resample('1H').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    elif original_timeframe == '2h' and alpaca_timeframe == AlpacaTimeFrame.Hour:
                        df = df.resample('2H').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    elif original_timeframe == '4h' and alpaca_timeframe == AlpacaTimeFrame.Hour:
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
                        return load_sample_data()
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

def load_sample_data():
    """Load sample data for demonstration"""
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
        
        # Request AAPL data as sample - use data that's 16 minutes old for free accounts
        eastern_tz = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern_tz)
        end = current_time - timedelta(minutes=16)
        start = end - timedelta(days=30)
        
        # Always use daily data for sample
        request_params = StockBarsRequest(
            symbol_or_symbols=["AAPL"],
            timeframe=AlpacaTimeFrame(1, AlpacaTimeFrame.Day),
            start=start,
            end=end,
            adjustment='all'
        )
        
        # Fetch bars data
        bars = data_client.get_stock_bars(request_params)
        
        # Convert to dataframe
        if bars and bars.data and "AAPL" in bars.data:
            df = bars.df.loc["AAPL"].copy()
            
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
            raise Exception("No sample data available from Alpaca")
            
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
    """Main function to run the Streamlit app"""
    
    # Check for Alpaca API keys at the start
    st.sidebar.title("Day Trading Helper")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = {}
    
    if 'symbol' not in st.session_state:
        st.session_state.symbol = 'AAPL'
        
    if 'timeframe' not in st.session_state:
        st.session_state.timeframe = '1h'
        
    if 'period' not in st.session_state:
        st.session_state.period = '1d'
        
    if 'data_source' not in st.session_state:
        st.session_state.data_source = 'Alpaca'
    
    # Sidebar for configuration
    with st.sidebar:
        data_source = st.radio(
            "Select Data Source",
            options=["Alpaca", "TradingView", "Sample Data"],
            index=0 if st.session_state.data_source == "Alpaca" else \
                  1 if st.session_state.data_source == "TradingView" else 2
        )
        st.session_state.data_source = data_source
        
        if data_source in ["Alpaca", "TradingView"]:
            # Show API key input fields
            if data_source == "Alpaca":
                alpaca_api_key = os.getenv("ALPACA_API_KEY", "")
                alpaca_secret_key = os.getenv("ALPACA_API_SECRET", "")
                
                if not alpaca_api_key or not alpaca_secret_key:
                    with st.form("alpaca_keys_form"):
                        new_api_key = st.text_input("Alpaca API Key", value=alpaca_api_key, key="alpaca_api_key")
                        new_secret_key = st.text_input("Alpaca API Secret", value=alpaca_secret_key, type="password", key="alpaca_secret_key")
                        
                        if st.form_submit_button("Save API Keys"):
                            # Save keys to .env file
                            with open(".env", "w") as f:
                                f.write(f"ALPACA_API_KEY={new_api_key}\n")
                                f.write(f"ALPACA_API_SECRET={new_secret_key}\n")
                                f.write(f"ALPACA_BASE_URL=https://paper-api.alpaca.markets\n")
                                
                            # Update environment variables
                            os.environ["ALPACA_API_KEY"] = new_api_key
                            os.environ["ALPACA_API_SECRET"] = new_secret_key
                            os.environ["ALPACA_BASE_URL"] = "https://paper-api.alpaca.markets"
                            
                            st.success("API keys saved successfully!")
                            
                            # Initialize Alpaca client
                            init_alpaca_client()
            
            # Symbol input
            symbol = st.text_input("Symbol", value=st.session_state.symbol).upper()
            st.session_state.symbol = symbol if symbol else 'AAPL'
            
            # Timeframe selector with multiple time frames
            timeframe_options = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
            timeframes = st.multiselect(
                "Timeframes",
                options=timeframe_options,
                default=[st.session_state.timeframe] if st.session_state.timeframe in timeframe_options else ["1h"]
            )
            
            if not timeframes:
                timeframes = ["1h"]  # Default if nothing selected
                
            # Primary timeframe selector
            st.session_state.timeframe = st.selectbox(
                "Primary Timeframe",
                options=timeframes if timeframes else ["1h"],
                index=timeframes.index(st.session_state.timeframe) if st.session_state.timeframe in timeframes else 0
            )
            
            # Period selector
            period = st.selectbox(
                "Period",
                options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                index=["1d", "5d", "1mo", "3mo", "6mo", "1y"].index(st.session_state.period) if st.session_state.period in ["1d", "5d", "1mo", "3mo", "6mo", "1y"] else 2
            )
            st.session_state.period = period
            
            # Data fetch button
            if st.button("Fetch Data"):
                with st.spinner("Fetching data..."):
                    st.session_state.data = {}  # Clear previous data
                    
                    # Fetch data for each timeframe
                    for tf in timeframes:
                        try:
                            data = fetch_data(st.session_state.symbol, st.session_state.exchange, tf, period)
                            if data is not None and not data.empty:
                                st.session_state.data[tf] = data
                        except Exception as e:
                            st.error(f"Error fetching {tf} data: {str(e)}")
                            
                    if not st.session_state.data:
                        st.error("Failed to fetch data for any timeframe.")
                    else:
                        st.success(f"Successfully fetched data for {', '.join(st.session_state.data.keys())} timeframes.")
        else:
            # Load sample data
            if st.button("Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    st.session_state.data = {}  # Clear previous data
                    
                    # Generate sample data for multiple timeframes
                    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
                    for tf in timeframes:
                        sample_data = load_sample_data()
                        if sample_data is not None and not sample_data.empty:
                            st.session_state.data[tf] = sample_data
                            
                    if not st.session_state.data:
                        st.error("Failed to load sample data.")
                    else:
                        st.session_state.symbol = "SAMPLE"
                        st.success(f"Successfully loaded sample data for {', '.join(st.session_state.data.keys())} timeframes.")
    
    # Main content area
    has_data = bool(st.session_state.data)
    
    if has_data:
        # Get primary data for main display
        primary_timeframe = st.session_state.timeframe
        if primary_timeframe not in st.session_state.data:
            primary_timeframe = list(st.session_state.data.keys())[0]
            
        data = st.session_state.data[primary_timeframe]
        symbol = st.session_state.symbol
        
        # Create tabs for different analysis views
        tabs = st.tabs([
            "Dashboard", "Technical Analysis", "Signals", 
            "Backtesting", "Risk Management", "Education"
        ])
        
        # Generate signals for all timeframes
        if 'signal_generator' not in st.session_state:
            from app.signals.generator import create_default_signal_generator
            st.session_state.signal_generator = create_default_signal_generator()
            
        # Process data with signal generator
        signals = st.session_state.signal_generator.generate_signals(st.session_state.data)
        
        # Dashboard tab
        with tabs[0]:
            render_dashboard(data, symbol)
            
            # Add signal chart below dashboard
            st.subheader("Price Chart with Signals")
            signal_chart = render_signal_chart(data, signals, symbol)
            if signal_chart:
                st.plotly_chart(signal_chart, use_container_width=True)
            
        # Technical analysis tab
        with tabs[1]:
            render_trends(data, symbol)
            
        # Signals tab
        with tabs[2]:
            # Create tabs for different signal views
            signal_tabs = st.tabs(["Standard Signals", "Advanced Signals"])
            
            with signal_tabs[0]:
                # Use primary timeframe data instead of the dictionary
                primary_data = st.session_state.data[primary_timeframe]
                render_signals(primary_data)
                
            with signal_tabs[1]:
                from app.components import render_advanced_signals
                render_advanced_signals(st.session_state.data, st.session_state.timeframe)
            
        # Backtesting tab
        with tabs[3]:
            render_backtest(data, symbol)
            
        # Risk management tab
        with tabs[4]:
            render_risk(data, symbol)
            
        # Education tab
        with tabs[5]:
            # Create tabs for different educational content
            education_tabs = st.tabs(["Indicator Combinations", "Advanced Concepts"])
            
            with education_tabs[0]:
                render_indicator_combinations()
                
            with education_tabs[1]:
                render_advanced_concepts()
    else:
        # Display initial empty state
        st.title("Day Trading Helper")
        st.markdown("""
        Welcome to the Day Trading Helper tool! 
        
        Please use the sidebar to configure your data source and fetch market data.
        
        **Features:**
        - Technical indicators and trend analysis
        - Multi-timeframe signal generation
        - Trade signal identification
        - Risk management analysis
        - Educational resources on trading strategies
        
        Get started by selecting a data source and fetching data for your desired symbol.
        """)
        
        # Display educational content in empty state
        st.header("Educational Resources")
        
        # Create tabs for different educational content
        education_tabs = st.tabs(["Indicator Combinations", "Advanced Concepts"])
        
        with education_tabs[0]:
            render_indicator_combinations()
            
        with education_tabs[1]:
            render_advanced_concepts()

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