import ibkr_runtime_patch  # Applied by fix_main_app.py to fix IBKR API parameter incompatibility

import os
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time
import pytz
import warnings
import traceback
from pathlib import Path
import asyncio
import nest_asyncio
import logging
from threading import Thread

# Create and set event loop before importing any ib_insync modules
nest_asyncio.apply()
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Now import ib_insync components
from ib_insync import util
util.patchAsyncio()

# Import helper modules directly from their respective files
from signals.signal_functions import generate_standard_signals as generate_signals
from signals.signal_functions import is_market_hours
from signals.generator import analyze_single_day, generate_signals_multi_timeframe
from signals.paired_signals import PairedSignalGenerator

# Import signal processing functions
try:
    from signals.generator import generate_signals_advanced
except ImportError:
    # Fallback if there's an import error
    def generate_signals_advanced(data, **kwargs):
        warnings.warn("generate_signals_advanced not available, using basic signals")
        return generate_signals(data)

# Import UI components
from components.signals import (
    render_signal_table,
    render_signals,
    render_orb_signals,
    render_advanced_signals,
    render_exit_strategy_analysis
)

# Import data functions
from data.loader import (
    load_sample_data
)

# Import IBKR client and real-time streaming components
try:
    # When imported as a module
    from app.ibkr_client import IBKRClient
    from app.streamlit_ibkr import render_ibkr_data_page, init_ibkr_client
    from app.asyncio_patch import ensure_event_loop, with_event_loop, run_periodically, background_loop
    IBKR_AVAILABLE = True
    print("IBKR components successfully imported, IBKR_AVAILABLE =", IBKR_AVAILABLE)
except ImportError:
    try:
        # When run directly
        from ibkr_client import IBKRClient  
        from streamlit_ibkr import render_ibkr_data_page, init_ibkr_client
        from asyncio_patch import ensure_event_loop, with_event_loop, run_periodically, background_loop
        IBKR_AVAILABLE = True
        print("IBKR components successfully imported (direct), IBKR_AVAILABLE =", IBKR_AVAILABLE)
    except ImportError as e:
        print(f"IBKR components not available: {e}")
        IBKR_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize background task flag
if 'background_task_running' not in st.session_state:
    st.session_state.background_task_running = False

# Set page config
st.set_page_config(
    page_title="Day Trading Helper",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables for API keys
if 'data' not in st.session_state:
    st.session_state.data = None
if 'signals' not in st.session_state:
    st.session_state.signals = None
if 'symbol' not in st.session_state:
    st.session_state.symbol = 'SPY'
if 'exchange' not in st.session_state:
    st.session_state.exchange = 'NASDAQ'
if 'api_keys_configured' not in st.session_state:
        st.session_state.api_keys_configured = True
if 'active_page' not in st.session_state:
    st.session_state.active_page = "main"

def init_data_client():
    """Initialize data client - in this case, we use Yahoo Finance which requires no authentication"""
    try:
        # Test the connection by getting a small amount of data
        test_data = yf.download("SPY", period="1d", interval="1h")
        if test_data is None or test_data.empty:
            st.sidebar.warning("Unable to fetch test data from Yahoo Finance. Check your internet connection.")
            return False
        return True
    except Exception as e:
        error_msg = str(e)
        st.sidebar.error(f"⚠️ Error initializing Yahoo Finance connection: {error_msg}")
        return False

def fetch_data(symbol, exchange=None, timeframe="1d", period="3mo"):
    """
    Fetch market data from Yahoo Finance
    
    Args:
        symbol (str): Trading symbol
        exchange (str): Not used, included for backward compatibility
        timeframe (str): Data timeframe (1m, 5m, 15m, 30m, 1h, 1d)
        period (str): Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, max)
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    try:
        # Map timeframes to Yahoo Finance intervals
        yahoo_interval = timeframe
        if timeframe == '1d':
            yahoo_interval = '1d'
        elif timeframe == '1h':
            yahoo_interval = '60m'
        elif timeframe == '4h':
            # Yahoo doesn't have 4h directly, use 1h and we'll resample later
            yahoo_interval = '60m'
        elif timeframe == '30m':
            yahoo_interval = '30m'
        elif timeframe == '15m':
            yahoo_interval = '15m'
        elif timeframe == '5m':
            yahoo_interval = '5m'
        elif timeframe == '1m':
            yahoo_interval = '1m'
        else:
            # Default to 1h for unknown timeframes
            yahoo_interval = '60m'
            st.warning(f"Unknown timeframe format: {timeframe}. Using 1 Hour as default.")
            
        # Map period strings to Yahoo Finance periods
        yahoo_period = period
        if period == '1wk':
            yahoo_period = '5d'  # Use 5d for 1 week
            
        # For 1m data, Yahoo only allows 7 days max
        if yahoo_interval == '1m' and period not in ['1d', '5d', '7d']:
            st.warning(f"Yahoo Finance only provides 1-minute data for up to 7 days. Using 7d period instead of {period}.")
            yahoo_period = '7d'
            
        # Fetch data from Yahoo Finance
        with st.spinner(f"Fetching {symbol} data from Yahoo Finance..."):
            stock = yf.Ticker(symbol)
            df = stock.history(period=yahoo_period, interval=yahoo_interval)
            
        # Check if we got data
        if df.empty:
            st.error(f"No data returned from Yahoo Finance for {symbol}")
            return None
            
        # Rename columns to match our expected format
        df.columns = [c.lower() for c in df.columns]
        
        # Resample if we need to convert 1h to 4h
        if timeframe == '4h' and yahoo_interval == '60m':
            df = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
        # Add a timestamp column (mainly for display)
        # Convert timestamps to Eastern Time for display
        if isinstance(df.index[0], pd.Timestamp):
            eastern = pytz.timezone('US/Eastern')
            if df.index[0].tzinfo is not None:
                # Convert existing timezone to Eastern
                df.index = df.index.tz_convert(eastern)
            else:
                # Add timezone info if missing
                df.index = df.index.tz_localize('UTC').tz_convert(eastern)
                
        return df
    except Exception as e:
        st.error(f"Error fetching {symbol} data: {str(e)}")
        traceback.print_exc()
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
        # Create synthetic data with realistic patterns
        eastern_tz = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern_tz)
        
        # Determine number of data points based on period and timeframe
        if period == '1d':
            periods = 1 * 24  # 1 day
        elif period == '5d':
            periods = 5 * 24  # 5 days
        elif period == '1mo':
            periods = 30 * 24  # 1 month
        elif period == '3mo':
            periods = 90 * 24  # 3 months
        elif period == '6mo':
            periods = 180 * 24  # 6 months
        elif period == '1y':
            periods = 365 * 24  # 1 year
        else:
            periods = 90 * 24  # Default to 3 months
        
        # Adjust frequency based on timeframe
        if timeframe == '1m':
            freq = '1min'
            periods = min(periods * 60, 1000)  # Cap at 1000 points
        elif timeframe == '5m':
            freq = '5min'
            periods = min(periods * 12, 1000)
        elif timeframe == '15m':
            freq = '15min'
            periods = min(periods * 4, 1000)
        elif timeframe == '30m':
            freq = '30min'
            periods = min(periods * 2, 1000)
        elif timeframe == '1h':
            freq = '1H'
        elif timeframe == '4h':
            freq = '4H'
            periods = periods // 4
        elif timeframe == '1d':
            freq = 'B'  # Business day frequency
            periods = periods // 24
        else:
            freq = '1H'  # Default to hourly
            
        # Create dates - business days only, during market hours
        end_date = current_time.replace(hour=16, minute=0)
        
        if freq in ['1min', '5min', '15min', '30min', '1H', '4H']:
            # For intraday data, create timestamps during market hours (9:30 AM - 4:00 PM ET)
            market_days = pd.date_range(
                end=end_date, 
                periods=min(periods // 8 + 1, 90),  # At most 90 days
                freq=pd.tseries.offsets.BDay(),  # Business day frequency
            )
            
            # For each day, create timestamps during market hours
            market_hours_timestamps = []
            for day in market_days:
                # Create timestamps for this day based on frequency
                day_start = day.replace(hour=9, minute=30)
                day_end = day.replace(hour=16, minute=0)
                
                if freq == '1min':
                    day_timestamps = pd.date_range(start=day_start, end=day_end, freq='1min')
                elif freq == '5min':
                    day_timestamps = pd.date_range(start=day_start, end=day_end, freq='5min')
                elif freq == '15min':
                    day_timestamps = pd.date_range(start=day_start, end=day_end, freq='15min')
                elif freq == '30min':
                    day_timestamps = pd.date_range(start=day_start, end=day_end, freq='30min')
                elif freq == '1H':
                    day_timestamps = pd.date_range(start=day_start, end=day_end, freq='1H')
                elif freq == '4H':
                    # Only 2 4H candles per day (9:30-13:30, 13:30-16:00)
                    day_timestamps = [day_start, day_start + pd.Timedelta(hours=4)]
                
                market_hours_timestamps.extend(day_timestamps)
            
            # Sort timestamps and take the most recent ones
            market_hours_timestamps.sort()
            dates = market_hours_timestamps[-periods:]  # Take last N timestamps
        else:
            # For daily data, use business days
            dates = pd.date_range(
                end=end_date, 
                periods=periods,
                freq='B',  # Business day frequency
            )
        
        # Generate more realistic price data based on the symbol
        base_price = 150  # Default base price
        
        # Use realistic price range based on symbol
        if symbol.upper() == "AAPL":
            base_price = 190
        elif symbol.upper() == "MSFT":
            base_price = 420
        elif symbol.upper() == "GOOGL":
            base_price = 170
        elif symbol.upper() == "AMZN":
            base_price = 180
        elif symbol.upper() == "META":
            base_price = 500
        elif symbol.upper() == "TSLA":
            base_price = 220
        elif symbol.upper() == "NVDA":
            base_price = 940
        elif symbol.upper() == "SPY":
            base_price = 550
            
        # Generate data with realistic price movement
        data = pd.DataFrame(index=dates)
        
        # Create realistic price movement with random walk and slight upward bias
        price = base_price
        closes = []
        for i in range(len(dates)):
            # Random walk with slight upward bias
            price_change = np.random.normal(0.0001, 0.002)  # Slight upward bias
            price *= (1 + price_change)
            closes.append(price)
        
        data['close'] = closes
        
        # Generate open, high, low based on close
        # For the first point, use same value
        data['open'] = [closes[0]] + closes[:-1]
        
        # Generate highs and lows with reasonable ranges
        highs = []
        lows = []
        
        for i in range(len(closes)):
            open_price = data['open'].iloc[i]
            close_price = data['close'].iloc[i]
            
            # Determine candle direction
            if close_price > open_price:
                # Bullish candle
                high = close_price * (1 + abs(np.random.normal(0, 0.001)))
                low = open_price * (1 - abs(np.random.normal(0, 0.001)))
            else:
                # Bearish candle
                high = open_price * (1 + abs(np.random.normal(0, 0.001)))
                low = close_price * (1 - abs(np.random.normal(0, 0.001)))
                
            highs.append(high)
            lows.append(low)
        
        data['high'] = highs
        data['low'] = lows
        
        # Generate realistic volume
        base_volume = 1000000  # Default base volume
        
        # Adjust volume based on symbol
        if symbol.upper() in ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA"]:
            base_volume = 5000000
        elif symbol.upper() == "SPY":
            base_volume = 50000000
            
        # Generate volume with daily pattern (higher at open and close)
        volumes = []
        for date in dates:
            # Higher volume at market open and close
            hour = date.hour
            minute = date.minute
            
            # Base multiplier
            vol_mult = 1.0
            
            # Adjust volume based on time of day
            if hour == 9 and minute == 30:
                vol_mult = 1.5  # Higher volume at market open
            elif hour >= 15:
                vol_mult = 1.3  # Higher volume toward market close
                
            # Add some randomness
            vol_mult *= np.random.normal(1, 0.2)
            
            # Floor at 0.5
            vol_mult = max(0.5, vol_mult)
            
            volume = int(base_volume * vol_mult)
            volumes.append(volume)
            
        data['volume'] = volumes
        
        # Add timezone information to index
        data.index = data.index.tz_localize(eastern_tz)
        
        st.warning(f"Using synthetic sample data for {symbol}. This data is generated and does not reflect actual market prices.")
        return data
            
    except Exception as e:
        st.error(f"Error generating sample data: {str(e)}")
        traceback.print_exc()
        
        # Create very basic fallback if everything else fails
        dates = pd.date_range(
            end=datetime.now(), 
            periods=100,
            freq='H'
        )
        
        data = pd.DataFrame(index=dates)
        data['close'] = np.linspace(100, 110, len(dates)) + np.random.normal(0, 1, len(dates))
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        data['high'] = data[['open', 'close']].max(axis=1) + abs(np.random.normal(0, 0.5, len(dates)))
        data['low'] = data[['open', 'close']].min(axis=1) - abs(np.random.normal(0, 0.5, len(dates)))
        data['volume'] = np.random.randint(100000, 1000000, size=len(data))
        
        return data

def fetch_yahoo_data(symbol, period='1d', interval='1m'):
    """
    Fetch data from Yahoo Finance
    
    Args:
        symbol (str): Trading symbol
        period (str): Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    try:
        # For 1m data, Yahoo only allows 7 days max
        if interval == '1m' and period not in ['1d', '5d', '7d']:
            st.warning(f"Yahoo Finance only provides 1-minute data for up to 7 days. Using 1d period instead of {period}.")
            period = '1d'
            
        # Add a day to ensure we get the most recent data
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            st.error(f"No data returned from Yahoo Finance for {symbol}")
            return None
            
        # Rename columns to match our expected format if needed
        df.columns = [c.lower() for c in df.columns]
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {str(e)}")
        return None

@run_periodically("ibkr_data_processor", interval=1.0)
def process_ibkr_data():
    """
    Process IBKR data in the background without using Streamlit context
    
    This runs in a background thread and just handles the IB event loop
    without trying to access Streamlit's session_state
    """
    try:
        # Check if we have an IB instance in our global variable
        global ibkr_client
        
        if ibkr_client is not None:
            # Run one iteration of the event loop
            ibkr_client.run_one_iteration()
            
            # Check if we should reconnect
            if not ibkr_client.connected:
                logger.info("IB disconnected, attempting to reconnect")
                try:
                    ibkr_client.connect()
                except Exception as e:
                    logger.error(f"Failed to reconnect to IB: {str(e)}")
    except Exception as e:
        logger.error(f"Error in process_ibkr_data: {str(e)}")

# Create a global var to avoid session_state access
ibkr_client = None

def start_background_tasks():
    """Start background tasks to handle IB data streaming"""
    global ibkr_client
    
    if 'background_task_running' not in st.session_state:
        st.session_state.background_task_running = False
        
    if not st.session_state.background_task_running:
        # Initialize client if needed
        init_ibkr_client()
        
        # Set the global client for use in background loops
        if 'ibkr' in st.session_state:
            ibkr_client = st.session_state.ibkr
        
        # The background loop is already running from asyncio_patch initialization
        st.session_state.background_task_running = True
        logger.info("Background tasks started")
    else:
        # Update the global client if needed
        if 'ibkr' in st.session_state:
            ibkr_client = st.session_state.ibkr

def render_debug_panel():
    """Render a debug panel to monitor the application state"""
    if st.sidebar.checkbox("Debug Mode", value=st.session_state.get("debug_mode", False)):
        st.session_state.debug_mode = True
        
        # Create a container for debug info
        debug_container = st.sidebar.container()
        
        with debug_container:
            st.write("### Debug Information")
            
            # Connection status
            if 'ibkr' in st.session_state:
                st.write("IB Connection Status:", st.session_state.ibkr.connected)
            
            # Market data types
            if 'market_data_types' in st.session_state:
                st.write("Market Data Types:", st.session_state.market_data_types)
            
            # Active streams
            if 'ibkr' in st.session_state and hasattr(st.session_state.ibkr, 'data_streams'):
                st.write("Active Streams:", list(st.session_state.ibkr.data_streams.keys()))
            
            # Session state data
            data_keys = [k for k in st.session_state.keys() if k.endswith('_data')]
            if data_keys:
                with st.expander("Raw Data in Session State"):
                    for k in data_keys:
                        st.write(f"{k}:", st.session_state[k])
            
            # Time offsets
            if 'ibkr' in st.session_state and hasattr(st.session_state.ibkr, 'time_offset'):
                st.write("Time Offset:", f"{st.session_state.ibkr.time_offset:.2f} seconds")
            
            # Last update time
            if 'last_ui_update' in st.session_state:
                st.write("Last UI Update:", time.time() - st.session_state.last_ui_update, "seconds ago")
    else:
        st.session_state.debug_mode = False

def setup_event_loop():
    """Set up the event loop properly for Streamlit."""
    try:
        logger.info("Setting up event loop for Streamlit")
        # Apply nest_asyncio to make asyncio work in Streamlit
        nest_asyncio.apply()
        
        # Create a new event loop if there isn't one already
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                logger.warning("Event loop was closed. Creating a new one.")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            logger.info("No event loop in current thread. Creating a new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Patch asyncio to work better with Streamlit
        util.patchAsyncio()
        
        logger.info(f"Event loop setup complete: {loop}")
        return loop
    except Exception as e:
        logger.error(f"Error setting up event loop: {e}")
        import traceback
        traceback.print_exc()
        raise

def ensure_ibkr_available():
    """Ensure IBKR components are available and update the global variable"""
    global IBKR_AVAILABLE
    try:
        # Try importing the components
        from ibkr_client import IBKRClient
        from streamlit_ibkr import init_ibkr_client, render_ibkr_data_page, add_ibkr_to_main_app
        IBKR_AVAILABLE = True
        logger.info("IBKR components available: %s", IBKR_AVAILABLE)
        return True
    except ImportError as e:
        IBKR_AVAILABLE = False
        logger.error("IBKR components not available: %s", str(e))
        return False

def main():
    """Main function to run the Day Trading Helper application"""
    
    # Ensure we have a valid event loop
    loop = asyncio.get_event_loop()
    
    # Make sure IBKR availability is checked
    ensure_ibkr_available()
    
    # Print IBKR status for debugging
    print("In main function, IBKR_AVAILABLE =", IBKR_AVAILABLE)
    
    # Start background tasks for IB data handling
    try:
        if IBKR_AVAILABLE:
            start_background_tasks()
    except Exception as e:
        logger.error(f"Error starting background tasks: {e}")
        import traceback
        traceback.print_exc()
    
    # Check if we should display the real-time dashboard
    if st.session_state.active_page == "real_time":
        if IBKR_AVAILABLE:
            render_ibkr_data_page()
            return
        else:
            st.error("Real-time trading is not available. Please install ib_insync and nest_asyncio packages.")
            st.session_state.active_page = "main"
        return
    
    # Create a sidebar for configuration options
    with st.sidebar:
        st.title("Day Trading Helper")
        st.subheader("Configuration")
        
        # Display data source indicator in sidebar
        if 'data' in st.session_state and st.session_state.data is not None:
            if hasattr(st.session_state, 'using_sample_data') and st.session_state.using_sample_data:
                st.sidebar.error("⚠️ USING SAMPLE DATA", icon="⚠️")
            elif hasattr(st.session_state, 'data_source_type') and "Sample" in st.session_state.data_source_type:
                st.sidebar.error("⚠️ USING SAMPLE DATA", icon="⚠️")
            else:
                st.sidebar.success("✅ USING REAL DATA", icon="✅")
        
        # Create tabs for different configuration sections
        config_tabs = st.tabs(["Data Source", "Analysis", "Tools"])
        
        with config_tabs[0]:
            # Data source selection
            # Check IBKR availability again
            ensure_ibkr_available()
            
            data_source_options = ["Yahoo Finance", "Sample Data"]
            
            # Add IBKR as a data source if available
            if IBKR_AVAILABLE:
                data_source_options.insert(0, "Interactive Brokers")
                st.success("Interactive Brokers integration is available")
            else:
                st.warning("Interactive Brokers integration is not available. Check logs for details.")
                
            data_source = st.selectbox(
                "Select Data Source",
                data_source_options,
                index=0
            )
            
            # Show specific instructions based on data source
            if data_source == "Interactive Brokers":
                st.info("📊 **Interactive Brokers:**\n"
                       "- Real-time market data access\n"
                       "- Requires TWS or Gateway running\n"
                       "- Connect via API on port 7497 (TWS) or 4002 (Gateway)", icon="ℹ️")
            elif data_source == "Yahoo Finance":
                st.info("📊 **Yahoo Finance:**\n"
                       "- For 1-minute data, only last 7 days available\n"
                       "- No authentication required\n"
                       "- 15+ minute delay for real-time data", icon="ℹ️")
            
            # Symbol selection with SPY as default
            symbol = st.text_input("Symbol", "SPY").upper()
            
            # Timeframe selection with 1m as default
            timeframe = st.selectbox(
                "Select Timeframe",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=0
            )
            
            # Period selection with 1d as default
            period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
            period = st.selectbox("Select Period", period_options, index=0)
            
            # Add single day trading mode option
            st.divider()
            single_day_mode = st.checkbox("Single Day Trading Mode", value=False, 
                                        help="Focus analysis on a single trading day with intraday patterns")
            
            if single_day_mode:
                st.info("Single day mode focuses on intraday trading strategies")
                
            # Add SPY day trading mode option
            spy_day_trading_mode = st.checkbox("SPY Day Trading Mode", value=False,
                                           help="Use optimized parameters for SPY day trading")
            
            if spy_day_trading_mode:
                st.info("Using SPY-specific configuration for optimal day trading")
                
            # Fetch data button
            fetch_data_button = st.button("Fetch Data", use_container_width=True)
            
            # Save settings to session state
            st.session_state['single_day_mode'] = single_day_mode
            st.session_state['spy_day_trading_mode'] = spy_day_trading_mode
            
            if fetch_data_button:
                with st.spinner(f"Fetching {symbol} data..."):
                    try:
                        # If using SPY mode with SPY symbol, use the SPY config
                        if spy_day_trading_mode and symbol.upper() == "SPY":
                            try:
                                from config import SPY_CONFIG
                                from signals.spy_strategy import analyze_spy_day_trading
                                st.session_state['config'] = SPY_CONFIG
                                st.session_state['use_spy_strategy'] = True
                                st.success(f"Using optimized SPY configuration and strategy")
                            except ImportError:
                                st.warning("SPY configuration not found. Using default settings.")
                                st.session_state['use_spy_strategy'] = False
                        else:
                            st.session_state['use_spy_strategy'] = False
                        
                        # Reset sample data flag
                        st.session_state.using_sample_data = False
                        
                        # Load data based on selected source
                        if data_source == "Interactive Brokers" and IBKR_AVAILABLE:
                            # Set active page to real-time dashboard
                            st.session_state.active_page = "real_time"
                            st.session_state.symbol = symbol
                            st.experimental_rerun()
                        elif data_source == "Sample Data":
                            st.session_state.data = load_sample_data(symbol, timeframe, period)
                            st.session_state.using_sample_data = True
                            st.session_state.data_source_type = "Sample Data"
                            st.success(f"Successfully loaded sample data for {symbol}")
                        elif data_source == "Yahoo Finance":
                            # Get Yahoo Finance data
                            yahoo_data = fetch_data(symbol, None, timeframe, period)
                            if yahoo_data is not None and not yahoo_data.empty:
                                st.session_state.data = yahoo_data
                                st.session_state.data_source_type = "Yahoo Finance"
                                st.success(f"Successfully loaded data from Yahoo Finance")
                            else:
                                st.warning("Could not fetch data from Yahoo Finance. Falling back to sample data.")
                                st.session_state.data = load_sample_data(symbol, timeframe, period)
                                st.session_state.using_sample_data = True
                                st.session_state.data_source_type = "Sample Data (Fallback)"
                        
                        # Store symbol, timeframe and period in session state for reference
                        st.session_state.symbol = symbol
                        st.session_state.timeframe = timeframe
                        st.session_state.period = period
                        
                        # Clear any previous signals
                        if 'signals' in st.session_state:
                            del st.session_state.signals
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
                        traceback.print_exc()
        
        with config_tabs[1]:
            # Signal analysis options
            st.subheader("Signal Options")
            
            # Add market hours filter option
            market_hours_filter = st.checkbox("Filter for Market Hours", value=True,
                                            help="Only analyze data during market hours (9:30 AM - 4:00 PM ET)")
            
            # Save market hours filter setting to session state
            st.session_state.market_hours_filter = market_hours_filter
            
            # Add strategy options
            st.subheader("Strategy Options")
            
            # Display different options based on whether SPY strategy is enabled
            if hasattr(st.session_state, 'use_spy_strategy') and st.session_state.use_spy_strategy:
                st.info("Using SPY-specific strategy settings")
                
                # Let user select emphasis on specific signals
                signal_emphasis = st.select_slider(
                    "Signal Emphasis",
                    options=["Conservative", "Balanced", "Aggressive"],
                    value="Balanced"
                )
                
                st.session_state.signal_emphasis = signal_emphasis
            else:
                # Standard strategy options
                st.info("Using standard technical analysis strategy")
                
                # Let user select which indicator types to prioritize
                col1, col2 = st.columns(2)
                
                with col1:
                    trend_weight = st.slider("Trend Indicators", 0.0, 2.0, 1.0, 0.1,
                                          help="Weight for trend-following indicators like moving averages")
                    
                    momentum_weight = st.slider("Momentum Indicators", 0.0, 2.0, 1.0, 0.1,
                                             help="Weight for momentum indicators like RSI and MACD")
                
                with col2:
                    volatility_weight = st.slider("Volatility Indicators", 0.0, 2.0, 1.0, 0.1,
                                               help="Weight for volatility indicators like Bollinger Bands")
                    
                    volume_weight = st.slider("Volume Indicators", 0.0, 2.0, 1.0, 0.1,
                                           help="Weight for volume-based indicators")
                
                # Save weights to session state
                st.session_state.weights = {
                    'trend': trend_weight,
                    'momentum': momentum_weight,
                    'volatility': volatility_weight,
                    'volume': volume_weight
                }
        
        with config_tabs[2]:
            # Tools tab
            st.subheader("Utility Tools")
            
            # Backtesting option
            enable_backtest = st.checkbox("Enable Backtesting", value=False,
                                        help="Simulate trading using generated signals")
            
            # Risk management settings
            st.subheader("Risk Management")
            
            # Account size
            account_size = st.number_input("Account Size ($)", value=10000.0, step=1000.0, min_value=1000.0)
            
            # Risk per trade
            risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1,
                                     help="Maximum percentage of account to risk on a single trade")
            
            # Save risk settings
            st.session_state.account_size = account_size
            st.session_state.risk_per_trade = risk_per_trade
    
    # Main content area
    if st.session_state.active_page == "real_time" and IBKR_AVAILABLE:
        render_ibkr_data_page()
    elif 'data' not in st.session_state or st.session_state.data is None:
        # Show welcome screen if no data is loaded
        st.title("Welcome to Day Trading Helper")
        st.write("Please select a data source and fetch some data to begin.")
        
        # Show a hero image or illustration
        st.image("https://static.vecteezy.com/system/resources/previews/001/879/536/original/stock-market-forex-trading-graph-free-vector.jpg", use_column_width=True)
        
        # Display some sample capabilities
        st.markdown("""
        ## Features
        
        * 📊 **Technical Analysis** - Generate trading signals using multiple indicators
        * 📈 **Real-time Data** - Fetch market data from Yahoo Finance or Interactive Brokers
        * 🔮 **Signal Generation** - Get buy/sell signals with confidence levels
        * 📉 **Risk Management** - Calculate position sizing and risk metrics
        * 📱 **Mobile Friendly** - Use on any device with a web browser
        
        ## Getting Started
        
        1. Select a data source in the sidebar
        2. Enter a symbol (e.g., AAPL, MSFT, SPY)
        3. Choose timeframe and period
        4. Click "Fetch Data" to begin
        """)
        
        # Show a disclaimer
        with st.expander("Disclaimer"):
            st.warning("""
            The information provided by this application is for informational and educational purposes only. 
            It is not intended to be and does not constitute financial advice or any other advice. 
            You should not make any decision, financial, investment, trading or otherwise, based on any 
            of the information presented without undertaking independent due diligence and consultation 
            with a professional financial advisor.
            """)
    else:
        # Display the selected symbol and timeframe in the title
        st.title(f"{st.session_state.symbol} Analysis ({st.session_state.timeframe})")
        
        # Create tabs for different analysis views
        tabs = st.tabs(["Chart View", "Signals Analysis", "ORB Analysis", "Exit Strategy"])
        
        # Tab 1: Chart View
        with tabs[0]:
            st.header("Price Chart")
            
            # Add data source note
            if hasattr(st.session_state, 'data_source_type'):
                if "Sample" in st.session_state.data_source_type:
                    st.error("⚠️ **USING SAMPLE DATA** ⚠️  \nNote that the analysis below is based on sample data, not real market data.", icon="⚠️")
                elif hasattr(st.session_state, 'data_source_type') and "Yahoo" in st.session_state.data_source_type:
                    st.success("✅ **USING YAHOO FINANCE DATA** ✅  \nTrading signals and prices are based on Yahoo Finance market data.", icon="✅")
            
            # Display basic statistics of the data
            with st.expander("Data Statistics", expanded=False):
                try:
                    # Calculate basic statistics
                    data = st.session_state.data
                    stats = pd.DataFrame({
                        'Open': [data['open'].min(), data['open'].max(), data['open'].mean(), data['open'].std()],
                        'High': [data['high'].min(), data['high'].max(), data['high'].mean(), data['high'].std()],
                        'Low': [data['low'].min(), data['low'].max(), data['low'].mean(), data['low'].std()],
                        'Close': [data['close'].min(), data['close'].max(), data['close'].mean(), data['close'].std()],
                        'Volume': [data['volume'].min(), data['volume'].max(), data['volume'].mean(), data['volume'].std()]
                    }, index=['Min', 'Max', 'Mean', 'Std'])
                    
                    st.dataframe(stats, use_container_width=True)
                    
                    # Calculate price change statistics
                    if len(data) >= 2:
                        first_price = data['close'].iloc[0]
                        last_price = data['close'].iloc[-1]
                        price_change = last_price - first_price
                        pct_change = (price_change / first_price) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Price Change", f"${price_change:.2f}", f"{pct_change:.2f}%")
                        with col2:
                            # Calculate daily volatility
                            daily_returns = data['close'].pct_change()
                            volatility = daily_returns.std() * 100
                            st.metric("Volatility", f"{volatility:.2f}%")
                        with col3:
                            # Calculate volume change
                            first_volume = data['volume'].iloc[0]
                            last_volume = data['volume'].iloc[-1]
                            volume_change = ((last_volume / first_volume) - 1) * 100 if first_volume > 0 else 0
                            st.metric("Volume Change", f"{volume_change:.2f}%")
                except Exception as e:
                    st.error(f"Error calculating statistics: {str(e)}")
            
            # Display candlestick chart
            try:
                # Create figure with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=st.session_state.data.index,
                        open=st.session_state.data['open'],
                        high=st.session_state.data['high'],
                        low=st.session_state.data['low'],
                        close=st.session_state.data['close'],
                        name="Price"
                    )
                )
                
                # Add volume as bar chart on secondary axis
                fig.add_trace(
                    go.Bar(
                        x=st.session_state.data.index,
                        y=st.session_state.data['volume'],
                        name="Volume",
                        marker=dict(color='rgba(100, 100, 255, 0.3)'),
                        opacity=0.3
                    ),
                    secondary_y=True
                )
                
                # Add moving averages
                ma_periods = [9, 20, 50, 200]
                ma_colors = ['blue', 'green', 'red', 'purple']
                
                for i, period in enumerate(ma_periods):
                    if len(st.session_state.data) >= period:
                        ma = st.session_state.data['close'].rolling(window=period).mean()
                        fig.add_trace(
                            go.Scatter(
                                x=st.session_state.data.index,
                                y=ma,
                                name=f"{period}-MA",
                                line=dict(color=ma_colors[i], width=1)
                            )
                        )
                
                # Update layout
                fig.update_layout(
                    title=f"{st.session_state.symbol} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=600,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Update y-axes titles
                fig.update_yaxes(title_text="Price", secondary_y=False)
                fig.update_yaxes(title_text="Volume", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering chart: {str(e)}")
                traceback.print_exc()

        # Tab 2: Signals Analysis
        with tabs[1]:
            st.header("DEBUG: Checking signal analysis tab")
            try:
                # Generate signals if not already done
                if 'signals' not in st.session_state or st.session_state.signals is None:
                    with st.spinner("Generating signals..."):
                        st.write("Generating signals...")
                        from signals.generator import generate_signals_with_pairs
                        
                        # Generate signals with paired information
                        paired_results = generate_signals_with_pairs(st.session_state.data)
                        
                        # Store signals and paired results in session state
                        st.session_state.signals = paired_results['signal_df']
                        st.session_state.paired_results = paired_results
                        
                        st.write(f"Generated {len(st.session_state.signals)} signal entries with paired signals")
                        
                        # Display info about paired signals if available
                        if 'paired_signals' in paired_results and not paired_results['paired_signals'].empty:
                            metrics = paired_results.get('metrics', {})
                            st.success(f"Found {len(paired_results['paired_signals'])} valid trade pairs with " +
                                     f"{metrics.get('win_rate', 0)*100:.1f}% win rate and " +
                                     f"{metrics.get('avg_profit', 0)*100:.2f}% average profit")
                
                # Display a complete table of all signals first
                st.subheader("All Signals Table")
                
                # Create a dataframe with all buy and sell signals
                all_signals_data = []
                for idx, row in st.session_state.signals.iterrows():
                    # Check for both raw and filtered signals
                    has_raw_signal = row.get('buy_signal', False) or row.get('sell_signal', False)
                    has_filtered_signal = row.get('filtered_buy_signal', False) or row.get('filtered_sell_signal', False)
                    
                    if has_raw_signal or has_filtered_signal:
                        # Determine signal types
                        raw_signal_type = "Buy" if row.get('buy_signal', False) else "Sell" if row.get('sell_signal', False) else "None"
                        filtered_signal_type = "Buy" if row.get('filtered_buy_signal', False) else "Sell" if row.get('filtered_sell_signal', False) else "None"
                        
                        # Only filtered signals have a price value
                        price = st.session_state.data.loc[idx, 'close'] if idx in st.session_state.data.index else row.get('signal_price', 0)
                        
                        # Format timestamp
                        if hasattr(idx, 'tzinfo'):
                            try:
                                # Convert to Eastern time if it's timezone-aware
                                eastern = pytz.timezone('US/Eastern')
                                date = idx.astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S ET')
                            except:
                                date = str(idx)
                        else:
                            date = str(idx)
                        
                        # Get details about the signal
                        details = []
                        for col in row.index:
                            if col.endswith('_bullish') or col.endswith('_bearish') or col.startswith('price_') or 'cross' in col:
                                if row[col]:
                                    details.append(col.replace('_', ' '))
                        
                        # Extract strength from score
                        raw_score = row.get('buy_score', 0) if raw_signal_type == "Buy" else row.get('sell_score', 0) if raw_signal_type == "Sell" else 0
                        
                        # Get ATR value if available
                        atr_value = st.session_state.data.loc[idx, 'atr'] if idx in st.session_state.data.index and 'atr' in st.session_state.data.columns else None
                        
                        all_signals_data.append({
                            "Date": date,
                            "Raw Signal": raw_signal_type,
                            "Filtered Signal": filtered_signal_type,
                            "Price": f"${price:.2f}" if isinstance(price, (int, float)) else "N/A",
                            "Score": f"{raw_score:.2f}" if raw_score else "N/A",
                            "ATR": f"{atr_value:.4f}" if atr_value is not None else "N/A",
                            "Details": ", ".join(details[:3]) # Limit to first 3 for clarity
                        })
                
                if all_signals_data:
                    # Add stats about raw vs filtered signals
                    st.subheader("ATR-Filtered Signals Stats")
                    
                    # Calculate stats
                    raw_buy_count = (st.session_state.signals['buy_signal'] == True).sum()
                    raw_sell_count = (st.session_state.signals['sell_signal'] == True).sum()
                    filtered_buy_count = (st.session_state.signals['filtered_buy_signal'] == True).sum() if 'filtered_buy_signal' in st.session_state.signals.columns else 0
                    filtered_sell_count = (st.session_state.signals['filtered_sell_signal'] == True).sum() if 'filtered_sell_signal' in st.session_state.signals.columns else 0
                    
                    # Display stats in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Raw Buy Signals", raw_buy_count)
                        st.metric("Raw Sell Signals", raw_sell_count)
                    with col2:
                        st.metric("Filtered Buy Signals", filtered_buy_count)
                        st.metric("Filtered Sell Signals", filtered_sell_count)
                    with col3:
                        reduction_pct_buy = ((raw_buy_count - filtered_buy_count) / raw_buy_count * 100) if raw_buy_count > 0 else 0
                        reduction_pct_sell = ((raw_sell_count - filtered_sell_count) / raw_sell_count * 100) if raw_sell_count > 0 else 0
                        st.metric("Noise Reduction (Buy)", f"{reduction_pct_buy:.1f}%")
                        st.metric("Noise Reduction (Sell)", f"{reduction_pct_sell:.1f}%")
                    
                    # Show info about the ATR filter
                    symbol = st.session_state.symbol
                    if symbol == 'SPY' and hasattr(st.session_state, 'use_spy_strategy') and st.session_state.use_spy_strategy:
                        st.info("Using optimized SPY-specific ATR filter settings (k=0.5)")
                    else:
                        st.info("Using standard ATR filter settings (k=0.7)")
                    
                    # Create the signals dataframe
                    signals_df = pd.DataFrame(all_signals_data)
                    
                    # Style the dataframe
                    def highlight_signals(row):
                        # No signal
                        if row['Raw Signal'] == 'None' and row['Filtered Signal'] == 'None':
                            return [''] * len(row)
                            
                        # Filter removed a signal (highlight in yellow)
                        if row['Raw Signal'] != 'None' and row['Filtered Signal'] == 'None':
                            style = ['background-color: #fff2cc'] * len(row)
                            return style
                        
                        # Raw buy signal
                        if row['Raw Signal'] == 'Buy':
                            style = ['background-color: #d4f7d4'] * len(row)
                            
                            # If filtered signal is different, highlight it
                            if row['Filtered Signal'] != 'Buy':
                                style[2] = 'background-color: #fff2cc; color: #000000'
                            return style
                            
                        # Raw sell signal
                        if row['Raw Signal'] == 'Sell':
                            style = ['background-color: #f7d4d4'] * len(row)
                            
                            # If filtered signal is different, highlight it
                            if row['Filtered Signal'] != 'Sell':
                                style[2] = 'background-color: #fff2cc; color: #000000'
                            return style
                            
                        return [''] * len(row)
                    
                    st.dataframe(
                        signals_df.style.apply(highlight_signals, axis=1),
                        use_container_width=True
                    )
                else:
                    st.warning("No buy or sell signals found in the data")
                
                # Now render signal visualization
                st.write("About to render signals...")
                from components.signals import render_signals
                render_signals(st.session_state.data, st.session_state.signals, st.session_state.symbol)
                
                # Add a button to navigate to the Paired Signals page
                if 'paired_results' in st.session_state and 'paired_signals' in st.session_state.paired_results:
                    paired_signals = st.session_state.paired_results['paired_signals']
                    if not paired_signals.empty:
                        st.info("For more detailed analysis of paired entry-exit signals, visit the Paired Signals page.")
                        if st.button("Go to Paired Signals Analysis"):
                            # Set a flag to navigate to the paired signals page
                            st.session_state.navigate_to_paired_signals = True
                            st.experimental_rerun()
                
                st.write("Signals should be rendered above this line")
            except Exception as e:
                st.error(f"Error in signals tab: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Tab 3: ORB Analysis
        with tabs[2]:
            st.header("DEBUG: Checking ORB analysis tab")
            try:
                # Ensure we have signals
                if 'signals' not in st.session_state or st.session_state.signals is None:
                    st.write("No signals available for ORB analysis")
                else:
                    st.write("About to render ORB signals...")
                    from components.signals import render_orb_signals
                    render_orb_signals(st.session_state.data, st.session_state.signals, st.session_state.symbol)
                    st.write("ORB signals should be rendered above this line")
            except Exception as e:
                st.error(f"Error in ORB analysis tab: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        # Tab 4: Exit Strategy Analysis
        with tabs[3]:
            st.header("Exit Strategy Analysis")
            try:
                # Ensure we have data
                if 'data' not in st.session_state or st.session_state.data is None:
                    st.write("No data available for exit strategy analysis")
                else:
                    render_exit_strategy_analysis(st.session_state.data)
            except Exception as e:
                st.error(f"Error in exit strategy analysis tab: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    try:
        import pytz  # Make sure pytz is available for timezone handling
    except ImportError:
        st.error("Missing required package: pytz. Install with 'pip install pytz'")
        st.stop()
    
    # Set up proper event loop for Streamlit
    try:
        setup_event_loop()
    except Exception as e:
        st.error(f"Failed to set up event loop: {str(e)}")
    
    # Initialize IBKR client if available
    if IBKR_AVAILABLE:
        try:
            init_ibkr_client()
        except Exception as e:
            st.error(f"Error initializing IBKR client: {str(e)}")
        
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}") 
        traceback.print_exc() 