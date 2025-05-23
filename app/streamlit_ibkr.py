import streamlit as st
import pandas as pd
import numpy as np
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging
import socket
import psutil
import requests

# Try different import approaches to handle both module and script execution
try:
    # When imported as a module
    from app.ibkr_client import IBKRClient
    from app.signals.generator import generate_signals, generate_signals_advanced, analyze_single_day
    from app.components.signals import render_signals, render_orb_signals, render_signal_table
    from app.asyncio_patch import ensure_event_loop, with_event_loop, run_periodically, background_loop
except ImportError:
    try:
        # When run directly
        from ibkr_client import IBKRClient
        from signals.generator import generate_signals, generate_signals_advanced, analyze_single_day
        from components.signals import render_signals, render_orb_signals, render_signal_table
        from asyncio_patch import ensure_event_loop, with_event_loop, run_periodically, background_loop
    except ImportError as e:
        print(f"Failed to import required modules: {e}")
        # Create dummy versions for graceful degradation
        IBKRClient = None

logger = logging.getLogger(__name__)

def register_ibkr_sync_task():
    """Register the IBKR sync task with the background loop manager"""
    if 'ibkr_sync_registered' not in st.session_state:
        st.session_state.ibkr_sync_registered = False
        
    if not st.session_state.ibkr_sync_registered:
        try:
            # Define a function that safely syncs IBKR data to session_state
            def sync_ibkr_data():
                try:
                    # Global variable for IBKR client (from main.py)
                    from app.main import ibkr_client
                    
                    # If we have a global client, use it to collect data
                    # We'll then sync it in the main thread
                    if ibkr_client is not None:
                        # We collect data but don't touch session_state here
                        ibkr_client.run_one_iteration()
                except Exception as e:
                    logger.error(f"Error in IBKR sync task: {str(e)}")
            
            # Register with the background loop
            background_loop.add_function("ibkr_sync", sync_ibkr_data)
            background_loop.interval = min(background_loop.interval, 0.1)  # Increased frequency for better responsiveness  # Run at least every 0.5 seconds
            
            st.session_state.ibkr_sync_registered = True
            logger.info("Registered IBKR sync task with background loop")
        except Exception as e:
            logger.error(f"Failed to register IBKR sync task: {str(e)}")

def init_ibkr_client():
    """Initialize the IBKR client if not already in session state"""
    if 'ibkr' not in st.session_state:
        st.session_state.ibkr = IBKRClient()
        
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
        
    if 'streaming_symbol' not in st.session_state:
        st.session_state.streaming_symbol = None
        
    if 'auto_signal_generation' not in st.session_state:
        st.session_state.auto_signal_generation = False
        
    if 'collected_bars' not in st.session_state:
        st.session_state.collected_bars = {}
        
    if 'last_signal_time' not in st.session_state:
        st.session_state.last_signal_time = datetime.now() - timedelta(minutes=5)
        
    if 'signals' not in st.session_state:
        st.session_state.signals = None
    
    # Register the sync task with the background loop
    register_ibkr_sync_task()

def connect_to_ibkr(port=4001):
    """Connect to IBKR TWS/Gateway with UI feedback"""
    
    st.subheader("IBKR Connection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        connection_status = st.empty()
        
        if not hasattr(st.session_state.ibkr, 'connected') or not st.session_state.ibkr.connected:
            connection_status.info("Connecting to Interactive Brokers...")
            try:
                # Update port if specified
                st.session_state.ibkr.port = port
                
                # Try to connect
                st.session_state.ibkr.connect()
                
                if st.session_state.ibkr.connected:
                    # Check if time offset is significant
                    if hasattr(st.session_state.ibkr, 'time_offset') and abs(st.session_state.ibkr.time_offset) > 5:
                        connection_status.warning(f"⚠️ Connected, but time offset detected: {st.session_state.ibkr.time_offset:.2f} seconds. This might affect data quality.")
                    else:
                        connection_status.success("✅ Connected to Interactive Brokers")
                else:
                    connection_status.error("❌ Failed to connect to Interactive Brokers. Check TWS/Gateway is running.")
            except Exception as e:
                connection_status.error(f"❌ Connection Error: {str(e)}")
        else:
            # Check if time offset is significant
            if hasattr(st.session_state.ibkr, 'time_offset') and abs(st.session_state.ibkr.time_offset) > 5:
                connection_status.warning(f"⚠️ Connected, but time offset detected: {st.session_state.ibkr.time_offset:.2f} seconds. This might affect data quality.")
            else:
                connection_status.success("✅ Connected to Interactive Brokers")
            
    with col2:
        # IBKR connection port selector
        port_options = {
            "TWS (Live)": 7496, 
            "TWS (Paper)": 7497,
            "Gateway (Live)": 4001, 
            "Gateway (Paper)": 4002
        }
        selected_app = st.radio("Application", list(port_options.keys()))
        port = port_options[selected_app]
        
        if st.button("Reconnect"):
            if hasattr(st.session_state.ibkr, 'connected') and st.session_state.ibkr.connected:
                st.session_state.ibkr.disconnect()
            
            # Clear connection status
            connection_status.info("Reconnecting to Interactive Brokers...")
            
            # Attempt to reconnect with the selected port
            try:
                st.session_state.ibkr.port = port
                st.session_state.ibkr.connect()
                
                if st.session_state.ibkr.connected:
                    # Check if time offset is significant
                    if hasattr(st.session_state.ibkr, 'time_offset') and abs(st.session_state.ibkr.time_offset) > 5:
                        connection_status.warning(f"⚠️ Connected, but time offset detected: {st.session_state.ibkr.time_offset:.2f} seconds. This might affect data quality.")
                    else:
                        connection_status.success("✅ Reconnected to Interactive Brokers")
                else:
                    connection_status.error("❌ Failed to reconnect to Interactive Brokers")
            except Exception as e:
                connection_status.error(f"❌ Reconnection Error: {str(e)}")
                
    # Add advanced troubleshooting options
    with st.expander("Advanced Troubleshooting"):
        st.markdown("""
        ### Common IBKR API Issues and Solutions
        
        1. **Book is absent during iserv update**
           - This is a server-side issue often related to market data feeds
           - Solutions:
             - Wait a few minutes and try again
             - Try switching between LIVE and DELAYED market data
             - Restart the IB Gateway/TWS
        
        2. **Time synchronization issues**
           - If you see time offset warnings, your computer's clock might be out of sync
           - Solutions:
             - Ensure your system clock is accurate and synced with internet time
             - Restart the IB Gateway/TWS
        
        3. **No market data**
           - Check if you have the required market data subscriptions
           - Solutions:
             - In TWS/Gateway, check that market data is enabled
             - Try requesting delayed data instead of live data
             - Verify your account has access to the markets you're requesting
        """)
        
        if st.button("Force Time Sync"):
            try:
                time_diff = st.session_state.ibkr.handle_time_offset()
                st.info(f"Time difference between local and server: {time_diff:.2f} seconds")
            except Exception as e:
                st.error(f"Failed to sync time: {str(e)}")
                
        if st.button("Try Delayed Data"):
            try:
                st.session_state.ibkr.ib.reqMarketDataType(3)
                st.info("Switched to DELAYED market data (type 3)")
            except Exception as e:
                st.error(f"Failed to set delayed data: {str(e)}")
                
    return st.session_state.ibkr.connected

def render_streaming_interface():
    """Render the real-time data streaming interface"""
    
    st.subheader("Real-Time Data Streaming")
    
    # Symbol selection
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        symbol = st.text_input("Symbol", value=st.session_state.get('symbol', 'SPY')).upper()
        
    with col2:
        exchange = st.selectbox("Exchange", ["SMART", "NASDAQ", "NYSE", "ARCA", "BATS", "IEX"], index=0)
        
    with col3:
        currency = st.selectbox("Currency", ["USD", "CAD", "EUR", "GBP"], index=0)
    
    # Sync data from background threads to session_state safely
    if hasattr(st.session_state, 'ibkr') and st.session_state.ibkr:
        if hasattr(st.session_state.ibkr, 'sync_to_streamlit'):
            st.session_state.ibkr.sync_to_streamlit()
        else:
            logger.warning("IBKRClient object missing sync_to_streamlit method")
    
    # Market data connection status
    if hasattr(st.session_state, 'market_data_type') and st.session_state.market_data_type:
        # Show the most recent market data type (higher reqId is more recent)
        latest_reqId = max(st.session_state.market_data_type.keys())
        data_type = st.session_state.market_data_type[latest_reqId]
        data_type_labels = {1: "LIVE", 2: "FROZEN", 3: "DELAYED", 4: "DELAYED FROZEN"}
        data_type_colors = {1: "green", 2: "orange", 3: "orange", 4: "red"}
        
        st.info(f"Market Data Type: **{data_type_labels.get(data_type, 'UNKNOWN')}** (Type {data_type})")
        
        # Show warning if not live data
        if data_type != 1:
            st.warning(f"You are receiving **{data_type_labels.get(data_type, 'UNKNOWN')}** market data. Live trading requires real-time data subscription.")
    
    # Add a test connection button
    if st.button("Test Market Data Connection"):
        try:
            st.info("Testing market data connection...")
            result = st.session_state.ibkr.test_market_data(symbol, exchange, currency)
            if result:
                # Show detailed test results including market price
                market_price = result.get('market_price', float('nan'))
                last_price = result.get('last', float('nan'))
                bid_price = result.get('bid', float('nan'))
                ask_price = result.get('ask', float('nan'))
                
                # Display the test results
                if not (pd.isna(market_price) and pd.isna(last_price) and pd.isna(bid_price) and pd.isna(ask_price)):
                    st.success(f"✅ Market data test successful:")
                    
                    # Create a table to display the results
                    results_data = {
                        "Field": ["Market Price", "Last Price", "Bid Price", "Ask Price", "Volume"],
                        "Value": [
                            f"${market_price:.2f}" if not pd.isna(market_price) else "N/A",
                            f"${last_price:.2f}" if not pd.isna(last_price) else "N/A",
                            f"${bid_price:.2f}" if not pd.isna(bid_price) else "N/A",
                            f"${ask_price:.2f}" if not pd.isna(ask_price) else "N/A",
                            f"{result.get('volume', 0):,}" if result.get('volume', 0) > 0 else "N/A"
                        ]
                    }
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, hide_index=True)
                    
                    # Check if we got any price data
                    if (pd.isna(market_price) and pd.isna(last_price) and pd.isna(bid_price) and pd.isna(ask_price)):
                        st.warning("⚠️ Connection OK but received empty price data. Market may be closed or subscription required.")
                else:
                    st.warning("⚠️ Connection OK but received empty market data. Market may be closed or subscription required.")
            else:
                st.error("❌ Market data test failed. Check TWS/Gateway settings.")
                
                # Add troubleshooting tips
                with st.expander("Troubleshooting Tips"):
                    st.markdown("""
                    ### Market Data Subscription Issues
                    
                    1. **Check your market data subscriptions** in TWS/Gateway:
                       - Log in to IBKR Account Management
                       - Go to Settings > Market Data Subscriptions
                       - Verify you have subscriptions for the exchange and symbol
                    
                    2. **Try different market data settings**:
                       - In TWS, go to Global Configuration > API > Settings
                       - Try enabling "Use market data for your region"
                    
                    3. **Verify your account permissions**:
                       - Some accounts may have restrictions on market data
                       - Contact IBKR support if you believe you should have access
                    """)
        except Exception as e:
            st.error(f"❌ Error testing connection: {str(e)}")
        
    # Streaming controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not st.session_state.streaming:
            if st.button("Start Streaming", use_container_width=True):
                # Save current settings to session state
                st.session_state.symbol = symbol
                st.session_state.streaming = True
                st.session_state.streaming_symbol = symbol
                
                # Force LIVE market data
                if hasattr(st.session_state, 'ibkr') and st.session_state.ibkr.connected:
                    st.session_state.ibkr.ib.reqMarketDataType(1)
                
                st.experimental_rerun()
        else:
            if st.button("Stop Streaming", use_container_width=True):
                # Stop the current stream
                try:
                    if hasattr(st.session_state, 'ibkr') and st.session_state.ibkr.connected:
                        st.session_state.ibkr.stop_streaming(
                            st.session_state.streaming_symbol or symbol, 
                            exchange, 
                            currency
                        )
                except Exception as e:
                    logger.error(f"Error stopping stream: {str(e)}")
                
                st.session_state.streaming = False
                st.session_state.streaming_symbol = None
                st.experimental_rerun()
    
    with col2:
        auto_signal = st.checkbox("Auto-Generate Signals", value=st.session_state.get('auto_signal_generation', False),
                                help="Automatically generate trading signals from real-time data")
        st.session_state.auto_signal_generation = auto_signal
        
    with col3:
        bar_collection = st.checkbox("Collect 1-min Bars", value=True,
                                  help="Collect 1-minute OHLCV bars for technical analysis")
    
    # Data Type Override section
    with st.expander("Advanced Market Data Settings"):
        md_type = st.radio(
            "Market Data Type",
            ["Live (1)", "Frozen (2)", "Delayed (3)", "Delayed Frozen (4)"],
            index=0,
            horizontal=True,
            help="Force a specific market data type request"
        )
        
        if st.button("Apply Market Data Type"):
            md_type_id = int(md_type.split("(")[1].split(")")[0])
            if hasattr(st.session_state, 'ibkr') and st.session_state.ibkr.connected:
                st.session_state.ibkr.ib.reqMarketDataType(md_type_id)
                st.success(f"Set market data type to {md_type}")
                # Force refresh any active streams
                if st.session_state.streaming and st.session_state.streaming_symbol:
                    key = f"{st.session_state.streaming_symbol}.{exchange}.{currency}"
                    if key in st.session_state.ibkr.data_streams:
                        contract = st.session_state.ibkr.data_streams[key].contract
                        # Cancel existing subscription
                        st.session_state.ibkr.ib.cancelMktData(contract)
                        st.session_state.ibkr.ib.sleep(1)
                        # Request data with new market data type
                        generic_tick_list = '100,101,105,106,165,221,225,233,236,258,411'
                        ticker = st.session_state.ibkr.ib.reqMktData(contract, genericTickList=generic_tick_list, snapshot=False, regulatorySnapshot=False)
                        st.session_state.ibkr.data_streams[key] = ticker
    
    # Display real-time data if streaming is active
    if st.session_state.streaming and st.session_state.streaming_symbol:
        try:
            # Start/continue streaming the selected symbol
            ticker = st.session_state.ibkr.stream_data(
                st.session_state.streaming_symbol, 
                exchange, 
                currency
            )
            
            # Show streaming status
            st.success(f"✅ Streaming {st.session_state.streaming_symbol} data in real-time")
            
            # Display the ticker data
            ticker_data_container = st.container()
            
            with ticker_data_container:
                # Get the data from session state
                data_key = f"{st.session_state.streaming_symbol}_data"
                
                if data_key in st.session_state:
                    ticker_data = st.session_state[data_key]
                    
                    # Create columns for the live data display
                    col1, col2, col3 = st.columns(3)
                    last_price = ticker_data.get('last', 0)
                    
                    with col1:
                        st.metric(
                            "Last Price", 
                            f"${last_price:.2f}" if last_price else "N/A", 
                            delta=None,
                            delta_color="normal"
                        )
                        
                    with col2:
                        bid = ticker_data.get('bid', 0)
                        ask = ticker_data.get('ask', 0)
                        if bid and ask:
                            st.metric(
                                "Bid/Ask", 
                                f"${bid:.2f} / ${ask:.2f}", 
                                delta=f"Spread: ${(ask - bid):.2f}",
                                delta_color="normal"
                            )
                        else:
                            st.metric("Bid/Ask", "N/A", delta=None)
                        
                    with col3:
                        # Format volume with commas for thousands
                        volume = ticker_data.get('volume', 0)
                        formatted_volume = f"{volume:,}" if volume else "N/A"
                        st.metric(
                            "Volume", 
                            formatted_volume,
                            delta=None
                        )
                    
                    # Add a manual refresh button
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Force Refresh Data"):
                            # Force a reconnection to the market data feed
                            try:
                                st.session_state.ibkr.ib.reqMarketDataType(1)  # Request LIVE data
                                st.info("Refreshing market data...")
                                # Get contract
                                key = f"{st.session_state.streaming_symbol}.{exchange}.{currency}"
                                if key in st.session_state.ibkr.data_streams:
                                    contract = st.session_state.ibkr.data_streams[key].contract
                                    # Cancel and re-request
                                    st.session_state.ibkr.ib.cancelMktData(contract)
                                    st.session_state.ibkr.ib.sleep(1)
                                    generic_tick_list = '100,101,105,106,165,221,225,233,236,258,411'
                                    ticker = st.session_state.ibkr.ib.reqMktData(contract, genericTickList=generic_tick_list, snapshot=False, regulatorySnapshot=False)
                                    st.session_state.ibkr.data_streams[key] = ticker
                                    st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error refreshing data: {str(e)}")
                    
                    with col2:
                        if st.button("Request All Tick Types"):
                            try:
                                st.info("Requesting all available tick types...")
                                key = f"{st.session_state.streaming_symbol}.{exchange}.{currency}"
                                if key in st.session_state.ibkr.data_streams:
                                    contract = st.session_state.ibkr.data_streams[key].contract
                                    # Cancel existing request
                                    st.session_state.ibkr.ib.cancelMktData(contract)
                                    st.session_state.ibkr.ib.sleep(1)
                                    # Request all tick types
                                    all_ticks = '100,101,105,106,165,221,225,233,236,258,411'
                                    ticker = st.session_state.ibkr.ib.reqMktData(contract, genericTickList=all_ticks, snapshot=False, regulatorySnapshot=False)
                                    st.session_state.ibkr.data_streams[key] = ticker
                                    st.success("Requested all tick types")
                                    st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error requesting tick types: {str(e)}")
                    
                    # Display timestamp of last update
                    last_update = ticker_data.get('timestamp')
                    if last_update:
                        st.caption(f"Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Get 1-minute bars if available
                    df = st.session_state.ibkr.get_ohlcv_dataframe(st.session_state.streaming_symbol)
                    
                    if not df.empty:
                        # Display the OHLCV chart
                        st.subheader("Price Chart (1-minute bars)")
                        
                        # Create figure
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Add candlestick chart
                        fig.add_trace(
                            go.Candlestick(
                                x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name="Price"
                            )
                        )
                        
                        # Add volume as bar chart on secondary axis
                        fig.add_trace(
                            go.Bar(
                                x=df.index,
                                y=df['volume'],
                                name="Volume",
                                marker=dict(color='rgba(100, 100, 255, 0.3)'),
                                opacity=0.3
                            ),
                            secondary_y=True
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{st.session_state.streaming_symbol} Real-Time Data",
                            xaxis_title="Time",
                            yaxis_title="Price",
                            xaxis_rangeslider_visible=False,
                            height=500
                        )
                        
                        # Update y-axes titles
                        fig.update_yaxes(title_text="Price", secondary_y=False)
                        fig.update_yaxes(title_text="Volume", secondary_y=True)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Automatically generate signals if enabled
                        if st.session_state.auto_signal_generation and df.shape[0] >= 20:  # Need enough bars for signals
                            now = datetime.now()
                            # Generate signals at most once per minute
                            if (now - st.session_state.last_signal_time).total_seconds() >= 60:
                                with st.spinner("Generating signals..."):
                                    # Generate signals based on collected bars
                                    signals = generate_signals(df)
                                    st.session_state.signals = signals
                                    st.session_state.last_signal_time = now
                                    
                                    # Render the signals
                                    render_signals(df, signals, st.session_state.streaming_symbol)
                    else:
                        st.info("Collecting price data... Please wait for enough bars to display the chart.")
                else:
                    st.info("Waiting for real-time data... Please wait.")
            
            # Use a safer method for UI updates instead of experimental_rerun
            time.sleep(0.1)  # Short delay for better performance
            
            # Create a regular interval to check for updates without using rerun
            # This will be much more efficient and reliable
            current_time = time.time()
            if 'next_ui_update' not in st.session_state:
                st.session_state.next_ui_update = current_time + 1  # Initial update in 1 second
            
            if current_time >= st.session_state.next_ui_update:
                # We'll perform a limited UI update every second
                # by syncing data and updating the next update time
                # This creates a smoother experience without full page reruns
                st.session_state.next_ui_update = current_time + 0.5  # Reduced from 2 to 0.5 for more frequent updates  # Schedule next update (2 seconds to reduce rerun frequency)
                
                # Sync again to get the latest data without a full rerun
                if hasattr(st.session_state, 'ibkr') and st.session_state.ibkr:
                    if hasattr(st.session_state.ibkr, 'sync_to_streamlit'):
                        st.session_state.ibkr.sync_to_streamlit()
                    else:
                        logger.warning("IBKRClient object missing sync_to_streamlit method")
                
                # Only force a rerun if there are pending updates and not too frequent
                if st.session_state.get('pending_update', False) and 'last_rerun_time' in st.session_state:
                    # Limit reruns to at most once every 3 seconds
                    if current_time - st.session_state.last_rerun_time > 3:
                        st.session_state.pending_update = False  # Reset the flag
                        st.session_state.last_rerun_time = current_time
                        st.experimental_rerun()
                elif st.session_state.get('pending_update', False):
                    # First rerun
                    st.session_state.pending_update = False
                    st.session_state.last_rerun_time = current_time
                    st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Streaming error: {str(e)}")
            logger.error(f"Error in streaming interface: {str(e)}", exc_info=True)
            
            # Reset streaming status if error persists
            st.session_state.streaming = False
            st.session_state.streaming_symbol = None
            
def render_ibkr_data_page():
    """Render the full IBKR real-time data page"""
    
    st.title("Real-Time Trading Data")
    
    # Initialize IBKR client if needed
    init_ibkr_client()
    
    # Add debug checkbox at the top
    enable_debug = st.checkbox("Enable Debug Mode", value=st.session_state.get("debug_mode", False))
    st.session_state.debug_mode = enable_debug
    
    if enable_debug:
        with st.expander("Debug Information", expanded=True):
            st.write("### IB Client Status")
            
            if 'ibkr' in st.session_state:
                # Connection details
                st.write("Connection Status:", st.session_state.ibkr.connected)
                st.write("Host:", st.session_state.ibkr.host)
                st.write("Port:", st.session_state.ibkr.port)
                st.write("Client ID:", st.session_state.ibkr.client_id)
                
                # Market data types
                if hasattr(st.session_state, 'market_data_types'):
                    st.write("Market Data Types:", st.session_state.market_data_types)
                
                # Time offset
                if hasattr(st.session_state.ibkr, 'time_offset'):
                    st.write("Time Offset:", f"{st.session_state.ibkr.time_offset:.2f} seconds")
                
                # Active streams
                if hasattr(st.session_state.ibkr, 'data_streams'):
                    st.write("Active Data Streams:", list(st.session_state.ibkr.data_streams.keys()))
                
                # Session data
                data_keys = [k for k in st.session_state.keys() if k.endswith('_data')]
                if data_keys:
                    st.write("Data in Session State:", data_keys)
    
    # Connect to IBKR
    connected = connect_to_ibkr()
    
    if connected:
        render_streaming_interface()
    else:
        st.error("Please connect to Interactive Brokers to access real-time data.")
        st.info("Make sure TWS or IB Gateway is running with API connections enabled.")
        
        with st.expander("Connection Troubleshooting"):
            st.markdown("""
            ### Troubleshooting IBKR Connection
            
            1. **Check if TWS or IB Gateway is running** - The application must be running to connect
            2. **Verify API settings** - In TWS, go to File > Global Configuration > API > Settings
                - Ensure "Enable ActiveX and Socket Clients" is checked
                - Set the socket port (usually 7497 for TWS or 4002 for Gateway)
                - Allow connections from localhost
            3. **Check your firewall** - Make sure the application is allowed through your firewall
            4. **Restart TWS or Gateway** - Sometimes a restart can resolve connection issues
            5. **Check API permissions** - Ensure your account has API access enabled
            """)
            
            # Advanced diagnostics for debug mode
            if enable_debug:
                st.write("### Advanced Diagnostics")
                
                if st.button("Run Connection Diagnostics"):
                    try:
                        st.write("Testing local port availability...")
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        port_result = s.connect_ex(('127.0.0.1', st.session_state.ibkr.port))
                        s.close()
                        
                        if port_result == 0:
                            st.success(f"Port {st.session_state.ibkr.port} is open and available")
                        else:
                            st.error(f"Port {st.session_state.ibkr.port} is not accessible. Check if IB Gateway/TWS is running.")
                            
                        # Check if IB process is running
                        st.write("Checking for IB processes...")
                        try:
                            ib_processes = [p.name() for p in psutil.process_iter() if 'tws' in p.name().lower() or 'gateway' in p.name().lower()]
                            if ib_processes:
                                st.success(f"Found IB processes: {', '.join(ib_processes)}")
                            else:
                                st.warning("No IB processes found running")
                        except:
                            st.warning("Unable to check for running processes")
                            
                        # Check network connectivity
                        st.write("Testing network connectivity to IB servers...")
                        try:
                            response = requests.get("https://www.interactivebrokers.com", timeout=5)
                            st.success(f"Successfully connected to IB website: {response.status_code}")
                        except Exception as e:
                            st.error(f"Failed to connect to IB website: {str(e)}")
                            
                    except Exception as e:
                        st.error(f"Diagnostics error: {str(e)}")
                
                # Force log events
                if st.button("Show recent log events"):
                    for handler in logging.getLogger().handlers:
                        if isinstance(handler, logging.StreamHandler):
                            messages = handler.stream.getvalue() if hasattr(handler.stream, 'getvalue') else "Log data not available"
                            st.code(messages)

def add_ibkr_to_main_app():
    """Add IBKR real-time capability to the sidebar in the main app"""
    
    # Initialize IBKR client if needed
    init_ibkr_client()
    
    # Sync data safely from background threads
    if hasattr(st.session_state, 'ibkr') and st.session_state.ibkr:
        if hasattr(st.session_state.ibkr, 'sync_to_streamlit'):
            st.session_state.ibkr.sync_to_streamlit()
        else:
            logger.warning("IBKRClient object missing sync_to_streamlit method")
    
    # Add to sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Real-Time Trading")
    
    # Show connection status
    if 'ibkr' in st.session_state and hasattr(st.session_state.ibkr, 'connected'):
        if st.session_state.ibkr.connected:
            st.sidebar.success("✅ Connected to IBKR")
        else:
            st.sidebar.error("❌ Not connected to IBKR")
            
            if st.sidebar.button("Connect to IBKR"):
                try:
                    st.session_state.ibkr.connect()
                    st.experimental_rerun()
                except Exception as e:
                    st.sidebar.error(f"Connection error: {str(e)}")
    else:
        st.sidebar.warning("⚠️ IBKR client not initialized")
        
    # Streaming controls in sidebar
    if hasattr(st.session_state, 'ibkr') and st.session_state.ibkr.connected:
        st.sidebar.markdown("---")
        
        streaming_symbol = st.sidebar.text_input(
            "Symbol to Stream", 
            value=st.session_state.get('symbol', 'SPY')
        ).upper()
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if not st.session_state.streaming:
                if st.button("Start Streaming", key="start_stream_btn"):
                    st.session_state.streaming = True
                    st.session_state.streaming_symbol = streaming_symbol
                    st.experimental_rerun()
            else:
                if st.button("Stop Streaming", key="stop_stream_btn"):
                    # Stop the current stream
                    try:
                        st.session_state.ibkr.stop_streaming(st.session_state.streaming_symbol)
                    except:
                        pass
                    
                    st.session_state.streaming = False
                    st.session_state.streaming_symbol = None
                    st.experimental_rerun()
                    
        with col2:
            auto_signals = st.checkbox(
                "Auto Signals", 
                value=st.session_state.get('auto_signal_generation', False),
                key="sidebar_auto_signals"
            )
            st.session_state.auto_signal_generation = auto_signals
            
        # Show streaming status
        if st.session_state.streaming and st.session_state.streaming_symbol:
            data_key = f"{st.session_state.streaming_symbol}_data"
            if data_key in st.session_state:
                ticker_data = st.session_state[data_key]
                last_price = ticker_data.get('last', 0)
                
                st.sidebar.metric(
                    f"{st.session_state.streaming_symbol} Price", 
                    f"${last_price:.2f}"
                )
            else:
                st.sidebar.info(f"Streaming {st.session_state.streaming_symbol}...")
                
            # Quick link to real-time page
            if st.sidebar.button("View Real-Time Dashboard"):
                # Set flag to navigate to real-time page
                st.session_state.active_page = "real_time"
                st.experimental_rerun()
    
    # Link to real-time page
    st.sidebar.markdown("---")
    if st.sidebar.button("Real-Time Trading Dashboard", use_container_width=True):
        # Set flag to navigate to real-time page
        st.session_state.active_page = "real_time"
        st.experimental_rerun() 