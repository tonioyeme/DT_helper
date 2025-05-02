"""
Enhanced IBKR Client - Market Data Fix

This version of the IBKRClient incorporates fixes for:
1. NaN market data values - auto-switching between LIVE and DELAYED data
2. Client ID conflicts - using randomized client IDs
3. Time synchronization issues - better handling of server time
4. Thread safety - properly syncing data between background threads and Streamlit
"""

try:
    from app.asyncio_patch import ensure_event_loop, with_event_loop, run_in_background, get_background_results
except ImportError:
    # Try direct import when running as a module
    from asyncio_patch import ensure_event_loop, with_event_loop, run_in_background, get_background_results

from ib_insync import *
import nest_asyncio
import threading
import logging
import time
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import asyncio
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Callable
import random
import pytz

# Configure logging
logger = logging.getLogger(__name__)

# Initialize event loop before importing ib_insync internals
ensure_event_loop()

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Patch asyncio to work better with Streamlit
util.patchAsyncio()

class IBKRClientEnhanced:
    """
    Enhanced Interactive Brokers client for real-time market data streaming
    using the ib_insync library with improved market data handling.
    """
    @run_in_background
    def connection_monitor(self):
        """
        Monitor the IB connection and attempt to reconnect if needed
        This runs in the background and doesn't access session_state
        """
        while True:
            try:
                if not self.ib.isConnected():
                    logger.warning("IB connection lost, attempting to reconnect...")
                    self.connected = False
                    self.connect()
                # Use util.sleep instead of time.sleep for better compatibility 
                util.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in connection monitor: {str(e)}")
                util.sleep(5)

    def __init__(self, host='127.0.0.1', port=4001, client_id=None, timeout=20):
        """
        Initialize the enhanced IBKR client
        
        Args:
            host (str): IBKR TWS/Gateway host address
            port (int): IBKR TWS/Gateway port (default: 4001 for IB Gateway, 7496 for TWS)
            client_id (int): Client ID for IB connection (random if None)
            timeout (int): Timeout in seconds for connection attempts
        """
        # Ensure event loop exists
        loop = ensure_event_loop()
        self.ib = IB()
        self.host = host
        self.port = port
        
        # Use a random client ID if none provided to avoid conflicts
        if client_id is None:
            self.client_id = random.randint(1000, 9999)
            logger.info(f"Using random client ID: {self.client_id}")
        else:
            self.client_id = client_id
            
        self.timeout = timeout
        self.connected = False
        self.data_streams = {}
        self.history_data = {}
        self.lock = threading.Lock()
        self.ticker_callbacks = {}
        self.last_1m_bar = {}
        self.last_volumes = {}
        self.last_prices = {}
        self.last_reconnect_attempt = None
        self.reconnect_interval = 10  # seconds between reconnection attempts
        self.current_market_data_type = None
        
        # Thread-safe containers for ticker updates
        self.ticker_data_updates = {}
        self.pending_updates = False
        
        # Add handlers for specific event types
        self.ib.marketDataTypeEvent += self.on_market_data_type
    
    def on_market_data_type(self, reqId, marketDataType):
        """
        Handler for market data type events
        
        Args:
            reqId (int): Request ID
            marketDataType (int): Market data type (1=Live, 2=Frozen, 3=Delayed, 4=Delayed Frozen)
        """
        type_names = {1: "LIVE", 2: "FROZEN", 3: "DELAYED", 4: "DELAYED_FROZEN"}
        type_name = type_names.get(marketDataType, f"UNKNOWN({marketDataType})")
        logger.info(f"Market Data Type: {type_name} for reqId {reqId}")
        
        # Store in a thread-safe way without directly accessing session_state
        self.current_market_data_type = marketDataType
        with self.lock:
            self.ticker_data_updates['market_data_type'] = {
                'reqId': reqId,
                'type': marketDataType,
                'type_name': type_name
            }
            self.pending_updates = True
    
    def _is_market_open(self):
        """
        Check if US markets are currently open
        Returns True if markets are open, False otherwise
        """
        # Get current time in Eastern timezone (US markets)
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
            
        # Check regular market hours (9:30 AM - 4:00 PM ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _select_market_data_type(self):
        """
        Select the appropriate market data type based on market hours
        - Use LIVE (1) during market hours
        - Use DELAYED (3) outside market hours
        """
        try:
            if self._is_market_open():
                logger.info("Markets are OPEN - requesting LIVE market data")
                self.ib.reqMarketDataType(1)
            else:
                logger.info("Markets are CLOSED - requesting DELAYED market data")
                self.ib.reqMarketDataType(3)
                
        except Exception as e:
            logger.warning(f"Error setting market data type: {str(e)}")
            # Fall back to delayed data
            logger.info("Falling back to DELAYED market data")
            try:
                self.ib.reqMarketDataType(3)
            except Exception as e2:
                logger.error(f"Error requesting DELAYED market data: {str(e2)}")
    
    def connect(self):
        """Connect to IBKR TWS/Gateway"""
        try:
            logger.info(f"Connecting to IBKR at {self.host}:{self.port} (client ID: {self.client_id})")
            
            # Ensure event loop
            loop = ensure_event_loop()
            
            # Create a new IB object if needed
            if self.ib is None:
                self.ib = IB()
                logger.info("Created new IB instance")
            
            # Connect to IBKR API
            if not self.ib.isConnected():
                logger.info("Connecting to IBKR...")
                self.ib.connect(
                    self.host, 
                    self.port, 
                    clientId=self.client_id,
                    timeout=self.timeout
                )
                logger.info(f"Connected successfully, isConnected={self.ib.isConnected()}")
            
            # Check if connected
            self.connected = self.ib.isConnected()
            logger.info(f"Final connection status: {self.connected}")
            
            if self.connected:
                # Synchronize time with server to prevent time offset issues
                self.handle_time_offset()
                
                # Select appropriate market data type based on market hours
                self._select_market_data_type()
                
                # Get account information
                try:
                    self.ib.reqAccountSummary()
                    logger.info("Requested account summary")
                except Exception as e:
                    logger.warning(f"Error requesting account summary: {str(e)}")
                
                # Start a background thread to monitor the connection
                if not hasattr(self, 'monitor_thread') or not self.monitor_thread.is_alive():
                    self.monitor_thread = threading.Thread(target=self.connection_monitor)
                    self.monitor_thread.daemon = True
                    self.monitor_thread.start()
                    logger.info("Started connection monitor thread")
            
            return self.connected
        except Exception as e:
            logger.error(f"Error in connect: {str(e)}")
            import traceback
            traceback.print_exc()
            self.connected = False
            return False
                
    def disconnect(self):
        """Disconnect from IB Gateway"""
        if self.connected:
            logger.info("Disconnecting from IB Gateway")
            self.ib.disconnect()
            self.connected = False
            self.data_streams = {}
    
    def ensure_connected(self):
        """Ensure connection is active, reconnect if needed"""
        try:
            if not self.connected or not self.ib.isConnected():
                current_time = time.time()
                
                # Limit reconnection attempts frequency
                if (self.last_reconnect_attempt is None or 
                    (current_time - self.last_reconnect_attempt > self.reconnect_interval)):
                    
                    logger.warning("IB Gateway connection lost, attempting to reconnect...")
                    self.last_reconnect_attempt = current_time
                    self.connected = False
                    
                    # Try to connect with retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            if self.connect():
                                logger.info("Successfully reconnected to IB Gateway")
                                return True
                            util.sleep(2)  # Wait between retries
                        except Exception as e:
                            logger.error(f"Reconnection attempt {attempt + 1} failed: {str(e)}")
                            if attempt < max_retries - 1:
                                util.sleep(2)
                    
                    logger.error("Failed to reconnect after multiple attempts")
                    return False
                else:
                    logger.debug("Waiting before reconnection attempt")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error in ensure_connected: {str(e)}")
            return False
    
    def stream_data(self, symbol, exchange='SMART', currency='USD'):
        """
        Stream real-time market data for the given symbol with improved
        data reliability and automatic switching between LIVE and DELAYED
        data based on market hours and data availability.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange name
            currency (str): Currency
            
        Returns:
            IB.Ticker: Ticker object with real-time data
        """
        logger.info(f"Starting data stream for {symbol} on exchange {exchange}")
        
        try:
            self.ensure_connected()
            
            # Define a unique key for this data stream
            key = f"{symbol}.{exchange}.{currency}"
            
            # Check if we already have this stream
            if key in self.data_streams:
                logger.info(f"Using existing data stream for {key}")
                return self.data_streams[key]
            
            # Create contract definition
            contract = Stock(symbol, exchange, currency)
            logger.info(f"Created contract {contract}")
            
            # Request market data with specific generic tick types that use Streaming Bundle
            generic_tick_list = "100,101,104,106,165"
            
            # Request market data with the current market data type
            ticker = self.ib.reqMktData(contract, genericTickList=generic_tick_list, snapshot=False, regulatorySnapshot=False)
            logger.info(f"Ticker created for {symbol}, contract ID: {ticker.contract.conId}")
            
            # Store the ticker
            self.data_streams[key] = ticker
            
            # Define an improved ticker update handler with auto-switching logic
            def on_ticker_update(ticker):
                try:
                    # Log what we received from ticker with detailed info
                    price = ticker.marketPrice()
                    logger.info(f"UPDATE - {symbol}: marketPrice={price}, last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}, volume={ticker.volume}")
                    
                    # Handle NaN values - try automatic data type switching if needed
                    if not (ticker.bid or ticker.ask or ticker.last):
                        logger.warning(f"No valid price data for {symbol}, trying data type switch")
                        
                        # If we're using LIVE data and market is closed, switch to DELAYED
                        if self.current_market_data_type == 1 and not self._is_market_open():
                            logger.info("Market appears closed, switching to DELAYED data")
                            self.ib.reqMarketDataType(3)
                        # If we're using DELAYED data and market is open, try LIVE
                        elif self.current_market_data_type == 3 and self._is_market_open():
                            logger.info("Market appears open, switching to LIVE data")
                            self.ib.reqMarketDataType(1)
                        # Otherwise just toggle between the two types
                        else:
                            new_type = 3 if self.current_market_data_type == 1 else 1
                            logger.info(f"Switching data type to {new_type}")
                            self.ib.reqMarketDataType(new_type)
                        
                        # Re-request market data
                        self.ib.reqMktData(ticker.contract, genericTickList=generic_tick_list, snapshot=False, regulatorySnapshot=False)
                        return
                    
                    # Process valid data
                    if price > 0 or ticker.last > 0 or (ticker.bid > 0 and ticker.ask > 0):
                        # Create a dict with current ticker data
                        current_data = {
                            'symbol': symbol,
                            'timestamp': datetime.now(),
                            'bid': ticker.bid if not np.isnan(ticker.bid) else None,
                            'ask': ticker.ask if not np.isnan(ticker.ask) else None,
                            'last': ticker.last if not np.isnan(ticker.last) and ticker.last > 0 else (
                                    price if not np.isnan(price) and price > 0 else None),
                            'close': ticker.close if not np.isnan(ticker.close) else None,
                            'open': ticker.open if not np.isnan(ticker.open) else None,
                            'high': ticker.high if not np.isnan(ticker.high) else None,
                            'low': ticker.low if not np.isnan(ticker.low) else None,
                            'volume': ticker.volume if not np.isnan(ticker.volume) else 0,
                            'bid_size': ticker.bidSize if not np.isnan(ticker.bidSize) else 0,
                            'ask_size': ticker.askSize if not np.isnan(ticker.askSize) else 0,
                            'market_data_type': self.current_market_data_type
                        }
                        
                        # Store in a thread-safe way
                        key_name = f"{symbol}_data"
                        with self.lock:
                            self.ticker_data_updates[key_name] = current_data
                            self.pending_updates = True
                        
                        # Update 1-minute bars if needed
                        self._update_1m_bars(symbol, current_data)
                        
                        # Call user callback if provided
                        cb = self.ticker_callbacks.get(key)
                        if cb:
                            cb(ticker, current_data)
                    else:
                        logger.warning(f"Received ticker update with invalid price data for {symbol}")
                except Exception as e:
                    logger.error(f"Error in ticker update handler: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Register the update handler
            ticker.updateEvent += on_ticker_update
            logger.info(f"Started streaming {symbol} from {exchange}")
            
            # Wait for initial data (up to 5 seconds)
            for i in range(5):
                self.ib.sleep(1)
                if ticker.last or ticker.bid or ticker.ask:
                    logger.info(f"Received initial data for {symbol} after {i+1} seconds")
                    break
                elif i == 4:  # Last attempt, try different market data type
                    logger.info(f"No initial data received for {symbol}, trying alternate market data type")
                    current_type = self.current_market_data_type or 1
                    new_type = 3 if current_type == 1 else 1
                    self.ib.reqMarketDataType(new_type)
            
            return ticker
            
        except Exception as e:
            logger.error(f"Error in stream_data: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _update_1m_bars(self, symbol, tick_data):
        """
        Update 1-minute bars from tick data
        
        Args:
            symbol (str): Trading symbol
            tick_data (dict): Current tick data
        """
        # Skip if we don't have valid price data
        if tick_data.get('last') is None:
            return
            
        current_time = tick_data['timestamp']
        current_minute = current_time.replace(second=0, microsecond=0)
        
        if symbol not in self.last_1m_bar:
            # Initialize bar for this symbol
            self.last_1m_bar[symbol] = {
                'minute': current_minute,
                'open': tick_data['last'],
                'high': tick_data['last'],
                'low': tick_data['last'],
                'close': tick_data['last'],
                'volume': tick_data.get('volume', 0)
            }
        else:
            last_bar = self.last_1m_bar[symbol]
            
            # If we're in a new minute, store the completed bar and start a new one
            if current_minute > last_bar['minute']:
                # Store completed bar in history
                if symbol not in self.history_data:
                    self.history_data[symbol] = []
                
                self.history_data[symbol].append({
                    'timestamp': last_bar['minute'],
                    'open': last_bar['open'],
                    'high': last_bar['high'],
                    'low': last_bar['low'],
                    'close': last_bar['close'],
                    'volume': last_bar['volume']
                })
                
                # Start a new bar
                self.last_1m_bar[symbol] = {
                    'minute': current_minute,
                    'open': tick_data['last'],
                    'high': tick_data['last'],
                    'low': tick_data['last'],
                    'close': tick_data['last'],
                    'volume': tick_data.get('volume', 0) - self.last_volumes.get(symbol, 0)
                }
                
                # Store current volume as reference
                self.last_volumes[symbol] = tick_data.get('volume', 0)
            else:
                # Update the current bar
                last_bar['high'] = max(last_bar['high'], tick_data['last'])
                last_bar['low'] = min(last_bar['low'], tick_data['last'])
                last_bar['close'] = tick_data['last']
                
                # Update volume if it changed
                if 'volume' in tick_data and tick_data['volume'] > self.last_volumes.get(symbol, 0):
                    vol_change = tick_data['volume'] - self.last_volumes.get(symbol, 0)
                    last_bar['volume'] += vol_change
                    self.last_volumes[symbol] = tick_data['volume']
    
    def handle_time_offset(self):
        """
        Handle time synchronization issues between client and server.
        This ensures the client is using the correct time delta for data requests.
        """
        try:
            logger.info("Checking time synchronization...")
            
            # Create a request without any specific contract, just to get the server time
            req = self.ib.reqCurrentTime()
            
            # Calculate the time difference
            local_time = datetime.now()
            server_time = datetime.fromtimestamp(req)
            time_diff = (server_time - local_time).total_seconds()
            
            logger.info(f"Time difference between local and server: {time_diff:.2f} seconds")
            
            # Store the time difference for future reference
            self.time_offset = time_diff
            
            return time_diff
        except Exception as e:
            logger.error(f"Error checking time synchronization: {str(e)}")
            return 0 

    def sync_to_streamlit(self):
        """
        Synchronize data from background thread to session_state.
        Call this from the main Streamlit thread.
        """
        try:
            # Safely copy any pending updates to session_state
            with self.lock:
                if self.pending_updates:
                    for key, value in self.ticker_data_updates.items():
                        st.session_state[key] = value
                    
                    # If we have market data type info, update session state
                    if 'market_data_type' in self.ticker_data_updates:
                        mdt = self.ticker_data_updates['market_data_type']
                        st.session_state.market_data_type = mdt
                    
                    # Clear the updates after syncing
                    self.ticker_data_updates = {}
                    self.pending_updates = False
                    
                    # Flag for UI refresh
                    st.session_state.last_ui_update = time.time()
                    st.session_state.pending_update = True
                    logger.info("Synced ticker data updates to Streamlit session_state")
                
            # Process any results from background tasks
            try:
                results = get_background_results()
                if results:
                    for task_id, result, error in results:
                        if error:
                            logger.error(f"Background task {task_id} failed: {error}")
                        else:
                            logger.debug(f"Background task {task_id} completed successfully")
            except Exception as e:
                logger.error(f"Error processing background results: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in sync_to_streamlit: {str(e)}")
            import traceback
            traceback.print_exc() 
            
def create_client(host='127.0.0.1', port=4001, client_id=None, timeout=20):
    """
    Factory function to create an enhanced IBKR client
    
    Args:
        host (str): Host address
        port (int): Port number (default is 4001 for IB Gateway live)
        client_id (int): Client ID (uses random ID if None)
        timeout (int): Connection timeout
        
    Returns:
        IBKRClientEnhanced: Configured client instance
    """
    client = IBKRClientEnhanced(host, port, client_id, timeout)
    return client
    
# Example usage:
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create client with random client ID (to prevent conflicts)
    client = create_client(client_id=None)
    
    # Connect to IB Gateway
    if client.connect():
        print("Connected to IB Gateway successfully")
        
        # Stream SPY data
        ticker = client.stream_data("SPY")
        
        # Process updates for 30 seconds
        for i in range(30):
            client.ib.sleep(1)
            print(f"SPY: last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
            
        # Disconnect
        client.disconnect()
    else:
        print("Failed to connect to IB Gateway") 