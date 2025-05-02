"""
IBKR Market Data Quick Fix

A simple wrapper to fix NaN market data issues with Interactive Brokers API.
This handles:
1. Automatic switching between LIVE and DELAYED market data based on market hours
2. Random client IDs to prevent connection conflicts
3. Specific generic tick list for streaming bundle

Usage:
    from ibkr_quick_fix import fix_ibkr_connection, get_market_data

    # Apply the fix when connecting
    ib = fix_ibkr_connection(port=4001)
    
    # Get market data with proper tick list
    ticker = get_market_data(ib, "SPY")
"""

from ib_insync import *
import random
import datetime
import pytz
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_market_open():
    """Check if US markets are currently open"""
    # Get current time in Eastern timezone (US markets)
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
        
    # Check regular market hours (9:30 AM - 4:00 PM ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def fix_ibkr_connection(host='127.0.0.1', port=4001, client_id=None, timeout=20):
    """
    Connect to IBKR with fixes for common issues
    
    Args:
        host (str): IBKR TWS/Gateway host address
        port (int): IBKR TWS/Gateway port (4001 for IB Gateway live)
        client_id (int): Client ID for IB connection (random if None)
        timeout (int): Timeout in seconds for connection attempts
        
    Returns:
        IB: Connected IB instance
    """
    # Patch asyncio for better behavior
    util.patchAsyncio()
    
    # Create IB instance
    ib = IB()
    
    # Use a random client ID if none provided to avoid conflicts
    if client_id is None:
        client_id = random.randint(1000, 9999)
        logger.info(f"Using random client ID: {client_id}")
    
    # Connect to IBKR API
    try:
        logger.info(f"Connecting to IBKR at {host}:{port} (client ID: {client_id})")
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        logger.info(f"Connected successfully, isConnected={ib.isConnected()}")
        
        # Set appropriate market data type based on market hours
        select_market_data_type(ib)
        
    except Exception as e:
        logger.error(f"Error connecting to IBKR: {str(e)}")
        raise e
    
    return ib

def select_market_data_type(ib):
    """
    Select the appropriate market data type based on market hours
    - Use LIVE (1) during market hours
    - Use DELAYED (3) outside market hours
    
    Args:
        ib (IB): Connected IB instance
    """
    try:
        if is_market_open():
            logger.info("Markets are OPEN - requesting LIVE market data")
            ib.reqMarketDataType(1)
        else:
            logger.info("Markets are CLOSED - requesting DELAYED market data")
            ib.reqMarketDataType(3)
    except Exception as e:
        logger.warning(f"Error setting market data type: {str(e)}")
        # Fall back to delayed data
        logger.info("Falling back to DELAYED market data")
        try:
            ib.reqMarketDataType(3)
        except Exception as e2:
            logger.error(f"Error requesting DELAYED market data: {str(e2)}")

def get_market_data(ib, symbol, exchange='SMART', currency='USD'):
    """
    Get market data with the correct generic tick list for streaming bundle
    
    Args:
        ib (IB): Connected IB instance
        symbol (str): Trading symbol
        exchange (str): Exchange name
        currency (str): Currency
        
    Returns:
        Ticker: Ticker object with real-time data
    """
    # Create contract definition
    contract = Stock(symbol, exchange, currency)
    logger.info(f"Created contract {contract}")
    
    # Request market data with specific generic tick types that use Streaming Bundle
    generic_tick_list = "100,101,104,106,165"
    
    # Request market data
    ticker = ib.reqMktData(contract, genericTickList=generic_tick_list, snapshot=False, regulatorySnapshot=False)
    logger.info(f"Requested market data for {symbol}")
    
    return ticker

def switch_market_data_type(ib):
    """
    Toggle between LIVE and DELAYED market data types
    
    Args:
        ib (IB): Connected IB instance
    """
    try:
        # Try to get current type (may not be exposed in all versions)
        wrapper = getattr(ib, 'wrapper', None)
        current_type = 1  # Default to LIVE
        
        if wrapper and hasattr(wrapper, 'marketDataType'):
            # If we can access the current type
            current_type = wrapper.marketDataType.get(0, 1)
            
        # Toggle between LIVE and DELAYED
        new_type = 3 if current_type == 1 else 1
        
        logger.info(f"Switching market data type from {current_type} to {new_type}")
        ib.reqMarketDataType(new_type)
        
    except Exception as e:
        logger.error(f"Error switching market data type: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        # Connect with the fixes
        ib = fix_ibkr_connection(port=4001)
        
        # Show market status
        market_status = "OPEN" if is_market_open() else "CLOSED"
        print(f"Market status: {market_status}")
        
        # Get SPY data with correct tick list
        ticker = get_market_data(ib, "SPY")
        
        # Wait for data to arrive
        for i in range(5):
            ib.sleep(1)
            print(f"SPY (after {i+1}s): last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
            if ticker.last or ticker.bid or ticker.ask:
                break
        
        # If we didn't get data, try switching market data type
        if not (ticker.last or ticker.bid or ticker.ask):
            print("No data received, trying different market data type")
            switch_market_data_type(ib)
            
            # Wait a bit more
            for i in range(3):
                ib.sleep(1)
                print(f"After switch {i+1}s: last={ticker.last}, bid={ticker.bid}, ask={ticker.ask}")
        
        # Disconnect
        ib.disconnect()
        print("Disconnected from IB")
        
    except Exception as e:
        print(f"Error in example: {str(e)}") 