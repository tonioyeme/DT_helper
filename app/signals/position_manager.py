import pandas as pd
import pytz
from datetime import datetime
import logging
import numpy as np

# Try to import streamlit conditionally
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# SPY-specific configuration for sequential trading
SPY_CONFIG = {
    'min_reentry_gap': '2s',     # Matches NYSE latency
    'liquidity_threshold': 100000,  # Shares/minute
    'max_queue_size': 3,         # Prevent signal overload
    'emergency_clear_time': '15:59',  # Pre-market close flush (3:59 PM ET)
    'slippage_factor': 0.0005    # 0.05% slippage estimate
}

class Position:
    """
    Represents a trading position with direction and metadata
    """
    def __init__(self, direction, entry_price, entry_time, size=1.0, metadata=None):
        self.direction = direction  # 'buy' or 'sell'
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size = size
        self.metadata = metadata or {}
    
    def __str__(self):
        return f"{self.direction.upper()} @ {self.entry_price:.2f} ({self.entry_time})"

class SequentialPositionManager:
    """
    Enhanced position manager that ensures proper sequencing of trades
    with SPY-specific optimizations for execution quality
    """
    def __init__(self):
        self.current_position = None
        self.pending_exit = False  # Track exit confirmation
        self.signal_queue = []  # Buffer for incoming signals
        self.exit_time = None  # Track when exit was initiated
        self.last_exit_time = None  # Track when last exit was confirmed
        self.min_reentry_gap = pd.Timedelta(SPY_CONFIG['min_reentry_gap'])
        self.max_queue_size = SPY_CONFIG['max_queue_size']
        self.logger = logging.getLogger("SequentialPositionManager")
        
        # Configure logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def process_signal(self, new_signal, data=None, current_price=None, timestamp=None):
        """
        Process a new trading signal with enhanced SPY-specific timing
        
        Args:
            new_signal (str): The signal direction ('buy', 'sell', or 'neutral')
            data (pd.DataFrame): OHLCV data with market information
            current_price (float): Current price (optional)
            timestamp (datetime): Signal timestamp (optional)
            
        Returns:
            dict: Signal processing result including whether the signal was accepted and reasons
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now(tz=pytz.timezone('US/Eastern'))
        
        # Extract current price from data if available
        if current_price is None and data is not None and len(data) > 0:
            current_price = data['close'].iloc[-1]
        elif current_price is None:
            self.logger.warning("No price provided for signal processing")
            return {"accepted": False, "reason": "missing_price"}
            
        self.logger.info(f"Processing signal: {new_signal} at {current_price}")
        
        # Initialize result
        result = {
            "accepted": False,
            "reason": "",
            "signal": new_signal,
            "position_before": str(self.current_position) if self.current_position else None,
            "pending_exit_before": self.pending_exit,
            "queue_length_before": len(self.signal_queue)
        }
        
        # Check for emergency clear time (close all positions before market close)
        if self._is_emergency_clear_time(timestamp) and self.current_position is not None:
            self.logger.warning("Emergency clear time reached, forcing position exit")
            self._force_exit(current_price, timestamp)
            result["accepted"] = False
            result["reason"] = "emergency_clear_time"
            return result
        
        # STRICT ORDERING: If an opposite signal comes in but we haven't exited yet, force exit
        if self.current_position and new_signal != 'neutral' and new_signal != self.current_position.direction:
            if not self.pending_exit:
                # Force exit first (creates a neutral state) before processing new entry
                self._queue_exit(new_signal, current_price, timestamp, data)
                result["accepted"] = False
                result["reason"] = "forced_exit_first" 
                return result
            else:
                # Exit already pending, just queue the signal
                self.logger.info(f"Exit already pending, queueing {new_signal}")
                if len(self.signal_queue) < self.max_queue_size:
                    self.signal_queue.append((new_signal, current_price, timestamp))
                result["accepted"] = False
                result["reason"] = "exit_already_pending"
                return result
        
        # Prevent immediate re-entry after exit
        if self.last_exit_time and (timestamp - self.last_exit_time) < self.get_reentry_gap(data):
            self.logger.info(f"Preventing immediate re-entry, min gap: {self.get_reentry_gap(data)}")
            # If the signal is important, queue it instead of dropping
            if new_signal in ['buy', 'sell'] and len(self.signal_queue) < self.max_queue_size:
                self.signal_queue.append((new_signal, current_price, timestamp))
            result["accepted"] = False
            result["reason"] = "reentry_gap_enforced"
            return result
        
        # If we have no position, verify liquidity before entering
        if self.current_position is None:
            if new_signal in ['buy', 'sell']:
                if data is not None and not self._verify_spy_liquidity(data):
                    self.logger.warning("Delaying entry due to low liquidity")
                    if len(self.signal_queue) < self.max_queue_size:
                        self.signal_queue.append((new_signal, current_price, timestamp))
                    result["accepted"] = False
                    result["reason"] = "insufficient_liquidity"
                    return result
                
                # Execute entry - this is an accepted signal
                self._execute_entry(new_signal, current_price, timestamp, data)
                result["accepted"] = True
                result["reason"] = "new_position_opened"
                return result
            
            # Neutral signal with no position is ignored
            result["accepted"] = False
            result["reason"] = "neutral_with_no_position"
            return result
            
        # If signal matches current position, maintain position
        if new_signal == self.current_position.direction:
            self.logger.info(f"Signal matches current position: {new_signal}")
            result["accepted"] = True
            result["reason"] = "reinforced_existing_position"
            return result
            
        # If signal differs from current position
        if new_signal != self.current_position.direction:
            if not self.pending_exit:
                # Queue exit before allowing entry of opposite position
                self._queue_exit(new_signal, current_price, timestamp, data)
                result["accepted"] = True
                result["reason"] = "exit_queued"
                return result
            else:
                # If we're already pending an exit, just queue the new signal if not at limit
                self.logger.info(f"Exit already pending, queueing {new_signal}")
                if len(self.signal_queue) < self.max_queue_size:
                    self.signal_queue.append((new_signal, current_price, timestamp))
                result["accepted"] = False
                result["reason"] = "exit_already_pending"
                return result
        
        # Default case (should not reach here)
        result["accepted"] = False
        result["reason"] = "unhandled_case"
        return result
    
    def _queue_exit(self, next_signal, current_price, timestamp, data=None):
        """
        Queue an exit before entering a new position with intelligent exit calculation
        
        Args:
            next_signal (str): The next signal to execute after exit
            current_price (float): Current price
            timestamp (datetime): Signal timestamp
            data (pd.DataFrame): Market data for intelligent exit calculation
        """
        self.logger.info(f"Queuing exit for {self.current_position.direction} before {next_signal} entry")
        self.pending_exit = True
        self.exit_time = timestamp
        
        # Calculate intelligent exit price
        exit_price = self.calculate_intelligent_exit(current_price, data)
        
        # Submit exit order
        self._submit_exit_order(exit_price, timestamp, data)
        
        # Store next signal (only if not neutral)
        if next_signal != 'neutral':
            self.signal_queue.append((next_signal, current_price, timestamp))
    
    def _submit_exit_order(self, exit_price, timestamp, data=None):
        """
        Submit an exit order to close the current position
        
        Args:
            exit_price (float): Calculated exit price
            timestamp (datetime): Signal timestamp
            data (pd.DataFrame): Market data for exit optimization
        """
        self.logger.info(f"Submitting exit order for {self.current_position.direction} at {exit_price}")
        
        # In a real system, you would submit an order here with the optimized exit price
        # For now, we'll simulate it and immediately confirm
        self._on_exit_confirmation(exit_price, timestamp)
    
    def _on_exit_confirmation(self, exit_price, timestamp):
        """
        Handle exit confirmation and proceed to next signal
        
        Args:
            exit_price (float): Exit price
            timestamp (datetime): Exit timestamp
        """
        direction = self.current_position.direction
        self.logger.info(f"Exit confirmed for {direction} at {exit_price}")
        
        # Reset position
        self.current_position = None
        self.pending_exit = False
        self.last_exit_time = timestamp
        
        # Process next queued signal if available
        if self.signal_queue:
            next_signal, next_price, next_time = self.signal_queue.pop(0)
            self.logger.info(f"Processing queued signal: {next_signal}")
            self._execute_entry(next_signal, next_price, next_time)
    
    def _execute_entry(self, signal, current_price, timestamp, data=None):
        """
        Execute an entry for a new position using VWAP for better execution
        
        Args:
            signal (str): Signal direction ('buy' or 'sell')
            current_price (float): Entry price
            timestamp (datetime): Entry timestamp
            data (pd.DataFrame): Market data for entry optimization
        """
        if signal not in ['buy', 'sell']:
            self.logger.info(f"Ignoring neutral signal for entry")
            return
            
        # Calculate optimized entry price using VWAP if available
        entry_price = self.calculate_vwap_entry(current_price, data)
        
        self.logger.info(f"Executing {signal} entry at {entry_price}")
        
        # Create new position
        self.current_position = Position(
            direction=signal,
            entry_price=entry_price,
            entry_time=timestamp
        )
    
    def _force_exit(self, current_price, timestamp):
        """
        Force exit all positions (used for emergency closing)
        
        Args:
            current_price (float): Current price
            timestamp (datetime): Current timestamp
        """
        if self.current_position is None:
            return
            
        self.logger.warning(f"Forcing exit of {self.current_position.direction} position")
        self.pending_exit = True
        self._submit_exit_order(current_price, timestamp)
        self.signal_queue = []  # Clear signal queue
    
    def calculate_intelligent_exit(self, current_price, data=None):
        """
        Calculate optimal exit price based on position direction and market conditions
        
        Args:
            current_price (float): Current market price
            data (pd.DataFrame): Market data with bid/ask if available
            
        Returns:
            float: Optimized exit price
        """
        # Get bid/ask if available (in real implementation, use order book data)
        bid, ask = self._get_current_spread(current_price, data)
        
        # Default slippage factor
        slippage = current_price * SPY_CONFIG['slippage_factor']
        
        # Calculate dynamic exit price based on position direction
        if self.current_position.direction == 'buy':  # Exiting a long position
            return max(
                bid,  # Prioritize bid price if available
                current_price - slippage  # Default to small slippage
            )
        else:  # Exiting a short position
            return min(
                ask,  # Prioritize ask price if available
                current_price + slippage  # Default to small slippage
            )
    
    def calculate_vwap_entry(self, current_price, data=None):
        """
        Calculate VWAP-based entry price for better execution
        
        Args:
            current_price (float): Current market price
            data (pd.DataFrame): Market data with volume information
            
        Returns:
            float: VWAP-adjusted entry price
        """
        # Calculate VWAP if data is available
        if data is not None and len(data) > 20 and 'volume' in data.columns:
            try:
                # Simple VWAP calculation over last 20 bars
                vwap_window = data.iloc[-20:]
                vwap = (vwap_window['close'] * vwap_window['volume']).sum() / vwap_window['volume'].sum()
                
                # Use VWAP as reference, but don't deviate too far from current price
                max_adjustment = current_price * 0.001  # Max 0.1% adjustment
                
                if abs(vwap - current_price) < max_adjustment:
                    return vwap
            except Exception as e:
                self.logger.warning(f"Error calculating VWAP: {str(e)}")
        
        # Default to current price if VWAP calculation fails
        return current_price
    
    def _get_current_spread(self, current_price, data=None):
        """
        Get current bid/ask spread, either from data or estimated
        
        Args:
            current_price (float): Current market price
            data (pd.DataFrame): Market data that might contain bid/ask
            
        Returns:
            tuple: (bid_price, ask_price)
        """
        # In a real implementation, this would get actual bid/ask from order book
        # For now, estimate based on typical SPY spread
        estimated_spread = current_price * 0.0001  # Typical SPY spread is ~0.01%
        
        bid = current_price - (estimated_spread / 2)
        ask = current_price + (estimated_spread / 2)
        
        return bid, ask
    
    def _verify_spy_liquidity(self, data):
        """
        Verify that current SPY market has sufficient liquidity for entry
        
        Args:
            data (pd.DataFrame): Market data with volume
            
        Returns:
            bool: True if liquidity is sufficient
        """
        if data is None or 'volume' not in data.columns or len(data) < 5:
            return True  # Default to true if we can't verify
            
        # Check recent volume against threshold
        recent_volume = data['volume'].iloc[-5:].mean()
        
        return recent_volume >= SPY_CONFIG['liquidity_threshold']
    
    def _is_emergency_clear_time(self, timestamp):
        """
        Check if current time is near market close (emergency clear time)
        
        Args:
            timestamp (datetime): Current timestamp
            
        Returns:
            bool: True if it's emergency clear time
        """
        if not isinstance(timestamp, pd.Timestamp):
            return False
            
        # Get time as HH:MM string
        time_str = timestamp.strftime('%H:%M')
        emergency_time = SPY_CONFIG['emergency_clear_time']
        
        return time_str >= emergency_time
    
    def get_reentry_gap(self, data=None):
        """
        Calculate dynamic reentry gap based on volatility (VIX)
        
        Args:
            data (pd.DataFrame): Market data for volatility calculation
            
        Returns:
            pd.Timedelta: Minimum time between exit and next entry
        """
        # Get VIX level, use adaptive calculation if data available
        vix = self._calculate_implied_volatility(data) if data is not None else get_vix_level()
        
        # Adjust timing based on volatility
        if vix > 25:
            return pd.Timedelta('1s')  # Faster re-entry in high volatility
        elif vix > 15:
            return pd.Timedelta('2s')  # Default
        else:
            return pd.Timedelta('3s')  # Slower in low volatility
    
    def _calculate_implied_volatility(self, data):
        """
        Estimate implied volatility from price data when VIX is not available
        
        Args:
            data (pd.DataFrame): Recent price data
            
        Returns:
            float: Estimated implied volatility (VIX equivalent)
        """
        try:
            # Use recent 20 bars to calculate realized volatility
            if data is not None and len(data) >= 20:
                # Calculate log returns
                log_returns = np.log(data['close'] / data['close'].shift(1)).iloc[-20:]
                
                # Calculate annualized volatility (standard deviation of returns * sqrt(252) * 100)
                # 252 trading days per year, *100 to convert to percentage
                realized_vol = log_returns.std() * np.sqrt(252) * 100
                
                # Add a small premium to convert realized vol to implied vol
                return realized_vol * 1.1  # Typical implied vol premium
        except Exception as e:
            self.logger.warning(f"Error calculating implied volatility: {str(e)}")
            
        # Default to medium volatility if calculation fails
        return 20.0
    
    def get_current_position(self):
        """
        Get the current position
        
        Returns:
            Position: The current position or None
        """
        return self.current_position
    
    def is_pending_exit(self):
        """
        Check if an exit is pending
        
        Returns:
            bool: True if an exit is pending
        """
        return self.pending_exit
    
    def has_queued_signals(self):
        """
        Check if there are signals in the queue
        
        Returns:
            bool: True if signals are queued
        """
        return len(self.signal_queue) > 0
    
    def get_trading_state(self):
        """
        Get the current trading state for monitoring
        
        Returns:
            dict: Current trading state information
        """
        return {
            'current_position': str(self.current_position) if self.current_position else None,
            'pending_exit': self.pending_exit,
            'queued_signals': len(self.signal_queue),
            'last_exit_time': self.last_exit_time,
            'min_reentry_gap': str(self.get_reentry_gap())
        }

def calculate_exit_timing(vix):
    """
    Calculate dynamic exit timing based on VIX levels
    
    Args:
        vix (float): Current VIX value
        
    Returns:
        pd.Timedelta: Time to wait before executing next order
    """
    if vix > 30:
        return pd.Timedelta('1s')  # Faster exits in high volatility
    elif vix > 20:
        return pd.Timedelta('3s')
    else:
        return pd.Timedelta('5s')

def get_vix_level():
    """
    Get current VIX level from Streamlit session if available
    
    Returns:
        float: VIX level (default 20 if not available)
    """
    if STREAMLIT_AVAILABLE:
        return getattr(st.session_state, 'vix_level', 20.0)
    return 20.0  # Default mid-range value

# Create a singleton position manager
_position_manager = None

def get_position_manager(reset=False):
    """
    Get or create the singleton position manager
    
    Args:
        reset (bool): If True, reset the position manager to a clean state
        
    Returns:
        SequentialPositionManager: The position manager instance
    """
    global _position_manager
    if _position_manager is None or reset:
        _position_manager = SequentialPositionManager()
    return _position_manager 