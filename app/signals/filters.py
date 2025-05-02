import pandas as pd
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional
from datetime import timedelta

from app.indicators import calculate_atr

def calculate_enhanced_atr(df: pd.DataFrame, period: int = 14, smoothing: str = 'ema') -> pd.Series:
    """
    Calculate enhanced ATR with smoothing options for high-frequency trading
    
    Args:
        df: DataFrame with OHLC price data
        period: Look-back period for ATR calculation (default now 14, was 20)
        smoothing: Smoothing method ('rma' for Wilder's, 'ema' for exponential)
        
    Returns:
        Series with ATR values
    """
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    # Calculate the three components of True Range
    high_low = high - low  # Current high minus current low
    high_prev_close = (high - close_prev).abs()  # Current high minus previous close
    low_prev_close = (low - close_prev).abs()  # Current low minus previous close
    
    # True Range is the maximum of these three values
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Apply appropriate smoothing method - default to EMA for better noise rejection
    if smoothing == 'rma':  # Wilder's smoothing
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
    elif smoothing == 'ema':  # EMA smoothing (preferred for HFT)
        atr = tr.ewm(span=period, adjust=False).mean()
    else:  # Default to simple moving average
        atr = tr.rolling(period).mean()
    
    return atr

def atr_signal_filter(
    current_signal: str,
    previous_signal: str,
    current_price: float,
    previous_price: float,
    cumulative_move: float,
    atr_value: float,
    k: float = 1.2,  # Increased to require 1.2×ATR movement
    spread: float = 0.01  # Typical SPY spread
) -> Tuple[str, float]:
    """
    Modified filter to only remove signals that don't allow profit after costs
    
    Args:
        current_signal: Current raw trading signal ('buy', 'sell', or 'neutral')
        previous_signal: Previous filtered signal
        current_price: Current price
        previous_price: Previous price when the signal changed
        cumulative_move: Cumulative price movement since last signal change
        atr_value: ATR value
        k: ATR multiplier (higher = less sensitive, requires larger moves)
        spread: Average spread for the instrument
        
    Returns:
        tuple: (filtered_signal, new_cumulative_move)
    """
    # Default to keeping the previous signal
    filtered_signal = previous_signal
    
    # Validate ATR value and use fallback if needed
    if atr_value is None or pd.isna(atr_value) or atr_value <= 0:
        atr_value = abs(current_price) * 0.01  # Use 1% as fallback
    
    # Calculate required movement including spread costs
    required_move = (k * atr_value) + (spread * 2)  # Full spread compensation
    
    # Always allow first signal (when coming from neutral)
    if previous_signal == 'neutral':
        return current_signal, 0.0
    
    # Handle neutral signals specially
    if current_signal == 'neutral':
        # If we're exiting a position (previous signal was not neutral), use reduced threshold
        if previous_signal != 'neutral':
            exit_threshold = 0.3 * (k * atr_value)  # 30% of entry threshold
            if cumulative_move >= exit_threshold:
                return 'neutral', 0.0
        return previous_signal, cumulative_move
    
    # Calculate effective price movement accounting for spread
    effective_move = abs(current_price - previous_price) - spread
    
    # STRICT ENFORCEMENT: No conflicting entry until previous position is exited
    # If current signal conflicts with previous and previous is not neutral, force neutral/exit first
    if previous_signal != 'neutral' and current_signal != previous_signal and current_signal != 'neutral':
        # Check if we've moved far enough to exit the previous position
        if effective_move >= required_move:
            # First change to neutral (exit) before allowing opposite entry
            return 'neutral', 0.0
        else:
            # Return previous signal with updated cumulative move - no entry until exit complete
            return previous_signal, cumulative_move + effective_move
    
    # Normal ATR-based filtering logic for non-conflicting signals
    if current_signal != previous_signal:
        if effective_move >= required_move:
            # Signal change confirmed, reset cumulative move
            filtered_signal = current_signal
            cumulative_move = 0.0
    else:
        # Same signal direction, add to cumulative move
        filtered_signal = current_signal
    
    return filtered_signal, cumulative_move

def effective_move_calculation(last_price, new_price, bid=None, ask=None):
    """
    Calculate effective price movement accounting for spread
    
    Args:
        last_price (float): Previous price
        new_price (float): Current price
        bid (float): Current bid price (optional)
        ask (float): Current ask price (optional)
        
    Returns:
        float: Effective price movement
    """
    # Calculate raw movement
    raw_move = abs(new_price - last_price)
    
    # If bid/ask are provided, account for spread
    if bid is not None and ask is not None:
        spread = ask - bid
        effective_move = raw_move - spread
        return max(effective_move, 0)  # Never return negative
    
    # Otherwise use standard SPY spread estimate
    estimated_spread = last_price * 0.0001  # Approx 0.01% spread for SPY
    effective_move = raw_move - estimated_spread
    return max(effective_move, 0)

def apply_atr_filter_to_signals(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    atr_period: int = 14,  # Changed from 20 to 14 for better responsiveness
    k: float = 1.2,  # Increased from 0.82 to 1.2 to require larger moves
    smoothing: str = 'ema',
    min_holding_period: pd.Timedelta = pd.Timedelta('15min'),  # Increased from 10min to 15min
    spread: float = 0.01,
    min_bar_distance: int = 3  # Minimum number of bars between signals
) -> pd.DataFrame:
    """
    Apply ATR-based filtering to an entire DataFrame of signals with improved filtering
    
    Args:
        data: DataFrame with price data
        signals: DataFrame with trading signals
        atr_period: Period for ATR calculation
        k: Multiplier for ATR threshold
        smoothing: ATR smoothing method
        min_holding_period: Minimum time to hold a position
        spread: Typical bid-ask spread
        min_bar_distance: Minimum number of bars between signals
        
    Returns:
        DataFrame with filtered signals
    """
    try:
        # Import SciPy for peak detection if available
        from scipy import signal as scipy_signal
    except ImportError:
        print("SciPy not available - profit potential check will be disabled")
        has_scipy = False
    else:
        has_scipy = True
    
    # Make a copy of the signals to avoid modifying the original
    filtered_signals = signals.copy()
    
    # Calculate ATR if not already present
    if 'atr' not in data.columns:
        data['atr'] = calculate_enhanced_atr(data, period=atr_period, smoothing=smoothing)
    
    # Initialize filtered signal columns if they don't exist
    if 'filtered_buy_signal' not in filtered_signals.columns:
        filtered_signals['filtered_buy_signal'] = False
    
    if 'filtered_sell_signal' not in filtered_signals.columns:
        filtered_signals['filtered_sell_signal'] = False
    
    # Initialize columns for tracking movement
    filtered_signals['cumulative_move'] = 0.0
    filtered_signals['required_move'] = 0.0  # Will be calculated per row
    filtered_signals['cum_debug'] = 0.0  # Add debugging column
    filtered_signals['filter_reason'] = None  # Add a column to track filter reasons
    
    # Initialize variables to track state
    last_signal = 'neutral'
    last_price = None  # Initialize as None instead of first close price
    last_signal_time = None
    cumulative_move = 0.0  # Track cumulative price movement
    signal_count = 0  # Track number of signals
    bar_count = 0  # Track bar count since last signal
    
    # Helper function to check for profit potential
    def has_profit_potential(prices, min_height=0.0015):  # 0.15% minimum swing
        if not has_scipy or len(prices) < 5:
            return True  # Default to True if SciPy not available or not enough data
        
        # Normalize prices to percentage changes
        normalized = (prices - prices.iloc[0]) / prices.iloc[0]
        
        # Find peaks and valleys
        peaks, _ = scipy_signal.find_peaks(normalized, height=min_height)
        valleys, _ = scipy_signal.find_peaks(-normalized, height=min_height)
        
        return len(peaks) > 0 or len(valleys) > 0
    
    # Process each row (timestamp) in the signals DataFrame
    for idx in signals.index:
        if idx not in data.index:
            continue  # Skip if timestamp not in price data
            
        current_price = data.loc[idx, 'close']
        current_atr = data.loc[idx, 'atr']
        
        # Add validation checks for ATR
        if current_atr <= 0 or pd.isna(current_atr):
            current_atr = data['atr'].mean()  # Fallback to average ATR
            
        # Calculate required move with minimum threshold
        required_move = max(
            (k * current_atr) + (spread * 2),  # Full spread compensation
            0.002 * current_price  # Minimum 0.2% of price (doubled from original)
        )
        
        filtered_signals.loc[idx, 'required_move'] = required_move
        
        # Determine raw signal
        if signals.loc[idx, 'buy_signal']:
            raw_signal = 'buy'
        elif signals.loc[idx, 'sell_signal']:
            raw_signal = 'sell'
        else:
            raw_signal = 'neutral'
        
        # First time setup - FIXED to allow first valid signal
        if last_price is None:
            last_price = current_price
            if raw_signal != 'neutral':  # Accept first non-neutral signal
                filtered_signals.loc[idx, f'filtered_{raw_signal}_signal'] = True
                filtered_signals.loc[idx, 'filter_reason'] = 'first_signal'
                last_signal = raw_signal
                last_signal_time = idx
                signal_count += 1
                bar_count = 0
            continue
        
        # Increment bar count
        bar_count += 1
        
        # FIXED: Track movement from last signal price instead of consecutive closes
        if last_signal != 'neutral':
            price_delta = abs(current_price - last_price)
            cumulative_move += price_delta
        
        # Store cumulative move for debugging
        filtered_signals.loc[idx, 'cumulative_move'] = cumulative_move
        filtered_signals.loc[idx, 'cum_debug'] = cumulative_move
        
        # Skip further processing if neutral signal and we're not in a position
        if raw_signal == 'neutral' and last_signal == 'neutral':
            continue
        
        # Apply cooldown period check - require min_bar_distance bars between signals
        if last_signal_time and raw_signal != 'neutral' and raw_signal != last_signal:
            if bar_count < min_bar_distance:
                filtered_signals.loc[idx, 'filter_reason'] = f'cooldown_period_{bar_count}/{min_bar_distance}'
                continue
        
        # Apply profit potential check for new signals
        if raw_signal != 'neutral' and raw_signal != last_signal:
            # Get recent price window (3x ATR period bars)
            lookback = min(len(data) - 1, atr_period * 3)
            if lookback > 5:  # Need at least a few bars for analysis
                end_idx = data.index.get_loc(idx)
                start_idx = max(0, end_idx - lookback)
                recent_prices = data.iloc[start_idx:end_idx+1]['close']
                
                if not has_profit_potential(recent_prices):
                    filtered_signals.loc[idx, 'filter_reason'] = 'no_profit_potential'
                    continue
        
        # Apply time-based filter - only for same-direction signals
        # Allow opposite signals to execute immediately
        if (last_signal_time and 
            (idx - last_signal_time) < min_holding_period and 
            raw_signal == last_signal):
            filtered_signal = last_signal
            filtered_signals.loc[idx, 'filter_reason'] = 'min_holding_period'
            # We're not changing the signal, just continue
        else:
            # Apply ATR filter
            filtered_signal, new_cumulative_move = atr_signal_filter(
                current_signal=raw_signal,
                previous_signal=last_signal,
                current_price=current_price,
                previous_price=last_price,
                cumulative_move=cumulative_move,
                atr_value=current_atr,
                k=k,
                spread=spread
            )
            
            # Only reset cumulative move if signal actually changed
            if filtered_signal != last_signal:
                cumulative_move = new_cumulative_move
                last_signal = filtered_signal
                last_price = current_price  # Update tracking price when signal changes
                last_signal_time = idx
                bar_count = 0  # Reset bar count
                
                if filtered_signal != 'neutral':
                    signal_count += 1
                    filtered_signals.loc[idx, 'filter_reason'] = 'passed_filter'
            else:
                filtered_signals.loc[idx, 'filter_reason'] = 'insufficient_movement'
        
        # Update filtered signal columns
        filtered_signals.loc[idx, 'filtered_buy_signal'] = (filtered_signal == 'buy')
        filtered_signals.loc[idx, 'filtered_sell_signal'] = (filtered_signal == 'sell')
    
    # Print filtering statistics
    print(f"ATR Filter Stats: {signal_count} signals passed filtering out of {len(filtered_signals[signals['buy_signal'] | signals['sell_signal']])}")
    
    return filtered_signals

def get_spy_atr_filter_config() -> Dict[str, Any]:
    """
    Get recommended ATR filter configuration for SPY
    
    Returns:
        Dictionary with ATR filter configuration optimized for SPY
    """
    return {
        'atr_period': 14,       # More responsive to recent volatility
        'k': 1.2,               # Require 1.2×ATR movement for signal changes
        'smoothing': 'ema',     # Better noise rejection than RMA
        'min_holding_period': pd.Timedelta('22min'),  # Optimized for SPY 5-min bars
        'spread': 0.01,         # Typical SPY spread
        'min_bar_distance': 3,  # Minimum bars between signals (15 minutes for 5-min bars)
        'dynamic_target': True  # Enable adaptive targets
    } 