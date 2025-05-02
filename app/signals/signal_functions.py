import pandas as pd
import numpy as np
from enum import Enum
import pytz
from datetime import datetime, time, timedelta
import logging
import traceback
import warnings
from typing import Dict, List, Optional, Union, Any

# Import streamlit conditionally
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from indicators import (
    calculate_ema,
    calculate_ema_cloud,
    calculate_macd,
    calculate_vwap,
    calculate_obv,
    calculate_ad_line,
    calculate_rsi,
    calculate_stochastic,
    calculate_fibonacci_sma,
    calculate_pain,
    calculate_ema_vwap_strategy,
    calculate_measured_move_volume_strategy,
    multi_indicator_confirmation,
    calculate_roc, 
    calculate_hull_moving_average, 
    calculate_ttm_squeeze,
    detect_hidden_divergence,
    calculate_opening_range,
    detect_orb_breakout,
    analyze_session_data,
    calculate_adaptive_rsi,
    calculate_atr,
    calculate_adx
)

# Logging setup
logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength classifications"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

def is_market_hours(timestamp):
    """
    Check if the given timestamp is during market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
    
    Args:
        timestamp: Datetime object or index
        
    Returns:
        bool: True if timestamp is during market hours, False otherwise
    """
    # If timestamp has no tzinfo, assume it's UTC and convert
    if hasattr(timestamp, 'tzinfo'):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=pytz.UTC)
        # Convert to Eastern Time
        eastern = pytz.timezone('US/Eastern')
        timestamp = timestamp.astimezone(eastern)
    else:
        # If not a datetime, just return True (can't determine)
        return True
    
    # Check if it's a weekday (0 = Monday, 4 = Friday)
    if timestamp.weekday() > 4:  # Saturday or Sunday
        return False
    
    # Check if within 9:30 AM - 4:00 PM ET
    market_open = time(9, 30, 0)
    market_close = time(16, 0, 0)
    
    return market_open <= timestamp.time() <= market_close

def calculate_adaptive_rsi(data, length=14):
    """
    Calculate RSI with adaptive length based on volatility
    
    Args:
        data: DataFrame with price data
        length: Base period for RSI calculation
        
    Returns:
        Series: RSI values
    """
    # Calculate price changes
    delta = data['close'].diff()
    
    # Calculate volatility (standard deviation of returns)
    volatility = data['close'].pct_change().rolling(window=20).std()
    
    # Adjust RSI length based on volatility
    # Higher volatility = shorter RSI period
    adaptive_length = length
    if not volatility.empty:
        mean_vol = volatility.mean()
        if not np.isnan(mean_vol) and mean_vol > 0:
            # Adjust between 10 and 20 periods based on volatility
            adaptive_length = int(length * (1 - 0.5 * (volatility.iloc[-1] / mean_vol - 1)))
            adaptive_length = max(min(adaptive_length, 20), 10)
    
    # Calculate RSI with adaptive length
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = -loss  # Convert to positive
    
    avg_gain = gain.rolling(window=adaptive_length).mean()
    avg_loss = loss.rolling(window=adaptive_length).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def generate_standard_signals(data):
    """
    Generate trading signals for a given price dataset
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with signals
    """
    try:
        # Create a copy of the data
        signals = pd.DataFrame(index=data.index)
        signals['close'] = data['close']
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['buy_strength'] = 0.0
        signals['sell_strength'] = 0.0
        
        # Calculate indicators for signal generation
        # EMA Crossover (Short-term vs Medium-term)
        data['ema9'] = data['close'].ewm(span=9, adjust=False).mean()
        data['ema21'] = data['close'].ewm(span=21, adjust=False).mean()
        
        # RSI
        if 'rsi' not in data.columns:
            data['rsi'] = calculate_adaptive_rsi(data)
        
        # Simple volume based signals
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        
        # Generate EMA crossover signals
        for i in range(1, len(data)):
            # EMA Crossover
            if data['ema9'].iloc[i-1] < data['ema21'].iloc[i-1] and data['ema9'].iloc[i] > data['ema21'].iloc[i]:
                signals.iloc[i, signals.columns.get_indexer(['buy_signal'])[0]] = True
                signals.iloc[i, signals.columns.get_indexer(['buy_strength'])[0]] = SignalStrength.MODERATE.value
            
            elif data['ema9'].iloc[i-1] > data['ema21'].iloc[i-1] and data['ema9'].iloc[i] < data['ema21'].iloc[i]:
                signals.iloc[i, signals.columns.get_indexer(['sell_signal'])[0]] = True
                signals.iloc[i, signals.columns.get_indexer(['sell_strength'])[0]] = SignalStrength.MODERATE.value
                
            # Add RSI-based signals (oversold/overbought)
            if not np.isnan(data['rsi'].iloc[i]):
                if data['rsi'].iloc[i] < 30 and data['rsi'].iloc[i-1] < 30 and data['close'].iloc[i] > data['close'].iloc[i-1]:
                    signals.iloc[i, signals.columns.get_indexer(['buy_signal'])[0]] = True
                    signals.iloc[i, signals.columns.get_indexer(['buy_strength'])[0]] = max(
                        signals.iloc[i, signals.columns.get_indexer(['buy_strength'])[0]], 
                        SignalStrength.STRONG.value
                    )
                
                elif data['rsi'].iloc[i] > 70 and data['rsi'].iloc[i-1] > 70 and data['close'].iloc[i] < data['close'].iloc[i-1]:
                    signals.iloc[i, signals.columns.get_indexer(['sell_signal'])[0]] = True
                    signals.iloc[i, signals.columns.get_indexer(['sell_strength'])[0]] = max(
                        signals.iloc[i, signals.columns.get_indexer(['sell_strength'])[0]], 
                        SignalStrength.STRONG.value
                    )
            
            # Volume confirmation
            if not np.isnan(data['volume_ma'].iloc[i]) and data['volume'].iloc[i] > 1.5 * data['volume_ma'].iloc[i]:
                # Increase strength of existing signals on high volume
                if signals.iloc[i, signals.columns.get_indexer(['buy_signal'])[0]]:
                    signals.iloc[i, signals.columns.get_indexer(['buy_strength'])[0]] = min(
                        SignalStrength.VERY_STRONG.value,
                        signals.iloc[i, signals.columns.get_indexer(['buy_strength'])[0]] + 1
                    )
                
                if signals.iloc[i, signals.columns.get_indexer(['sell_signal'])[0]]:
                    signals.iloc[i, signals.columns.get_indexer(['sell_strength'])[0]] = min(
                        SignalStrength.VERY_STRONG.value,
                        signals.iloc[i, signals.columns.get_indexer(['sell_strength'])[0]] + 1
                    )
        
        return signals
    except Exception as e:
        print(f"Error generating standard signals: {str(e)}")
        traceback.print_exc()
        # Return an empty DataFrame with the same index as data
        return pd.DataFrame(index=data.index)

def calculate_adaptive_rsi(data, period=14, ma_type='ewm', ma_period=14):
    """
    Calculate Adaptive Relative Strength Index
    
    Args:
        data (pd.DataFrame): OHLCV data
        period (int): RSI period
        ma_type (str): Moving average type ('ewm', 'sma', 'wma')
        ma_period (int): Moving average period
        
    Returns:
        pd.Series: Adaptive RSI values
    """
    # Calculate standard RSI
    close = data['close']
    delta = close.diff()
    
    # Create gain/loss Series
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain/loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Apply moving average to RSI based on ma_type
    if ma_type == 'ewm':
        adaptive_rsi = rsi.ewm(span=ma_period, adjust=False).mean()
    elif ma_type == 'wma':
        weights = np.arange(1, ma_period + 1)
        adaptive_rsi = rsi.rolling(ma_period).apply(lambda x: np.sum(weights * x) / np.sum(weights))
    else:  # 'sma'
        adaptive_rsi = rsi.rolling(ma_period).mean()
    
    return adaptive_rsi

def generate_standard_signals(data):
    """
    Generate standard trading signals using technical indicators
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with signals
    """
    # Initialize signals DataFrame with necessary columns
    signals = pd.DataFrame(index=data.index)
    signals["buy_signal"] = False
    signals["sell_signal"] = False
    signals["buy_score"] = 0.0
    signals["sell_score"] = 0.0
    signals["buy_strength"] = 0
    signals["sell_strength"] = 0  # Added sell_strength column
    signals["strong_buy_signal"] = False
    signals["strong_sell_signal"] = False
    signals["signal_price"] = data["close"]
    signals["target_price"] = None
    signals["stop_loss"] = None
    
    # Convert timestamps to Eastern Time for display
    eastern = pytz.timezone('US/Eastern')
    signals["signal_time_et"] = [idx.tz_localize('UTC', ambiguous='raise').tz_convert(eastern).strftime('%Y-%m-%d %H:%M:%S') 
                               if hasattr(idx, 'tz_localize') and idx.tzinfo is None 
                               else (idx.astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'astimezone') else str(idx))
                               for idx in signals.index]
    
    # Create market hours mask for all data points
    market_hours_mask = [is_market_hours(idx) for idx in data.index]
    
    # Mark market status for all data points
    signals["market_status"] = ["Open" if mask else "Closed" for mask in market_hours_mask]
    
    # Create a market hours subset for signal calculation
    market_data = data.loc[market_hours_mask] if any(market_hours_mask) else data
    
    # === Calculate indicators for all data to ensure continuity ===
    fast_ema, slow_ema = calculate_ema_cloud(data, fast_period=5, slow_period=13)
    rsi = calculate_rsi(data, 14)
    adaptive_rsi = calculate_adaptive_rsi(data)
    stoch_k, stoch_d = calculate_stochastic(data)
    macd, macd_signal, macd_hist = calculate_macd(data)
    
    # === Calculate Opening Range Breakout (ORB) signals ===
    try:
        # Calculate opening range (first 5 minutes of the trading day)
        # Convert data index to Eastern Time for proper filtering
        market_data_copy = market_data.copy()
        if market_data_copy.index.tzinfo is None:
            # Handle timezone-naive index by first localizing to UTC
            market_data_copy.index = market_data_copy.index.tz_localize('UTC', ambiguous='raise')
            
        # Now convert to Eastern Time for ORB calculation
        if market_data_copy.index.tzinfo is not None:
            market_data_copy.index = market_data_copy.index.tz_convert(eastern)
        
        # Get the opening range if we have data from market open
        orb_high, orb_low = calculate_opening_range(market_data_copy, minutes=5)
        
        if orb_high is not None and orb_low is not None:
            # Detect breakout signals
            # Create opening range data dictionary for detect_orb_breakout
            orb_signals = detect_orb_breakout(market_data_copy, orb_high, orb_low)
            
            # Merge with signals DataFrame
            for idx in orb_signals.index:
                if idx in signals.index:
                    if orb_signals.loc[idx, 'orb_long']:
                        signals.loc[idx, 'buy_signal'] = True
                        signals.loc[idx, 'buy_score'] += 0.8  # Strong signal
                        # Add ORB-specific columns
                        signals.loc[idx, 'orb_signal'] = True
                        signals.loc[idx, 'orb_level'] = orb_high
                        
                    if orb_signals.loc[idx, 'orb_short']:
                        signals.loc[idx, 'sell_signal'] = True
                        signals.loc[idx, 'sell_score'] += 0.8  # Strong signal
                        # Add ORB-specific columns
                        signals.loc[idx, 'orb_signal'] = True
                        signals.loc[idx, 'orb_level'] = orb_low
    except Exception as e:
        print(f"Error calculating ORB signals: {str(e)}")
    
    # === Calculate EMA Cloud crossover signals (only for market hours) ===
    try:
        # Calculate crossover points
        signals['ema_cloud_cross_bullish'] = False
        signals['ema_cloud_cross_bearish'] = False
        
        # First point must have both EMAs available
        valid_indices = fast_ema.dropna().index.intersection(slow_ema.dropna().index)
        
        if len(valid_indices) >= 2:
            # Calculate for remaining points
            for i in range(1, len(valid_indices)):
                current_idx = valid_indices[i]
                prev_idx = valid_indices[i-1]
                
                # Only process market hours points
                if current_idx not in market_data.index:
                    continue
                
                # Bullish crossover: fast EMA crosses above slow EMA
                if fast_ema[prev_idx] <= slow_ema[prev_idx] and fast_ema[current_idx] > slow_ema[current_idx]:
                    signals.loc[current_idx, 'ema_cloud_cross_bullish'] = True
                    signals.loc[current_idx, 'buy_signal'] = True
                    signals.loc[current_idx, 'buy_score'] += 0.7  # Strong signal
                
                # Bearish crossover: fast EMA crosses below slow EMA
                if fast_ema[prev_idx] >= slow_ema[prev_idx] and fast_ema[current_idx] < slow_ema[current_idx]:
                    signals.loc[current_idx, 'ema_cloud_cross_bearish'] = True
                    signals.loc[current_idx, 'sell_signal'] = True
                    signals.loc[current_idx, 'sell_score'] += 0.7  # Strong signal
    except Exception as e:
        print(f"Error calculating EMA cloud signals: {str(e)}")
        
    # Calculate price trend signals (only for market hours)
    try:
        # Use a short SMA to determine trend direction (8-period)
        sma8 = data['close'].rolling(window=8).mean()
        sma20 = data['close'].rolling(window=20).mean()
        
        # Add trend signals only to market hours data points
        for idx in market_data.index:
            if idx in sma8.index and idx in sma20.index:
                # Uptrend: current close > SMAs
                uptrend = (data.loc[idx, 'close'] > sma8[idx]) and (sma8[idx] > sma20[idx])
                downtrend = (data.loc[idx, 'close'] < sma8[idx]) and (sma8[idx] < sma20[idx])
                
                if uptrend:
                    signals.loc[idx, 'buy_score'] += 0.2
                elif downtrend:
                    signals.loc[idx, 'sell_score'] += 0.2
    except Exception as e:
        print(f"Error calculating trend signals: {str(e)}")

    # Set signal strength based on buy/sell scores (only for market hours)
    for idx in market_data.index:
        # Set buy signals based on score thresholds
        if signals.at[idx, 'buy_score'] >= 0.8:
            signals.at[idx, 'buy_signal'] = True
            signals.at[idx, 'buy_strength'] = SignalStrength.VERY_STRONG.value
        elif signals.at[idx, 'buy_score'] >= 0.6:
            signals.at[idx, 'buy_signal'] = True
            signals.at[idx, 'buy_strength'] = SignalStrength.STRONG.value
        elif signals.at[idx, 'buy_score'] >= 0.4:
            signals.at[idx, 'buy_signal'] = True
            signals.at[idx, 'buy_strength'] = SignalStrength.MODERATE.value
        
        # Set sell signals based on score thresholds
        if signals.at[idx, 'sell_score'] >= 0.8:
            signals.at[idx, 'sell_signal'] = True
            signals.at[idx, 'sell_strength'] = SignalStrength.VERY_STRONG.value
        elif signals.at[idx, 'sell_score'] >= 0.6:
            signals.at[idx, 'sell_signal'] = True
            signals.at[idx, 'sell_strength'] = SignalStrength.STRONG.value
        elif signals.at[idx, 'sell_score'] >= 0.4:
            signals.at[idx, 'sell_signal'] = True
            signals.at[idx, 'sell_strength'] = SignalStrength.MODERATE.value
            
        # Mark strong signals
        if signals.at[idx, 'buy_strength'] >= SignalStrength.STRONG.value:
            signals.at[idx, 'strong_buy_signal'] = True
        
        if signals.at[idx, 'sell_strength'] >= SignalStrength.STRONG.value:
            signals.at[idx, 'strong_sell_signal'] = True
    
    # Apply ATR filter to reduce noise in signals
    # Get symbol if available from streamlit session
    symbol = None
    is_spy_mode = False
    
    if STREAMLIT_AVAILABLE:
        symbol = getattr(st.session_state, 'symbol', None)
        is_spy_mode = hasattr(st.session_state, 'is_spy_mode') and st.session_state.is_spy_mode
    
    # We'll return raw signals without ATR filter here to avoid circular imports
    # The ATR filtering will be applied in the main generator.py
    return signals 