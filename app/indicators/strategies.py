import pandas as pd
import numpy as np

def calculate_ema_vwap_strategy(df):
    """
    Implement the EMA Cloud + VWAP Convergence strategy
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data and calculated indicators
        
    Returns:
        tuple: (bullish_signals, bearish_signals, targets, stop_losses)
    """
    df = df.copy()
    
    # Check if indicators are already calculated, if not calculate them
    if 'fast_ema' not in df.columns or 'slow_ema' not in df.columns:
        # Calculate EMA Cloud (8-9 EMA)
        df['fast_ema'] = df['close'].ewm(span=8, adjust=False).mean()
        df['slow_ema'] = df['close'].ewm(span=9, adjust=False).mean()
        df['cloud_bullish'] = df['fast_ema'] > df['slow_ema']
    
    if 'vwap' not in df.columns:
        # Calculate VWAP
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['tp_volume'] = df['typical_price'] * df['volume']
        
        # Reset cumulative calculations each day if datetime index is available
        if isinstance(df.index, pd.DatetimeIndex):
            df['day'] = df.index.date
            df['cum_tp_volume'] = df.groupby('day')['tp_volume'].cumsum()
            df['cum_volume'] = df.groupby('day')['volume'].cumsum()
        else:
            # For non-datetime indices, just do a regular cumsum
            df['cum_tp_volume'] = df['tp_volume'].cumsum()
            df['cum_volume'] = df['volume'].cumsum()
            
        # Protect against division by zero
        df['vwap'] = np.where(df['cum_volume'] > 0, 
                             df['cum_tp_volume'] / df['cum_volume'], 
                             df['close'])  # Use close price as fallback
    
    # Initialize signal arrays
    bullish_signals = pd.Series(False, index=df.index)
    bearish_signals = pd.Series(False, index=df.index)
    targets = pd.Series(np.nan, index=df.index)
    stop_losses = pd.Series(np.nan, index=df.index)
    
    # Look for EMA Cloud + VWAP setups
    for i in range(3, len(df)):
        try:
            # Bullish setup: price bounces off cloud bottom while above VWAP
            cloud_bottom = min(df['fast_ema'].iloc[i-1], df['slow_ema'].iloc[i-1])
            if (df['low'].iloc[i-1] <= cloud_bottom and 
                df['close'].iloc[i-1] > cloud_bottom and
                df['close'].iloc[i-1] > df['vwap'].iloc[i-1] and
                df['cloud_bullish'].iloc[i-1]):
                
                bullish_signals.iloc[i] = True
                
                # Calculate target and stop loss
                recent_high_end = min(i, len(df)-1)
                recent_high_start = max(0, i-10)
                recent_low_end = min(i, len(df)-1)
                recent_low_start = max(0, i-5)
                
                recent_swing_high = max(df['high'].iloc[recent_high_start:recent_high_end].max(), df['close'].iloc[i])
                recent_swing_low = min(df['low'].iloc[recent_low_start:recent_low_end])
                
                targets.iloc[i] = df['close'].iloc[i] + (1.8 * (df['close'].iloc[i] - recent_swing_low))
                stop_losses.iloc[i] = recent_swing_low
                
            # Bearish setup: price bounces down from cloud top while below VWAP
            cloud_top = max(df['fast_ema'].iloc[i-1], df['slow_ema'].iloc[i-1])
            if (df['high'].iloc[i-1] >= cloud_top and 
                df['close'].iloc[i-1] < cloud_top and
                df['close'].iloc[i-1] < df['vwap'].iloc[i-1] and
                not df['cloud_bullish'].iloc[i-1]):
                
                bearish_signals.iloc[i] = True
                
                # Calculate target and stop loss
                recent_high_end = min(i, len(df)-1)
                recent_high_start = max(0, i-5)
                recent_low_end = min(i, len(df)-1)
                recent_low_start = max(0, i-10)
                
                recent_swing_high = max(df['high'].iloc[recent_high_start:recent_high_end])
                recent_swing_low = min(df['low'].iloc[recent_low_start:recent_low_end].min(), df['close'].iloc[i])
                
                targets.iloc[i] = df['close'].iloc[i] - (1.8 * (recent_swing_high - df['close'].iloc[i]))
                stop_losses.iloc[i] = recent_swing_high
        except Exception as e:
            # Skip this iteration if any error occurs
            continue
    
    return bullish_signals, bearish_signals, targets, stop_losses

def calculate_measured_move_volume_strategy(df, window=20, volume_threshold=1.2):
    """
    Implement the Measured Move + Volume Confirmation strategy
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        window (int): Window size for swing detection
        volume_threshold (float): Volume threshold as multiple of 20-period average
        
    Returns:
        tuple: (bullish_signals, bearish_signals, targets, stop_losses)
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Initialize output Series
    bullish_signals = pd.Series(False, index=df.index)
    bearish_signals = pd.Series(False, index=df.index)
    targets = pd.Series(np.nan, index=df.index)
    stop_losses = pd.Series(np.nan, index=df.index)
    
    # If we don't have enough data for the calculations, return empty signals
    if len(df) < window + 5:
        return bullish_signals, bearish_signals, targets, stop_losses
    
    try:
        # Calculate ATR for trailing stop
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Calculate volume average and ratio for later filtering
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Find local highs and lows using rolling window
        # We'll identify points where the current value is the highest/lowest in the window
        df['local_high'] = df['high'].rolling(window=window, center=True).apply(
            lambda x: x.iloc[len(x)//2] == max(x), raw=False
        ).fillna(False).astype(bool)
        
        df['local_low'] = df['low'].rolling(window=window, center=True).apply(
            lambda x: x.iloc[len(x)//2] == min(x), raw=False
        ).fillna(False).astype(bool)
        
        # Process each row in the dataframe to look for patterns
        for i in range(window, len(df)-1):  # Ensure we have enough data before and after
            # Skip if volume is not above threshold
            if df['volume_ratio'].iloc[i] <= volume_threshold:
                continue
            
            # Look back for recent swing points
            look_back_start = max(0, i-window)
            recent_section = df.iloc[look_back_start:i]
            
            # Find positions of recent high and low points
            high_positions = recent_section.index[recent_section['local_high']].tolist()
            low_positions = recent_section.index[recent_section['local_low']].tolist()
            
            # We need at least one high and one low to form a pattern
            if not high_positions or not low_positions:
                continue
            
            # Get most recent high and low
            recent_high_idx = high_positions[-1]
            recent_low_idx = low_positions[-1]
            
            # Skip if we can't determine the pattern order
            if recent_high_idx not in df.index or recent_low_idx not in df.index:
                continue
                
            # Get location indices to determine order
            high_loc = df.index.get_loc(recent_high_idx)
            low_loc = df.index.get_loc(recent_low_idx)
            
            # Calculate swing values
            swing_high = df.loc[recent_high_idx, 'high']
            swing_low = df.loc[recent_low_idx, 'low']
            curr_close = df.iloc[i, df.columns.get_loc('close')]
            curr_high = df.iloc[i, df.columns.get_loc('high')]
            curr_low = df.iloc[i, df.columns.get_loc('low')]
            prev_high = df.iloc[i-1, df.columns.get_loc('high')]
            prev_low = df.iloc[i-1, df.columns.get_loc('low')]
            
            # Bullish pattern: High came after Low 
            if high_loc > low_loc:
                # Calculate Fibonacci levels
                retracement_50 = swing_high - (swing_high - swing_low) * 0.5
                retracement_618 = swing_high - (swing_high - swing_low) * 0.618
                
                # Check for retracement in the 50-61.8% zone and a breakout
                if (retracement_618 <= prev_low <= retracement_50 and curr_close > prev_high):
                    # Calculate target (23.6% extension) and stop loss
                    target = swing_high + (swing_high - swing_low) * 0.236
                    
                    # Stop loss is below recent low with ATR buffer
                    recent_lows = df.iloc[max(0, i-3):i+1]['low']
                    stop_loss = recent_lows.min() - df.iloc[i]['atr']
                    
                    # Set the signal
                    bullish_signals.iloc[i] = True
                    targets.iloc[i] = target
                    stop_losses.iloc[i] = stop_loss
            
            # Bearish pattern: Low came after High
            elif low_loc > high_loc:
                # Calculate Fibonacci levels
                retracement_50 = swing_low + (swing_high - swing_low) * 0.5
                retracement_618 = swing_low + (swing_high - swing_low) * 0.618
                
                # Check for retracement in the 50-61.8% zone and a breakdown
                if (retracement_50 <= prev_high <= retracement_618 and curr_close < prev_low):
                    # Calculate target (23.6% extension) and stop loss
                    target = swing_low - (swing_high - swing_low) * 0.236
                    
                    # Stop loss is above recent high with ATR buffer
                    recent_highs = df.iloc[max(0, i-3):i+1]['high']
                    stop_loss = recent_highs.max() + df.iloc[i]['atr']
                    
                    # Set the signal
                    bearish_signals.iloc[i] = True
                    targets.iloc[i] = target
                    stop_losses.iloc[i] = stop_loss
                    
    except Exception as e:
        # If anything fails, log it and return empty signals
        print(f"Error in measured move calculation: {str(e)}")
        pass
        
    return bullish_signals, bearish_signals, targets, stop_losses

def multi_indicator_confirmation(df, threshold=2):
    """
    Generate signals based on multiple indicator confirmation
    
    Args:
        df (pd.DataFrame): DataFrame with signals columns
        threshold (int): Minimum number of signals required for confirmation
        
    Returns:
        tuple: (confirmed_buy_signals, confirmed_sell_signals)
    """
    # Count number of active buy signals per row
    buy_signal_columns = [col for col in df.columns if 'buy' in col.lower() and df[col].dtype == bool]
    buy_signal_count = df[buy_signal_columns].sum(axis=1)
    
    # Count number of active sell signals per row
    sell_signal_columns = [col for col in df.columns if 'sell' in col.lower() and df[col].dtype == bool]
    sell_signal_count = df[sell_signal_columns].sum(axis=1)
    
    # Generate confirmed signals based on threshold
    confirmed_buy_signals = buy_signal_count >= threshold
    confirmed_sell_signals = sell_signal_count >= threshold
    
    return confirmed_buy_signals, confirmed_sell_signals

def multi_tf_confirmation(primary_tf, trading_tf, entry_tf):
    """
    Check alignment between multiple timeframes for trade confirmation
    
    This function verifies that:
    1. The primary trend aligns with the trading timeframe
    2. The trading timeframe has stronger momentum than the entry timeframe
    
    Args:
        primary_tf (dict): Data from primary (highest) timeframe with 'trend' key
        trading_tf (dict): Data from trading timeframe with 'trend' and 'momentum' keys
        entry_tf (dict): Data from entry timeframe with 'momentum' key
        
    Returns:
        bool: True if timeframes align for a valid trade, False otherwise
    """
    # Primary trend alignment check
    if (primary_tf['trend'] == trading_tf['trend'] 
        and trading_tf['momentum'] > entry_tf['momentum']):
        return True
    return False

def calculate_vwap_bollinger_strategy(df, vwap_period='day', bb_length=20, bb_std=2.0):
    """
    VWAP + Bollinger Bands strategy for 5-minute SPY trading
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV price data
        vwap_period (str): Period for VWAP reset ('day' for daily)
        bb_length (int): Period for Bollinger Bands calculation
        bb_std (float): Number of standard deviations for Bollinger Bands
        
    Returns:
        tuple: (bullish_signals, bearish_signals, targets, stop_losses)
    """
    from app.indicators import calculate_vwap, calculate_atr
    
    df = df.copy()
    
    # Calculate VWAP
    if 'vwap' not in df.columns:
        df['vwap'] = calculate_vwap(df)
    
    # Calculate Bollinger Bands on VWAP
    vwap_std = df['vwap'].rolling(bb_length).std()
    df['vwap_upper'] = df['vwap'] + bb_std * vwap_std
    df['vwap_lower'] = df['vwap'] - bb_std * vwap_std
    
    # Calculate ATR for stop losses if not already present
    if 'atr' not in df.columns:
        df['atr'] = calculate_atr(df)
    
    # Initialize signals
    bullish_signals = pd.Series(False, index=df.index)
    bearish_signals = pd.Series(False, index=df.index)
    targets = pd.Series(np.nan, index=df.index)
    stop_losses = pd.Series(np.nan, index=df.index)
    
    # Generate signals
    for i in range(3, len(df)):
        # Bullish signal: Price crosses above VWAP from below after touching lower band
        if (df['close'].iloc[i-1] < df['vwap'].iloc[i-1] and 
            df['close'].iloc[i] > df['vwap'].iloc[i] and
            df['low'].iloc[i-1] <= df['vwap_lower'].iloc[i-1]):
            
            bullish_signals.iloc[i] = True
            targets.iloc[i] = df['close'].iloc[i] + (df['close'].iloc[i] - df['vwap_lower'].iloc[i])
            stop_losses.iloc[i] = df['vwap_lower'].iloc[i] - df['atr'].iloc[i] * 0.5
        
        # Bearish signal: Price crosses below VWAP from above after touching upper band
        if (df['close'].iloc[i-1] > df['vwap'].iloc[i-1] and 
            df['close'].iloc[i] < df['vwap'].iloc[i] and
            df['high'].iloc[i-1] >= df['vwap_upper'].iloc[i-1]):
            
            bearish_signals.iloc[i] = True
            targets.iloc[i] = df['close'].iloc[i] - (df['vwap_upper'].iloc[i] - df['close'].iloc[i])
            stop_losses.iloc[i] = df['vwap_upper'].iloc[i] + df['atr'].iloc[i] * 0.5
    
    return bullish_signals, bearish_signals, targets, stop_losses

def calculate_orb_strategy(df, orb_minutes=5, confirmation_candles=1):
    """
    Opening Range Breakout strategy for SPY
    
    Args:
        df (pd.DataFrame): OHLCV data with datetime index
        orb_minutes (int): Opening range duration in minutes
        confirmation_candles (int): Number of candles to confirm breakout
        
    Returns:
        tuple: (bullish_signals, bearish_signals, targets, stop_losses)
    """
    from app.indicators import calculate_opening_range
    import pytz
    
    df = df.copy()
    
    # Initialize signals
    bullish_signals = pd.Series(False, index=df.index)
    bearish_signals = pd.Series(False, index=df.index)
    targets = pd.Series(np.nan, index=df.index)
    stop_losses = pd.Series(np.nan, index=df.index)
    
    # Ensure index is in Eastern Time for market hours
    eastern = pytz.timezone('US/Eastern')
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert(eastern)
    
    # Get opening range
    or_high, or_low = calculate_opening_range(df, minutes=orb_minutes)
    if or_high is None or or_low is None:
        return bullish_signals, bearish_signals, targets, stop_losses
    
    # Add ORB levels to dataframe
    df['or_high'] = or_high
    df['or_low'] = or_low
    
    # Find the index where opening range ends
    try:
        market_open_end = df.between_time('09:30', '09:' + str(30 + orb_minutes)).index[-1]
        or_end_idx = df.index.get_loc(market_open_end)
    except Exception:
        # If we can't determine the end of opening range, return empty signals
        return bullish_signals, bearish_signals, targets, stop_losses
    
    # Generate signals after opening range
    for i in range(or_end_idx + 1, len(df)):
        # Skip if we don't have enough data for confirmation
        if i - confirmation_candles < 0:
            continue
            
        # Check for bullish breakout
        if (df['close'].iloc[i] > or_high and 
            all(df['close'].iloc[i-confirmation_candles:i] <= or_high)):
            
            bullish_signals.iloc[i] = True
            range_size = or_high - or_low
            targets.iloc[i] = df['close'].iloc[i] + range_size
            stop_losses.iloc[i] = or_low
        
        # Check for bearish breakout
        if (df['close'].iloc[i] < or_low and 
            all(df['close'].iloc[i-confirmation_candles:i] >= or_low)):
            
            bearish_signals.iloc[i] = True
            range_size = or_high - or_low
            targets.iloc[i] = df['close'].iloc[i] - range_size
            stop_losses.iloc[i] = or_high
    
    return bullish_signals, bearish_signals, targets, stop_losses 