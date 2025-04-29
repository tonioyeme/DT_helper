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