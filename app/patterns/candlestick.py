"""
Candlestick Pattern Detection Module

This module provides functions to identify various candlestick patterns in OHLC data.
Each function returns a pandas Series with True/False values indicating where the pattern is detected.
"""

import numpy as np
import pandas as pd


# Helper functions
def calculate_body_size(df):
    """Calculate the absolute and relative body size of candles."""
    df = df.copy()
    df['body_size'] = abs(df['close'] - df['open'])
    df['body_pct'] = df['body_size'] / ((df['high'] - df['low']) + 1e-10)  # Avoid division by zero
    return df

def calculate_upper_shadow(df):
    """Calculate the upper shadow of candles."""
    df = df.copy()
    df['upper_shadow'] = df.apply(
        lambda x: x['high'] - max(x['open'], x['close']), axis=1
    )
    df['upper_shadow_pct'] = df['upper_shadow'] / ((df['high'] - df['low']) + 1e-10)
    return df

def calculate_lower_shadow(df):
    """Calculate the lower shadow of candles."""
    df = df.copy()
    df['lower_shadow'] = df.apply(
        lambda x: min(x['open'], x['close']) - x['low'], axis=1
    )
    df['lower_shadow_pct'] = df['lower_shadow'] / ((df['high'] - df['low']) + 1e-10)
    return df

def is_bullish(df):
    """Check if a candle is bullish (close > open)."""
    return df['close'] > df['open']

def is_bearish(df):
    """Check if a candle is bearish (close < open)."""
    return df['close'] < df['open']

def calculate_trend(df, window=14):
    """Determine the trend using SMA."""
    df = df.copy()
    df['sma'] = df['close'].rolling(window=window).mean()
    df['trend_up'] = df['close'] > df['sma']
    df['trend_down'] = df['close'] < df['sma']
    return df

# Single Candlestick Patterns

def identify_hammer(df, trend_window=14, body_size_pct=0.3, lower_shadow_ratio=2.0, success_rate=0.65):
    """
    Identify Hammer candlestick pattern (bullish reversal).
    
    A hammer has a small body at the top of the candle with a long lower shadow.
    It appears in a downtrend and signals a potential reversal.
    
    Success Rate: 65%
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for trend determination
        body_size_pct: Maximum body size as a percentage of the total range
        lower_shadow_ratio: Minimum ratio of lower shadow to body size
        success_rate: Historical success rate for this pattern
    
    Returns:
        Series with boolean values where True indicates a hammer pattern
    """
    df = calculate_trend(df, window=trend_window)
    df = calculate_body_size(df)
    df = calculate_lower_shadow(df)
    df = calculate_upper_shadow(df)
    
    # Criteria for hammer:
    # 1. In a downtrend
    # 2. Small body at the top (small upper shadow)
    # 3. Long lower shadow (at least 2x the body size)
    
    hammer = (
        df['trend_down'] &  # In a downtrend
        (df['body_pct'] <= body_size_pct) &  # Small body
        (df['upper_shadow_pct'] <= 0.1) &  # Small upper shadow
        (df['lower_shadow'] >= df['body_size'] * lower_shadow_ratio)  # Long lower shadow
    )
    
    hammer.name = f'hammer_{int(success_rate*100)}'
    return hammer

def identify_shooting_star(df, trend_window=14, body_size_pct=0.3, upper_shadow_ratio=2.0, success_rate=0.72):
    """
    Identify Shooting Star candlestick pattern (bearish reversal).
    
    A shooting star has a small body at the bottom of the candle with a long upper shadow.
    It appears in an uptrend and signals a potential reversal.
    
    Success Rate: 72%
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for trend determination
        body_size_pct: Maximum body size as a percentage of the total range
        upper_shadow_ratio: Minimum ratio of upper shadow to body size
        success_rate: Historical success rate for this pattern
    
    Returns:
        Series with boolean values where True indicates a shooting star pattern
    """
    df = calculate_trend(df, window=trend_window)
    df = calculate_body_size(df)
    df = calculate_upper_shadow(df)
    df = calculate_lower_shadow(df)
    
    # Criteria for shooting star:
    # 1. In an uptrend
    # 2. Small body at the bottom (small lower shadow)
    # 3. Long upper shadow (at least 2x the body size)
    
    shooting_star = (
        df['trend_up'] &  # In an uptrend
        (df['body_pct'] <= body_size_pct) &  # Small body
        (df['lower_shadow_pct'] <= 0.1) &  # Small lower shadow
        (df['upper_shadow'] >= df['body_size'] * upper_shadow_ratio)  # Long upper shadow
    )
    
    shooting_star.name = f'shooting_star_{int(success_rate*100)}'
    return shooting_star

def identify_doji(df, body_size_threshold=0.1, shadow_ratio=0.1, success_rate=0.53):
    """
    Identify Doji candlestick pattern (indecision, potential reversal).
    
    A doji has an extremely small body (open and close at nearly the same level)
    with upper and lower shadows of varying length.
    
    Success Rate: 53%
    
    Args:
        df: DataFrame with OHLC data
        body_size_threshold: Maximum body size as a percentage of the total range
        shadow_ratio: Minimum ratio of shadows to total range
        success_rate: Historical success rate for this pattern
    
    Returns:
        Series with boolean values where True indicates a doji pattern
    """
    df = calculate_body_size(df)
    df = calculate_upper_shadow(df)
    df = calculate_lower_shadow(df)
    
    # Criteria for doji:
    # 1. Very small body (open and close nearly equal)
    # 2. Upper and lower shadows can vary
    
    doji = (
        (df['body_pct'] <= body_size_threshold) &  # Very small body
        (df['upper_shadow_pct'] + df['lower_shadow_pct'] >= shadow_ratio)  # Has shadows
    )
    
    doji.name = f'doji_{int(success_rate*100)}'
    return doji

# Two Candlestick Patterns

def identify_bullish_engulfing(df, trend_window=14, success_rate=0.78):
    """
    Identify Bullish Engulfing candlestick pattern (bullish reversal).
    
    A bullish engulfing pattern consists of a bearish candle followed by a bullish candle
    that completely 'engulfs' the previous candle. It signals a potential trend reversal from
    bearish to bullish.
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for detecting prior trend
        success_rate: Historical success rate of this pattern
        
    Returns:
        Series with boolean values where True indicates a bullish engulfing pattern
    """
    df = calculate_body_size(df)
    df = calculate_trend(df, window=trend_window)
    
    # Criteria for bullish engulfing:
    # 1. Downtrend (price below MA)
    # 2. Previous candle is bearish
    # 3. Current candle is bullish
    # 4. Current candle's body completely engulfs previous candle's body
    bullish_engulfing = (
        df['trend_down'].shift(1) &  # In downtrend
        ~is_bullish(df.shift(1)) &  # Previous candle is bearish
        is_bullish(df) &  # Current candle is bullish
        (df['open'] <= df['close'].shift(1)) &  # Current open <= previous close
        (df['close'] >= df['open'].shift(1))  # Current close >= previous open
    )
    
    bullish_engulfing.name = f'bullish_engulfing_{int(success_rate*100)}'
    return bullish_engulfing

def identify_bearish_engulfing(df, trend_window=14, success_rate=0.82):
    """
    Identify Bearish Engulfing candlestick pattern (bearish reversal).
    
    A bearish engulfing pattern consists of a bullish candle followed by a bearish candle
    that completely 'engulfs' the previous candle. It signals a potential trend reversal from
    bullish to bearish.
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for detecting prior trend
        success_rate: Historical success rate of this pattern
        
    Returns:
        Series with boolean values where True indicates a bearish engulfing pattern
    """
    df = calculate_body_size(df)
    df = calculate_trend(df, window=trend_window)
    
    # Criteria for bearish engulfing:
    # 1. Uptrend (price above MA)
    # 2. Previous candle is bullish
    # 3. Current candle is bearish
    # 4. Current candle's body completely engulfs previous candle's body
    bearish_engulfing = (
        df['trend_up'].shift(1) &  # In uptrend
        is_bullish(df.shift(1)) &  # Previous candle is bullish
        ~is_bullish(df) &  # Current candle is bearish
        (df['open'] >= df['close'].shift(1)) &  # Current open >= previous close
        (df['close'] <= df['open'].shift(1))  # Current close <= previous open
    )
    
    bearish_engulfing.name = f'bearish_engulfing_{int(success_rate*100)}'
    return bearish_engulfing

def identify_engulfing(df, trend_window=14):
    """
    Identify both Bullish and Bearish Engulfing patterns.
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for detecting prior trend
        
    Returns:
        Tuple of (bullish_engulfing, bearish_engulfing) Series
    """
    bullish = identify_bullish_engulfing(df, trend_window)
    bearish = identify_bearish_engulfing(df, trend_window)
    return bullish, bearish

def identify_piercing_line(df, trend_window=14, pierce_threshold=0.5, success_rate=0.64):
    """
    Identify Piercing Line candlestick pattern (bullish reversal).
    
    A piercing line consists of a bearish candle followed by a bullish candle
    that opens below the previous candle's low and closes above the midpoint
    of the previous candle's body. It appears in a downtrend.
    
    Success Rate: 64%
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for trend determination
        pierce_threshold: Minimum threshold for piercing (0.5 = midpoint)
        success_rate: Historical success rate for this pattern
    
    Returns:
        Series with boolean values where True indicates a piercing line pattern
    """
    df = calculate_trend(df, window=trend_window)
    
    # Create shifted columns for the previous candle
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_trend_down'] = df['trend_down'].shift(1)
    
    # Calculate the midpoint of the previous candle's body
    df['prev_midpoint'] = df['prev_open'] + (df['prev_open'] - df['prev_close']) * pierce_threshold
    
    # Criteria for piercing line:
    # 1. In a downtrend (previous candle)
    # 2. Previous candle is bearish (close < open)
    # 3. Current candle is bullish (close > open)
    # 4. Current candle opens below previous candle's low
    # 5. Current candle closes above the midpoint of previous candle's body
    
    piercing_line = (
        df['prev_trend_down'] &  # In a downtrend
        (df['prev_close'] < df['prev_open']) &  # Previous candle is bearish
        (df['close'] > df['open']) &  # Current candle is bullish
        (df['open'] <= df['prev_low']) &  # Current candle opens below previous low
        (df['close'] > df['prev_midpoint']) &  # Current candle closes above midpoint
        (df['close'] < df['prev_open'])  # But doesn't close above previous open
    )
    
    piercing_line.name = f'piercing_line_{int(success_rate*100)}'
    return piercing_line

def identify_dark_cloud_cover(df, trend_window=14, cloud_threshold=0.5, success_rate=0.73):
    """
    Identify Dark Cloud Cover candlestick pattern (bearish reversal).
    
    A dark cloud cover consists of a bullish candle followed by a bearish candle
    that opens above the previous candle's high and closes below the midpoint
    of the previous candle's body. It appears in an uptrend.
    
    Success Rate: 73%
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for trend determination
        cloud_threshold: Minimum threshold for penetration (0.5 = midpoint)
        success_rate: Historical success rate for this pattern
    
    Returns:
        Series with boolean values where True indicates a dark cloud cover pattern
    """
    df = calculate_trend(df, window=trend_window)
    
    # Create shifted columns for the previous candle
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_trend_up'] = df['trend_up'].shift(1)
    
    # Calculate the midpoint of the previous candle's body
    df['prev_midpoint'] = df['prev_close'] - (df['prev_close'] - df['prev_open']) * cloud_threshold
    
    # Criteria for dark cloud cover:
    # 1. In an uptrend (previous candle)
    # 2. Previous candle is bullish (close > open)
    # 3. Current candle is bearish (close < open)
    # 4. Current candle opens above previous candle's high
    # 5. Current candle closes below the midpoint of previous candle's body
    
    dark_cloud_cover = (
        df['prev_trend_up'] &  # In an uptrend
        (df['prev_close'] > df['prev_open']) &  # Previous candle is bullish
        (df['close'] < df['open']) &  # Current candle is bearish
        (df['open'] >= df['prev_high']) &  # Current candle opens above previous high
        (df['close'] < df['prev_midpoint']) &  # Current candle closes below midpoint
        (df['close'] > df['prev_open'])  # But doesn't close below previous open
    )
    
    dark_cloud_cover.name = f'dark_cloud_cover_{int(success_rate*100)}'
    return dark_cloud_cover

def identify_bullish_harami(df, trend_window=14, success_rate=0.68):
    """
    Identify Bullish Harami candlestick pattern (bullish reversal).
    
    A bullish harami consists of a large bearish candle followed by a small bullish candle
    that is completely contained within the previous candle's body. It appears in a downtrend.
    
    Success Rate: 68%
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for trend determination
        success_rate: Historical success rate for this pattern
    
    Returns:
        Series with boolean values where True indicates a bullish harami pattern
    """
    df = calculate_trend(df, window=trend_window)
    
    # Create shifted columns for the previous candle
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_trend_down'] = df['trend_down'].shift(1)
    
    # Criteria for bullish harami:
    # 1. In a downtrend (previous candle)
    # 2. Previous candle is bearish (close < open)
    # 3. Current candle is bullish (close > open)
    # 4. Current candle is completely contained within the previous candle's body
    
    bullish_harami = (
        df['prev_trend_down'] &  # In a downtrend
        (df['prev_close'] < df['prev_open']) &  # Previous candle is bearish
        (df['close'] > df['open']) &  # Current candle is bullish
        (df['open'] > df['prev_close']) &  # Current candle opens above previous close
        (df['close'] < df['prev_open'])  # Current candle closes below previous open
    )
    
    bullish_harami.name = f'bullish_harami_{int(success_rate*100)}'
    return bullish_harami

def identify_bearish_harami(df, trend_window=14, success_rate=0.70):
    """
    Identify Bearish Harami candlestick pattern (bearish reversal).
    
    A bearish harami consists of a large bullish candle followed by a small bearish candle
    that is completely contained within the previous candle's body. It appears in an uptrend.
    
    Success Rate: 70%
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for trend determination
        success_rate: Historical success rate for this pattern
    
    Returns:
        Series with boolean values where True indicates a bearish harami pattern
    """
    df = calculate_trend(df, window=trend_window)
    
    # Create shifted columns for the previous candle
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_trend_up'] = df['trend_up'].shift(1)
    
    # Criteria for bearish harami:
    # 1. In an uptrend (previous candle)
    # 2. Previous candle is bullish (close > open)
    # 3. Current candle is bearish (close < open)
    # 4. Current candle is completely contained within the previous candle's body
    
    bearish_harami = (
        df['prev_trend_up'] &  # In an uptrend
        (df['prev_close'] > df['prev_open']) &  # Previous candle is bullish
        (df['close'] < df['open']) &  # Current candle is bearish
        (df['open'] < df['prev_close']) &  # Current candle opens below previous close
        (df['close'] > df['prev_open'])  # Current candle closes above previous open
    )
    
    bearish_harami.name = f'bearish_harami_{int(success_rate*100)}'
    return bearish_harami

# Three Candlestick Patterns

def identify_morning_star(df, trend_window=14, doji_threshold=0.1, success_rate=0.76):
    """
    Identify Morning Star candlestick pattern (bullish reversal).
    
    A morning star consists of a bearish candle, followed by a small-bodied candle 
    (often a doji) that gaps down, followed by a bullish candle that gaps up and 
    closes at least halfway up the first candle's body. It appears in a downtrend.
    
    Success Rate: 76%
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for trend determination
        doji_threshold: Maximum body size of middle candle as a percentage of total range
        success_rate: Historical success rate for this pattern
    
    Returns:
        Series with boolean values where True indicates a morning star pattern
    """
    df = calculate_trend(df, window=trend_window)
    df = calculate_body_size(df)
    
    # Create shifted columns
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_body_pct'] = df['body_pct'].shift(1)
    
    df['prev2_open'] = df['open'].shift(2)
    df['prev2_close'] = df['close'].shift(2)
    df['prev2_trend_down'] = df['trend_down'].shift(2)
    
    # Calculate midpoint of the first candle
    df['prev2_midpoint'] = df['prev2_open'] + (df['prev2_open'] - df['prev2_close']) * 0.5
    
    # Criteria for morning star:
    # 1. In a downtrend (first candle)
    # 2. First candle is bearish (close < open)
    # 3. Second candle has a small body (doji-like)
    # 4. Second candle gaps down from first candle
    # 5. Third candle is bullish (close > open)
    # 6. Third candle closes at least halfway up the first candle's body
    
    morning_star = (
        df['prev2_trend_down'] &  # In a downtrend
        (df['prev2_close'] < df['prev2_open']) &  # First candle is bearish
        (df['prev_body_pct'] <= doji_threshold) &  # Second candle has small body
        (df['prev_open'] < df['prev2_close']) &  # Second candle gaps down
        (df['close'] > df['open']) &  # Third candle is bullish
        (df['close'] >= df['prev2_midpoint'])  # Third candle closes above midpoint of first
    )
    
    morning_star.name = f'morning_star_{int(success_rate*100)}'
    return morning_star

def identify_evening_star(df, trend_window=14, doji_threshold=0.1, success_rate=0.72):
    """
    Identify Evening Star candlestick pattern (bearish reversal).
    
    An evening star consists of a bullish candle, followed by a small-bodied candle 
    (often a doji) that gaps up, followed by a bearish candle that gaps down and 
    closes at least halfway down the first candle's body. It appears in an uptrend.
    
    Success Rate: 72%
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for trend determination
        doji_threshold: Maximum body size of middle candle as a percentage of total range
        success_rate: Historical success rate for this pattern
    
    Returns:
        Series with boolean values where True indicates an evening star pattern
    """
    df = calculate_trend(df, window=trend_window)
    df = calculate_body_size(df)
    
    # Create shifted columns
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_body_pct'] = df['body_pct'].shift(1)
    
    df['prev2_open'] = df['open'].shift(2)
    df['prev2_close'] = df['close'].shift(2)
    df['prev2_trend_up'] = df['trend_up'].shift(2)
    
    # Calculate midpoint of the first candle
    df['prev2_midpoint'] = df['prev2_close'] - (df['prev2_close'] - df['prev2_open']) * 0.5
    
    # Criteria for evening star:
    # 1. In an uptrend (first candle)
    # 2. First candle is bullish (close > open)
    # 3. Second candle has a small body (doji-like)
    # 4. Second candle gaps up from first candle
    # 5. Third candle is bearish (close < open)
    # 6. Third candle closes at least halfway down the first candle's body
    
    evening_star = (
        df['prev2_trend_up'] &  # In an uptrend
        (df['prev2_close'] > df['prev2_open']) &  # First candle is bullish
        (df['prev_body_pct'] <= doji_threshold) &  # Second candle has small body
        (df['prev_open'] > df['prev2_close']) &  # Second candle gaps up
        (df['close'] < df['open']) &  # Third candle is bearish
        (df['close'] <= df['prev2_midpoint'])  # Third candle closes below midpoint of first
    )
    
    evening_star.name = f'evening_star_{int(success_rate*100)}'
    return evening_star

def identify_three_white_soldiers(df, trend_window=14, success_rate=0.83):
    """
    Identify Three White Soldiers candlestick pattern (bullish reversal/continuation).
    
    Three White Soldiers consists of three consecutive bullish candles, each opening
    within the previous candle's body and closing higher than the previous close.
    
    Success Rate: 83%
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for trend determination
        success_rate: Historical success rate for this pattern
    
    Returns:
        Series with boolean values where True indicates a three white soldiers pattern
    """
    df = calculate_trend(df, window=trend_window)
    
    # Create shifted columns for the previous candles
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    
    df['prev2_open'] = df['open'].shift(2)
    df['prev2_close'] = df['close'].shift(2)
    df['prev2_trend_down'] = df['trend_down'].shift(2)
    
    # Criteria for three white soldiers:
    # 1. Often (but not always) in a downtrend before pattern
    # 2. Three consecutive bullish candles (close > open)
    # 3. Each candle opens within the previous candle's body
    # 4. Each candle closes higher than the previous candle's close
    
    three_white_soldiers = (
        # Optional criteria for trend (not strictly required)
        # df['prev2_trend_down'] &  
        
        # First candle is bullish
        (df['prev2_close'] > df['prev2_open']) &
        
        # Second candle is bullish and opens within previous body
        (df['prev_close'] > df['prev_open']) &
        (df['prev_open'] > df['prev2_open']) &
        (df['prev_open'] < df['prev2_close']) &
        (df['prev_close'] > df['prev2_close']) &
        
        # Third candle is bullish and opens within previous body
        (df['close'] > df['open']) &
        (df['open'] > df['prev_open']) &
        (df['open'] < df['prev_close']) &
        (df['close'] > df['prev_close'])
    )
    
    three_white_soldiers.name = f'three_white_soldiers_{int(success_rate*100)}'
    return three_white_soldiers

def identify_three_black_crows(df, trend_window=14, success_rate=0.78):
    """
    Identify Three Black Crows candlestick pattern (bearish reversal/continuation).
    
    Three Black Crows consists of three consecutive bearish candles, each opening
    within the previous candle's body and closing lower than the previous close.
    
    Success Rate: 78%
    
    Args:
        df: DataFrame with OHLC data
        trend_window: Window size for trend determination
        success_rate: Historical success rate for this pattern
    
    Returns:
        Series with boolean values where True indicates a three black crows pattern
    """
    df = calculate_trend(df, window=trend_window)
    
    # Create shifted columns for the previous candles
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    
    df['prev2_open'] = df['open'].shift(2)
    df['prev2_close'] = df['close'].shift(2)
    df['prev2_trend_up'] = df['trend_up'].shift(2)
    
    # Criteria for three black crows:
    # 1. Often (but not always) in an uptrend before pattern
    # 2. Three consecutive bearish candles (close < open)
    # 3. Each candle opens within the previous candle's body
    # 4. Each candle closes lower than the previous candle's close
    
    three_black_crows = (
        # Optional criteria for trend (not strictly required)
        # df['prev2_trend_up'] &  
        
        # First candle is bearish
        (df['prev2_close'] < df['prev2_open']) &
        
        # Second candle is bearish and opens within previous body
        (df['prev_close'] < df['prev_open']) &
        (df['prev_open'] < df['prev2_open']) &
        (df['prev_open'] > df['prev2_close']) &
        (df['prev_close'] < df['prev2_close']) &
        
        # Third candle is bearish and opens within previous body
        (df['close'] < df['open']) &
        (df['open'] < df['prev_open']) &
        (df['open'] > df['prev_close']) &
        (df['close'] < df['prev_close'])
    )
    
    three_black_crows.name = f'three_black_crows_{int(success_rate*100)}'
    return three_black_crows

# Combined pattern detection

def detect_all_patterns(df):
    """
    Detect all candlestick patterns in the given OHLC data.
    
    Args:
        df: DataFrame with OHLC data (must contain columns: open, high, low, close)
    
    Returns:
        DataFrame with boolean columns for each detected pattern
    """
    result = df.copy()
    
    # Single candlestick patterns
    result['hammer'] = identify_hammer(df)
    result['shooting_star'] = identify_shooting_star(df)
    result['doji'] = identify_doji(df)
    
    # Two candlestick patterns
    result['bullish_engulfing'], result['bearish_engulfing'] = identify_engulfing(df)
    result['piercing_line'] = identify_piercing_line(df)
    result['dark_cloud_cover'] = identify_dark_cloud_cover(df)
    result['bullish_harami'] = identify_bullish_harami(df)
    result['bearish_harami'] = identify_bearish_harami(df)
    
    # Three candlestick patterns
    result['morning_star'] = identify_morning_star(df)
    result['evening_star'] = identify_evening_star(df)
    result['three_white_soldiers'] = identify_three_white_soldiers(df)
    result['three_black_crows'] = identify_three_black_crows(df)
    
    return result 