"""
Price Action Pattern Detection Module

This module provides functions to identify chart patterns in OHLC data.
Each function returns a pandas Series with True/False values indicating where the pattern is detected.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# Helper Functions

def find_local_peaks(series, window=10):
    """Find local peaks in a time series."""
    peaks, _ = find_peaks(series, distance=window//2)
    return peaks

def find_local_troughs(series, window=10):
    """Find local troughs in a time series."""
    troughs, _ = find_peaks(-series, distance=window//2)
    return troughs

def smoothed_series(series, window=5):
    """
    Create a smoothed version of a series using rolling average.
    
    Args:
        series: Series to smooth
        window: Window size for rolling average
    
    Returns:
        Smoothed series
    """
    return series.rolling(window=window, center=True).mean()

def calculate_trend(df, column='close', window=14):
    """Determine the trend using SMA."""
    df = df.copy()
    df['sma'] = df[column].rolling(window=window).mean()
    df['trend_up'] = df[column] > df['sma']
    df['trend_down'] = df[column] < df['sma']
    return df


# Reversal Patterns

def detect_head_and_shoulders(df, window=20, depth_pct=0.03, success_rate=0.84):
    """
    Detects Head and Shoulders pattern (bearish reversal)
    Success rate: 84%
    
    Parameters:
    - window: lookback window for peak detection
    - depth_pct: minimum percentage depth between peaks and troughs
    """
    pattern = pd.Series(False, index=df.index)
    
    # Find peaks and troughs
    peaks = find_local_peaks(df['high'], window)
    troughs = find_local_troughs(df['low'], window)
    
    if len(peaks) < 3 or len(troughs) < 2:
        return pattern
    
    for i in range(len(peaks) - 2):
        # Need 3 peaks and 2 troughs
        if i + 2 >= len(peaks) or i + 1 >= len(troughs):
            continue
            
        p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
        t1, t2 = troughs[i], troughs[i+1]
        
        # Check sequence (p1 - t1 - p2 - t2 - p3)
        if not (p1 < t1 < p2 < t2 < p3):
            continue
        
        # Get price values
        p1_val, p2_val, p3_val = df['high'].iloc[p1], df['high'].iloc[p2], df['high'].iloc[p3]
        t1_val, t2_val = df['low'].iloc[t1], df['low'].iloc[t2]
        
        # Head should be higher than shoulders
        if not (p2_val > p1_val and p2_val > p3_val):
            continue
            
        # Shoulders should be roughly at same level
        if abs(p1_val - p3_val) / p1_val > 0.1:
            continue
            
        # Troughs should be roughly at same level (neckline)
        if abs(t1_val - t2_val) / t1_val > 0.05:
            continue
            
        # Pattern should have adequate depth
        depth1 = (p1_val - t1_val) / p1_val
        depth2 = (p2_val - max(t1_val, t2_val)) / p2_val
        depth3 = (p3_val - t2_val) / p3_val
        
        if depth1 > depth_pct and depth2 > depth_pct and depth3 > depth_pct:
            # Mark pattern at the right shoulder
            pattern.iloc[p3:p3+window] = True
            
    return pattern

def detect_inverted_head_and_shoulders(df, window=20, depth_pct=0.03, success_rate=0.83):
    """
    Detects Inverted Head and Shoulders pattern (bullish reversal)
    Success rate: 83%
    
    Parameters:
    - window: lookback window for peak detection
    - depth_pct: minimum percentage depth between peaks and troughs
    """
    pattern = pd.Series(False, index=df.index)
    
    # Find peaks and troughs
    peaks = find_local_peaks(df['high'], window)
    troughs = find_local_troughs(df['low'], window)
    
    if len(troughs) < 3 or len(peaks) < 2:
        return pattern
    
    for i in range(len(troughs) - 2):
        # Need 3 troughs and 2 peaks
        if i + 2 >= len(troughs) or i + 1 >= len(peaks):
            continue
            
        t1, t2, t3 = troughs[i], troughs[i+1], troughs[i+2]
        p1, p2 = peaks[i], peaks[i+1]
        
        # Check sequence (t1 - p1 - t2 - p2 - t3)
        if not (t1 < p1 < t2 < p2 < t3):
            continue
        
        # Get price values
        t1_val, t2_val, t3_val = df['low'].iloc[t1], df['low'].iloc[t2], df['low'].iloc[t3]
        p1_val, p2_val = df['high'].iloc[p1], df['high'].iloc[p2]
        
        # Head should be lower than shoulders
        if not (t2_val < t1_val and t2_val < t3_val):
            continue
            
        # Shoulders should be roughly at same level
        if abs(t1_val - t3_val) / t1_val > 0.1:
            continue
            
        # Peaks should be roughly at same level (neckline)
        if abs(p1_val - p2_val) / p1_val > 0.05:
            continue
            
        # Pattern should have adequate depth
        depth1 = (p1_val - t1_val) / p1_val
        depth2 = (min(p1_val, p2_val) - t2_val) / min(p1_val, p2_val)
        depth3 = (p2_val - t3_val) / p2_val
        
        if depth1 > depth_pct and depth2 > depth_pct and depth3 > depth_pct:
            # Mark pattern at the right shoulder
            pattern.iloc[t3:t3+window] = True
            
    return pattern

def detect_double_top(df, window=15, depth_pct=0.03, success_rate=0.79):
    """
    Detects Double Top pattern (bearish reversal)
    Success rate: 79%
    
    Parameters:
    - window: lookback window for peak detection
    - depth_pct: minimum percentage depth between peaks and troughs
    """
    pattern = pd.Series(False, index=df.index)
    
    # Find peaks and troughs
    peaks = find_local_peaks(df['high'], window)
    troughs = find_local_troughs(df['low'], window)
    
    if len(peaks) < 2 or len(troughs) < 1:
        return pattern
    
    for i in range(len(peaks) - 1):
        # Need 2 peaks and 1 trough
        if i >= len(troughs):
            continue
            
        p1, p2 = peaks[i], peaks[i+1]
        t1 = troughs[i]
        
        # Check sequence (p1 - t1 - p2)
        if not (p1 < t1 < p2):
            continue
        
        # Get price values
        p1_val, p2_val = df['high'].iloc[p1], df['high'].iloc[p2]
        t1_val = df['low'].iloc[t1]
        
        # Peaks should be at similar levels (within 3%)
        if abs(p1_val - p2_val) / p1_val > 0.03:
            continue
            
        # Adequate depth between peak and trough
        depth = (p1_val - t1_val) / p1_val
        
        if depth > depth_pct:
            # Mark pattern at the second peak
            pattern.iloc[p2:p2+window] = True
            
    return pattern

def detect_double_bottom(df, window=15, depth_pct=0.03, success_rate=0.74):
    """
    Detects Double Bottom pattern (bullish reversal)
    Success rate: 74%
    
    Parameters:
    - window: lookback window for peak detection
    - depth_pct: minimum percentage depth between peaks and troughs
    """
    pattern = pd.Series(False, index=df.index)
    
    # Find peaks and troughs
    peaks = find_local_peaks(df['high'], window)
    troughs = find_local_troughs(df['low'], window)
    
    if len(troughs) < 2 or len(peaks) < 1:
        return pattern
    
    for i in range(len(troughs) - 1):
        # Need 2 troughs and 1 peak
        if i >= len(peaks):
            continue
            
        t1, t2 = troughs[i], troughs[i+1]
        p1 = peaks[i]
        
        # Check sequence (t1 - p1 - t2)
        if not (t1 < p1 < t2):
            continue
        
        # Get price values
        t1_val, t2_val = df['low'].iloc[t1], df['low'].iloc[t2]
        p1_val = df['high'].iloc[p1]
        
        # Troughs should be at similar levels (within 3%)
        if abs(t1_val - t2_val) / t1_val > 0.03:
            continue
            
        # Adequate depth between peak and trough
        depth = (p1_val - t1_val) / p1_val
        
        if depth > depth_pct:
            # Mark pattern at the second trough
            pattern.iloc[t2:t2+window] = True
            
    return pattern

def detect_triple_top(df, window=20, depth_pct=0.03, success_rate=0.87):
    """
    Detects Triple Top pattern (bearish reversal)
    Success rate: 87%
    
    Parameters:
    - window: lookback window for peak detection
    - depth_pct: minimum percentage depth between peaks and troughs
    """
    pattern = pd.Series(False, index=df.index)
    
    # Find peaks and troughs
    peaks = find_local_peaks(df['high'], window)
    troughs = find_local_troughs(df['low'], window)
    
    if len(peaks) < 3 or len(troughs) < 2:
        return pattern
    
    for i in range(len(peaks) - 2):
        # Need 3 peaks and 2 troughs
        if i + 1 >= len(troughs):
            continue
            
        p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
        t1, t2 = troughs[i], troughs[i+1]
        
        # Check sequence (p1 - t1 - p2 - t2 - p3)
        if not (p1 < t1 < p2 < t2 < p3):
            continue
        
        # Get price values
        p1_val, p2_val, p3_val = df['high'].iloc[p1], df['high'].iloc[p2], df['high'].iloc[p3]
        t1_val, t2_val = df['low'].iloc[t1], df['low'].iloc[t2]
        
        # Peaks should be at similar levels (within 3%)
        if abs(p1_val - p2_val) / p1_val > 0.03 or abs(p2_val - p3_val) / p2_val > 0.03:
            continue
            
        # Troughs should have adequate depth
        depth1 = (p1_val - t1_val) / p1_val
        depth2 = (p2_val - t2_val) / p2_val
        
        if depth1 > depth_pct and depth2 > depth_pct:
            # Mark pattern at the third peak
            pattern.iloc[p3:p3+window] = True
            
    return pattern

def detect_triple_bottom(df, window=20, depth_pct=0.03, success_rate=0.91):
    """
    Detects Triple Bottom pattern (bullish reversal)
    Success rate: 91%
    
    Parameters:
    - window: lookback window for peak detection
    - depth_pct: minimum percentage depth between peaks and troughs
    """
    pattern = pd.Series(False, index=df.index)
    
    # Find peaks and troughs
    peaks = find_local_peaks(df['high'], window)
    troughs = find_local_troughs(df['low'], window)
    
    if len(troughs) < 3 or len(peaks) < 2:
        return pattern
    
    for i in range(len(troughs) - 2):
        # Need 3 troughs and 2 peaks
        if i + 1 >= len(peaks):
            continue
            
        t1, t2, t3 = troughs[i], troughs[i+1], troughs[i+2]
        p1, p2 = peaks[i], peaks[i+1]
        
        # Check sequence (t1 - p1 - t2 - p2 - t3)
        if not (t1 < p1 < t2 < p2 < t3):
            continue
        
        # Get price values
        t1_val, t2_val, t3_val = df['low'].iloc[t1], df['low'].iloc[t2], df['low'].iloc[t3]
        p1_val, p2_val = df['high'].iloc[p1], df['high'].iloc[p2]
        
        # Troughs should be at similar levels (within 3%)
        if abs(t1_val - t2_val) / t1_val > 0.03 or abs(t2_val - t3_val) / t2_val > 0.03:
            continue
            
        # Peaks should have adequate height
        depth1 = (p1_val - t1_val) / p1_val
        depth2 = (p2_val - t2_val) / p2_val
        
        if depth1 > depth_pct and depth2 > depth_pct:
            # Mark pattern at the third trough
            pattern.iloc[t3:t3+window] = True
            
    return pattern


# Continuation Patterns

def detect_rectangle(df, window=15, tolerance=0.03, success_rate=0.65):
    """
    Detects Rectangle pattern (continuation)
    Success rate: 65%
    
    Parameters:
    - window: lookback window for peak/trough detection
    - tolerance: maximum deviation from the support/resistance lines
    """
    pattern = pd.Series(False, index=df.index)
    
    # Find peaks and troughs
    peaks = find_local_peaks(df['high'], window)
    troughs = find_local_troughs(df['low'], window)
    
    if len(peaks) < 2 or len(troughs) < 2:
        return pattern
    
    for i in range(len(df.index) - window):
        end_idx = i + window
        if end_idx >= len(df):
            break
            
        # Get section of data
        section = df.iloc[i:end_idx]
        
        # Find resistance and support levels
        section_peaks = [p for p in peaks if i <= p < end_idx]
        section_troughs = [t for t in troughs if i <= t < end_idx]
        
        if len(section_peaks) < 2 or len(section_troughs) < 2:
            continue
            
        # Calculate potential resistance level (average of peaks)
        resistance = np.mean([df['high'].iloc[p] for p in section_peaks])
        
        # Calculate potential support level (average of troughs)
        support = np.mean([df['low'].iloc[t] for t in section_troughs])
        
        # Check if highs and lows are within tolerance of their levels
        within_tolerance = True
        for p in section_peaks:
            if abs(df['high'].iloc[p] - resistance) / resistance > tolerance:
                within_tolerance = False
                break
                
        for t in section_troughs:
            if abs(df['low'].iloc[t] - support) / support > tolerance:
                within_tolerance = False
                break
                
        if within_tolerance and (resistance - support) / support > 0.02:
            pattern.iloc[end_idx-5:end_idx] = True
            
    return pattern

def detect_channel(df, window=20, success_rate=0.67):
    """
    Detects Channel patterns (ascending/descending)
    Success rate: 67%
    
    Parameters:
    - window: lookback window for detection
    """
    pattern = pd.Series(False, index=df.index)
    
    # Find peaks and troughs
    peaks = find_local_peaks(df['high'], window//2)
    troughs = find_local_troughs(df['low'], window//2)
    
    if len(peaks) < 3 or len(troughs) < 3:
        return pattern
    
    for i in range(len(df.index) - window):
        end_idx = i + window
        if end_idx >= len(df):
            break
            
        # Get section of data
        section = df.iloc[i:end_idx]
        
        # Find resistance and support points
        section_peaks = [p for p in peaks if i <= p < end_idx]
        section_troughs = [t for t in troughs if i <= t < end_idx]
        
        if len(section_peaks) < 3 or len(section_troughs) < 3:
            continue
            
        # Get peak values and indices relative to section
        peak_values = [df['high'].iloc[p] for p in section_peaks]
        peak_indices = [p - i for p in section_peaks]
        
        # Get trough values and indices relative to section
        trough_values = [df['low'].iloc[t] for t in section_troughs]
        trough_indices = [t - i for t in section_troughs]
        
        # Fit lines to peaks and troughs
        try:
            peak_slope, peak_intercept = np.polyfit(peak_indices, peak_values, 1)
            trough_slope, trough_intercept = np.polyfit(trough_indices, trough_values, 1)
            
            # Check if slopes are similar (parallel lines)
            if abs(peak_slope - trough_slope) / abs(peak_slope) < 0.3:
                # Calculate channel width
                channel_width = (peak_intercept - trough_intercept) / trough_intercept
                
                # Channel should have adequate width
                if channel_width > 0.02:
                    # Calculate r-squared for both lines
                    peak_fitted = [peak_slope * idx + peak_intercept for idx in peak_indices]
                    trough_fitted = [trough_slope * idx + trough_intercept for idx in trough_indices]
                    
                    peak_ss_tot = sum((np.mean(peak_values) - peak_val) ** 2 for peak_val in peak_values)
                    peak_ss_res = sum((peak_val - fit_val) ** 2 for peak_val, fit_val in zip(peak_values, peak_fitted))
                    
                    trough_ss_tot = sum((np.mean(trough_values) - trough_val) ** 2 for trough_val in trough_values)
                    trough_ss_res = sum((trough_val - fit_val) ** 2 for trough_val, fit_val in zip(trough_values, trough_fitted))
                    
                    if peak_ss_tot > 0 and trough_ss_tot > 0:
                        peak_r2 = 1 - (peak_ss_res / peak_ss_tot)
                        trough_r2 = 1 - (trough_ss_res / trough_ss_tot)
                        
                        # Both lines should fit well
                        if peak_r2 > 0.75 and trough_r2 > 0.75:
                            pattern.iloc[end_idx-5:end_idx] = True
        except:
            continue
            
    return pattern

def detect_triangle(df, window=20, success_rate=0.73):
    """
    Detects Triangle patterns (ascending/descending/symmetric)
    Success rate: 73%
    
    Parameters:
    - window: lookback window for detection
    """
    pattern = pd.Series(False, index=df.index)
    
    # Find peaks and troughs
    peaks = find_local_peaks(df['high'], window//2)
    troughs = find_local_troughs(df['low'], window//2)
    
    if len(peaks) < 3 or len(troughs) < 3:
        return pattern
    
    for i in range(len(df.index) - window):
        end_idx = i + window
        if end_idx >= len(df):
            break
            
        # Get section of data
        section = df.iloc[i:end_idx]
        
        # Find resistance and support points
        section_peaks = [p for p in peaks if i <= p < end_idx]
        section_troughs = [t for t in troughs if i <= t < end_idx]
        
        if len(section_peaks) < 3 or len(section_troughs) < 3:
            continue
            
        # Get peak values and indices relative to section
        peak_values = [df['high'].iloc[p] for p in section_peaks]
        peak_indices = [p - i for p in section_peaks]
        
        # Get trough values and indices relative to section
        trough_values = [df['low'].iloc[t] for t in section_troughs]
        trough_indices = [t - i for t in section_troughs]
        
        # Fit lines to peaks and troughs
        try:
            peak_slope, peak_intercept = np.polyfit(peak_indices, peak_values, 1)
            trough_slope, trough_intercept = np.polyfit(trough_indices, trough_values, 1)
            
            # For triangle, slopes should have opposite signs
            if peak_slope * trough_slope < 0:
                # Calculate r-squared for both lines
                peak_fitted = [peak_slope * idx + peak_intercept for idx in peak_indices]
                trough_fitted = [trough_slope * idx + trough_intercept for idx in trough_indices]
                
                peak_ss_tot = sum((np.mean(peak_values) - peak_val) ** 2 for peak_val in peak_values)
                peak_ss_res = sum((peak_val - fit_val) ** 2 for peak_val, fit_val in zip(peak_values, peak_fitted))
                
                trough_ss_tot = sum((np.mean(trough_values) - trough_val) ** 2 for trough_val in trough_values)
                trough_ss_res = sum((trough_val - fit_val) ** 2 for trough_val, fit_val in zip(trough_values, trough_fitted))
                
                if peak_ss_tot > 0 and trough_ss_tot > 0:
                    peak_r2 = 1 - (peak_ss_res / peak_ss_tot)
                    trough_r2 = 1 - (trough_ss_res / trough_ss_tot)
                    
                    # Both lines should fit well
                    if peak_r2 > 0.7 and trough_r2 > 0.7:
                        # Calculate convergence point
                        x_intercept = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
                        
                        # Convergence point should be ahead but not too far
                        if x_intercept > window and x_intercept < 2 * window:
                            pattern.iloc[end_idx-5:end_idx] = True
        except:
            continue
            
    return pattern

def detect_flag(df, window=15, success_rate=0.67):
    """
    Detects Flag patterns (bullish/bearish)
    Success rate: 67%
    
    Parameters:
    - window: lookback window for detection
    """
    pattern = pd.Series(False, index=df.index)
    
    for i in range(window, len(df.index) - window):
        # Check for pole (strong prior move)
        pole_section = df.iloc[i-window:i]
        flag_section = df.iloc[i:i+window]
        
        # Pole should be a strong move
        pole_open = pole_section['open'].iloc[0]
        pole_close = pole_section['close'].iloc[-1]
        pole_change = pole_close - pole_open
        
        # Skip if pole is not significant
        if abs(pole_change) / pole_open < 0.05:
            continue
            
        # Flag should be a small consolidation
        flag_peaks = find_local_peaks(flag_section['high'], window//3)
        flag_troughs = find_local_troughs(flag_section['low'], window//3)
        
        if len(flag_peaks) < 2 or len(flag_troughs) < 2:
            continue
            
        # Get peak and trough values and indices
        peak_values = [flag_section['high'].iloc[p] for p in flag_peaks]
        peak_indices = [p for p in flag_peaks]
        
        trough_values = [flag_section['low'].iloc[t] for t in flag_troughs]
        trough_indices = [t for t in flag_troughs]
        
        # Fit lines to peaks and troughs
        try:
            peak_slope, peak_intercept = np.polyfit(peak_indices, peak_values, 1)
            trough_slope, trough_intercept = np.polyfit(trough_indices, trough_values, 1)
            
            # Slopes should be similar (parallel lines)
            if abs(peak_slope - trough_slope) / (abs(peak_slope) + 1e-10) < 0.3:
                # For bullish flag, pole is up and flag slopes down
                bullish_flag = pole_change > 0 and peak_slope < 0
                
                # For bearish flag, pole is down and flag slopes up
                bearish_flag = pole_change < 0 and peak_slope > 0
                
                # Calculate channel width
                channel_width = (peak_intercept - trough_intercept) / (trough_intercept + 1e-10)
                
                # Flag should be a small consolidation relative to pole
                if (bullish_flag or bearish_flag) and channel_width > 0.01 and channel_width < 0.05:
                    pattern.iloc[i+window-5:i+window] = True
        except:
            continue
            
    return pattern

def detect_all_patterns(df, window=20):
    """
    Detects all price action patterns and returns them in a DataFrame
    """
    results = pd.DataFrame(index=df.index)
    
    # Reversal patterns
    results['HeadAndShoulders'] = detect_head_and_shoulders(df, window)
    results['InvertedHeadAndShoulders'] = detect_inverted_head_and_shoulders(df, window)
    results['DoubleTop'] = detect_double_top(df, window)
    results['DoubleBottom'] = detect_double_bottom(df, window)
    results['TripleTop'] = detect_triple_top(df, window)
    results['TripleBottom'] = detect_triple_bottom(df, window)
    
    # Continuation patterns
    results['Rectangle'] = detect_rectangle(df, window)
    results['Channel'] = detect_channel(df, window)
    results['Triangle'] = detect_triangle(df, window)
    results['Flag'] = detect_flag(df, window)
    
    return results 