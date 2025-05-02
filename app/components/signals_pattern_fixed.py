import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict
import plotly.express as px
import traceback
import matplotlib.pyplot as plt

from app.signals import generate_signals, SignalStrength
from app.signals.generator import create_default_signal_generator, generate_signals_advanced, analyze_exit_strategy
from app.signals.timeframes import TimeFrame, TimeFramePriority
from app.indicators import calculate_opening_range

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

def format_timestamp_as_et(timestamp):
    """
    Format timestamp in Eastern Time with appropriate market hours indication
    
    Args:
        timestamp: Datetime object or index
        
    Returns:
        str: Formatted timestamp string in Eastern Time
    """
    # Handle None value
    if timestamp is None:
        return "N/A"
        
    if not hasattr(timestamp, 'tzinfo'):
        return str(timestamp)
    
    # Convert to Eastern Time
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=pytz.UTC)
    eastern = pytz.timezone('US/Eastern')
    et_time = timestamp.astimezone(eastern)
    
    # Format as string with ET indicator
    formatted = et_time.strftime('%Y-%m-%d %H:%M ET')
    
    # Add market status
    if is_market_hours(timestamp):
        return f"{formatted} (Market Open)"
    else:
        return f"{formatted} (Market Closed)"

def render_signal_table(data):
    """
    Render a table of trading signals with explanations
    
    Args:
        data (pd.DataFrame): DataFrame with price data
    """
    # Check if signals exist in session state first
    if 'signals' not in st.session_state or st.session_state.signals is None:
        # If not in session state, generate them
        from app.signals import generate_signals
        
        st.header("Trading Signals")
        
        # Generate signals from data
        with st.spinner("Generating signals..."):
            signals = generate_signals(data)
    else:
        # Use existing signals from session state
        signals = st.session_state.signals
        st.header("Trading Signals")
        
    # Check if we have advanced results available
    has_advanced = hasattr(st.session_state, 'advanced_results') and st.session_state.advanced_results is not None
    advanced_results = getattr(st.session_state, 'advanced_results', None) if has_advanced else None
    
    # Ensure signal prices match actual data prices for display
    if 'signal_price' in signals.columns:
        signal_indices = signals.index
        for idx in signal_indices:
            if idx in data.index and 'close' in data.columns:
                # Update signal price to match actual close price
                signals.loc[idx, 'signal_price'] = data.loc[idx, 'close']
    
    # Display basic information about signals
    buy_signals_count = signals['buy_signal'].astype(bool).sum() if 'buy_signal' in signals.columns else 0
    sell_signals_count = signals['sell_signal'].astype(bool).sum() if 'sell_signal' in signals.columns else 0
    exit_buy_count = signals['exit_buy'].astype(bool).sum() if 'exit_buy' in signals.columns else 0
    exit_sell_count = signals['exit_sell'].astype(bool).sum() if 'exit_sell' in signals.columns else 0
    total_signals = buy_signals_count + sell_signals_count + exit_buy_count + exit_sell_count
    
    if has_advanced:
        regime = advanced_results.get('market_regime', 'unknown').replace('_', ' ').title()
        st.info(f"Found {total_signals} signals ({buy_signals_count} buy, {sell_signals_count} sell, {exit_buy_count} exit long, {exit_sell_count} exit short) - Market Regime: {regime}")
    else:
        st.info(f"Found {total_signals} signals ({buy_signals_count} buy, {sell_signals_count} sell, {exit_buy_count} exit long, {exit_sell_count} exit short)")
    
    # Import exit strategy analyzer
    from app.signals.generator import analyze_exit_strategy
    
    # Create a list to store signal data
    signal_data = []
    entry_exit_pairs = [] 