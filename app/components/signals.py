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

from signals.signal_functions import generate_standard_signals as generate_signals, SignalStrength
from signals.generator import create_default_signal_generator, generate_signals_advanced, analyze_exit_strategy
from signals.timeframes import TimeFrame, TimeFramePriority
from indicators import calculate_opening_range

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

def render_signals(data, signals=None, symbol=None):
    """
    Render a visualization of trading signals on a price chart
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        signals (pd.DataFrame, optional): DataFrame with signal data. If None, generates signals from data
        symbol (str, optional): Symbol being analyzed
    """
    # Use provided signals or generate them if not provided
    if signals is None and ('buy_signal' not in data.columns or 'sell_signal' not in data.columns):
        from app.signals import generate_signals
        signals = generate_signals(data)
    elif signals is None:
        signals = data
    
    # Display symbol if provided
    title = "Trading Signals"
    if symbol:
        title += f" - {symbol}"
    st.subheader(title)
    
    # Create plot with price data
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        )
    )
    
    # Calculate y-range for offsets
    y_range = data['high'].max() - data['low'].min()
    offset_percent = 0.005  # 0.5% offset between signal types
    
    # Add buy signals
    buy_signals = signals[signals['buy_signal'] == True] if 'buy_signal' in signals.columns else pd.DataFrame()
    if not buy_signals.empty:
        buy_y_values = []
        for idx in buy_signals.index:
            if idx in data.index:
                buy_y_values.append(data.loc[idx, 'close'] * (1 + offset_percent))
            elif 'close' in buy_signals.columns:
                buy_y_values.append(buy_signals.loc[idx, 'close'] * (1 + offset_percent))
    else:
        buy_y_values.append(None)
            
        fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_y_values,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Buy Signal'
                )
            )
    
    # Add sell signals
    sell_signals = signals[signals['sell_signal'] == True] if 'sell_signal' in signals.columns else pd.DataFrame()
    if not sell_signals.empty:
        sell_y_values = []
        for idx in sell_signals.index:
            if idx in data.index:
                sell_y_values.append(data.loc[idx, 'close'] * (1 - offset_percent))
            elif 'close' in sell_signals.columns:
                sell_y_values.append(sell_signals.loc[idx, 'close'] * (1 - offset_percent))
            else:
                sell_y_values.append(None)
                
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_y_values,
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                name='Sell Signal'
            )
        )
    
    # Add exit buy signals (exit long position)
    if 'exit_buy' in signals.columns:
        exit_buy_signals = signals[signals['exit_buy'] == True]
        if not exit_buy_signals.empty:
            exit_buy_y_values = []
            for idx in exit_buy_signals.index:
                if idx in data.index:
                    exit_buy_y_values.append(data.loc[idx, 'close'] * (1 + 2 * offset_percent))
                elif 'close' in exit_buy_signals.columns:
                    exit_buy_y_values.append(exit_buy_signals.loc[idx, 'close'] * (1 + 2 * offset_percent))
    else:
        exit_buy_y_values.append(None)
                    
        fig.add_trace(
                go.Scatter(
                    x=exit_buy_signals.index,
                    y=exit_buy_y_values,
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='orange',
                        line=dict(width=2, color='darkorange')
                    ),
                    name='Exit Long'
                )
            )
            
    # Add exit sell signals (exit short position)
    if 'exit_sell' in signals.columns:
        exit_sell_signals = signals[signals['exit_sell'] == True]
        if not exit_sell_signals.empty:
            exit_sell_y_values = []
            for idx in exit_sell_signals.index:
                if idx in data.index:
                    exit_sell_y_values.append(data.loc[idx, 'close'] * (1 - 2 * offset_percent))
                elif 'close' in exit_sell_signals.columns:
                    exit_sell_y_values.append(exit_sell_signals.loc[idx, 'close'] * (1 - 2 * offset_percent))
    else:
        exit_sell_y_values.append(None)
                    
        fig.add_trace(
                go.Scatter(
                    x=exit_sell_signals.index,
                    y=exit_sell_y_values,
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='purple',
                        line=dict(width=2, color='indigo')
                    ),
                    name='Exit Short'
                )
            )
    
    # Add filtered signals if available
    if 'filtered_buy_signal' in signals.columns:
        filtered_buy = signals[signals['filtered_buy_signal'] == True]
        if not filtered_buy.empty:
            filtered_buy_y_values = []
            for idx in filtered_buy.index:
                if idx in data.index:
                    filtered_buy_y_values.append(data.loc[idx, 'close'] * (1 + 3 * offset_percent))
                elif 'close' in filtered_buy.columns:
                    filtered_buy_y_values.append(filtered_buy.loc[idx, 'close'] * (1 + 3 * offset_percent))
    else:
        filtered_buy_y_values.append(None)
                    
        fig.add_trace(
                go.Scatter(
                    x=filtered_buy.index,
                    y=filtered_buy_y_values,
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='lime',
                        line=dict(width=2, color='green')
                    ),
                    name='Filtered Buy Signal'
                )
            )
    
    if 'filtered_sell_signal' in signals.columns:
        filtered_sell = signals[signals['filtered_sell_signal'] == True]
        if not filtered_sell.empty:
            filtered_sell_y_values = []
            for idx in filtered_sell.index:
                if idx in data.index:
                    filtered_sell_y_values.append(data.loc[idx, 'close'] * (1 - 3 * offset_percent))
                elif 'close' in filtered_sell.columns:
                    filtered_sell_y_values.append(filtered_sell.loc[idx, 'close'] * (1 - 3 * offset_percent))
                else:
                    filtered_sell_y_values.append(None)
                    
            fig.add_trace(
                go.Scatter(
                    x=filtered_sell.index,
                    y=filtered_sell_y_values,
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='pink',
                        line=dict(width=2, color='red')
                    ),
                    name='Filtered Sell Signal'
                )
            )
            
    # Add paired signals if available
    if 'paired_entry' in signals.columns and 'paired_exit' in signals.columns:
        # Get paired entry signals
        paired_entries = signals[signals['paired_entry'] == True]
        paired_exits = signals[signals['paired_exit'] == True]
        
        # Highlight paired buy entries
        paired_buy_entries = paired_entries[paired_entries['buy_signal'] == True]
        if not paired_buy_entries.empty:
            paired_buy_y_values = []
            for idx in paired_buy_entries.index:
                if idx in data.index:
                    paired_buy_y_values.append(data.loc[idx, 'close'] * (1 + 4 * offset_percent))
                elif 'close' in paired_buy_entries.columns:
                    paired_buy_y_values.append(paired_buy_entries.loc[idx, 'close'] * (1 + 4 * offset_percent))
            else:
                    paired_buy_y_values.append(None)
                    
            fig.add_trace(
                go.Scatter(
                    x=paired_buy_entries.index,
                    y=paired_buy_y_values,
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=18,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Paired Buy Entry'
                )
            )
        
        # Highlight paired sell entries
        paired_sell_entries = paired_entries[paired_entries['sell_signal'] == True]
        if not paired_sell_entries.empty:
            paired_sell_y_values = []
            for idx in paired_sell_entries.index:
                if idx in data.index:
                    paired_sell_y_values.append(data.loc[idx, 'close'] * (1 - 4 * offset_percent))
                elif 'close' in paired_sell_entries.columns:
                    paired_sell_y_values.append(paired_sell_entries.loc[idx, 'close'] * (1 - 4 * offset_percent))
                else:
                    paired_sell_y_values.append(None)
                    
            fig.add_trace(
                go.Scatter(
                    x=paired_sell_entries.index,
                    y=paired_sell_y_values,
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=18,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Paired Sell Entry'
                )
            )
            
        # Highlight paired exits
        paired_exit_buy = paired_exits[paired_exits['exit_buy'] == True]
        if not paired_exit_buy.empty:
            paired_exit_buy_y_values = []
            for idx in paired_exit_buy.index:
                if idx in data.index:
                    paired_exit_buy_y_values.append(data.loc[idx, 'close'] * (1 + 5 * offset_percent))
                elif 'close' in paired_exit_buy.columns:
                    paired_exit_buy_y_values.append(paired_exit_buy.loc[idx, 'close'] * (1 + 5 * offset_percent))
                else:
                    paired_exit_buy_y_values.append(None)
                    
            fig.add_trace(
                go.Scatter(
                    x=paired_exit_buy.index,
                    y=paired_exit_buy_y_values,
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=18,
                        color='orange',
                        line=dict(width=2, color='darkorange')
                    ),
                    name='Paired Exit Long'
                )
            )
            
        paired_exit_sell = paired_exits[paired_exits['exit_sell'] == True]
        if not paired_exit_sell.empty:
            paired_exit_sell_y_values = []
            for idx in paired_exit_sell.index:
                if idx in data.index:
                    paired_exit_sell_y_values.append(data.loc[idx, 'close'] * (1 - 5 * offset_percent))
                elif 'close' in paired_exit_sell.columns:
                    paired_exit_sell_y_values.append(paired_exit_sell.loc[idx, 'close'] * (1 - 5 * offset_percent))
                else:
                    paired_exit_sell_y_values.append(None)
                    
            fig.add_trace(
                go.Scatter(
                    x=paired_exit_sell.index,
                    y=paired_exit_sell_y_values,
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=18,
                        color='purple',
                        line=dict(width=2, color='indigo')
                    ),
                    name='Paired Exit Short'
                )
            )
            
        # Connect paired signals with lines if we have the necessary data
        try:
            # Try to extract or generate paired signal dataframe
            paired_signals = None
            
            # Check if we have paired signals in session state
            if hasattr(st.session_state, 'paired_results'):
                paired_results = st.session_state.paired_results
                if 'paired_signals' in paired_results and not paired_results['paired_signals'].empty:
                    paired_signals = paired_results['paired_signals']
            
            # If paired signals dataframe available, connect entry-exit points
            if paired_signals is not None and not paired_signals.empty:
                for _, pair in paired_signals.iterrows():
                    # Draw line connecting entry and exit
                    entry_time = pair['entry_time']
                    exit_time = pair['exit_time']
                    
                    if entry_time in data.index and exit_time in data.index:
                        entry_price = data.loc[entry_time, 'close']
                        exit_price = data.loc[exit_time, 'close']
                        
                        # Apply offset to entry/exit prices for line endpoints to match markers
                        if pair['type'] == 'long':
                            entry_price *= (1 + 4 * offset_percent)  # Match paired buy entry offset
                            exit_price *= (1 + 5 * offset_percent)   # Match paired exit long offset
                else:
                    entry_price *= (1 - 4 * offset_percent)  # Match paired sell entry offset
                    exit_price *= (1 - 5 * offset_percent)   # Match paired exit short offset
                        
                    fig.add_shape(
                            type="line",
                            x0=entry_time,
                            y0=entry_price,
                            x1=exit_time,
                            y1=exit_price,
                            line=dict(
                                color="green" if pair['type'] == 'long' else "red",
                                width=1,
                                dash="dot",
                            )
                        )
        except Exception as e:
            print(f"Warning: Could not add paired signal lines: {str(e)}")
    
    # Update layout
    chart_title = 'Price Chart with Trading Signals'
    if symbol:
        chart_title += f" - {symbol}"
    
    fig.update_layout(
        title=chart_title,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Create and display signal table with all signals in chronological order
    display_signal_table(data, signals)
    
    return signals

def display_signal_table(data, signals):
    """
    Display a table with all signals (buy, sell, exit) in chronological order
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        signals (pd.DataFrame): DataFrame with signal data
    """
    # Create a combined signal DataFrame
    signal_rows = []
    
    # Process buy signals
    if 'buy_signal' in signals.columns:
        buy_signals = signals[signals['buy_signal'] == True]
        for idx in buy_signals.index:
            signal_rows.append({
                'time': idx,
                'signal_type': 'BUY',
                'price': buy_signals.loc[idx, 'close'] if 'close' in buy_signals.columns else data.loc[idx, 'close'],
                'strength': buy_signals.loc[idx, 'buy_strength'] if 'buy_strength' in buy_signals.columns else '-',
                'score': buy_signals.loc[idx, 'buy_score'] if 'buy_score' in buy_signals.columns else '-',
                'filtered': buy_signals.loc[idx, 'filtered_buy_signal'] if 'filtered_buy_signal' in buy_signals.columns else False,
                'paired': buy_signals.loc[idx, 'paired_entry'] if 'paired_entry' in buy_signals.columns else False
            })
    
    # Process sell signals
    if 'sell_signal' in signals.columns:
        sell_signals = signals[signals['sell_signal'] == True]
        for idx in sell_signals.index:
            signal_rows.append({
                'time': idx,
                'signal_type': 'SELL',
                'price': sell_signals.loc[idx, 'close'] if 'close' in sell_signals.columns else data.loc[idx, 'close'],
                'strength': sell_signals.loc[idx, 'sell_strength'] if 'sell_strength' in sell_signals.columns else '-',
                'score': sell_signals.loc[idx, 'sell_score'] if 'sell_score' in sell_signals.columns else '-',
                'filtered': sell_signals.loc[idx, 'filtered_sell_signal'] if 'filtered_sell_signal' in sell_signals.columns else False,
                'paired': sell_signals.loc[idx, 'paired_entry'] if 'paired_entry' in sell_signals.columns else False
            })
    
    # Process exit buy signals (exit long)
    if 'exit_buy' in signals.columns:
        exit_buy_signals = signals[signals['exit_buy'] == True]
        for idx in exit_buy_signals.index:
            signal_rows.append({
                'time': idx,
                'signal_type': 'EXIT LONG',
                'price': exit_buy_signals.loc[idx, 'close'] if 'close' in exit_buy_signals.columns else data.loc[idx, 'close'],
                'strength': '-',
                'score': '-',
                'filtered': True,  # Exit signals are always considered filtered
                'paired': exit_buy_signals.loc[idx, 'paired_exit'] if 'paired_exit' in exit_buy_signals.columns else False
            })
    
    # Process exit sell signals (exit short)
    if 'exit_sell' in signals.columns:
        exit_sell_signals = signals[signals['exit_sell'] == True]
        for idx in exit_sell_signals.index:
            signal_rows.append({
                'time': idx,
                'signal_type': 'EXIT SHORT',
                'price': exit_sell_signals.loc[idx, 'close'] if 'close' in exit_sell_signals.columns else data.loc[idx, 'close'],
                'strength': '-',
                'score': '-',
                'filtered': True,  # Exit signals are always considered filtered
                'paired': exit_sell_signals.loc[idx, 'paired_exit'] if 'paired_exit' in exit_sell_signals.columns else False
            })
    
    # Sort by time
    if signal_rows:
        signal_df = pd.DataFrame(signal_rows)
        signal_df = signal_df.sort_values('time')
        
        # Format time column to Eastern Time
        eastern = pytz.timezone('US/Eastern')
        signal_df['time_formatted'] = [format_timestamp_as_et(t) for t in signal_df['time']]
        
        # Add market status
        signal_df['market_hours'] = [is_market_hours(t) for t in signal_df['time']]
        signal_df['market_status'] = ["Market Open" if mh else "Market Closed" for mh in signal_df['market_hours']]
        
        # Display the table
        st.subheader("Signal Table (Chronological Order)")
        
        # Color-code signal types
        def highlight_signal_type(val):
            if val == 'BUY':
                return 'background-color: rgba(0, 255, 0, 0.2)'  # Light green
            elif val == 'SELL':
                return 'background-color: rgba(255, 0, 0, 0.2)'  # Light red
            elif val == 'EXIT LONG':
                return 'background-color: rgba(255, 165, 0, 0.2)'  # Light orange
            elif val == 'EXIT SHORT':
                return 'background-color: rgba(128, 0, 128, 0.2)'  # Light purple
            return ''
        
        # Highlight paired signals
        def highlight_paired(val):
            if val == True:
                return 'background-color: rgba(0, 150, 255, 0.2)'  # Light blue background
            return ''
        
        # Apply styling
        styled_df = signal_df[['time_formatted', 'signal_type', 'price', 'strength', 'score', 'paired', 'market_status']].copy()
        styled_df = styled_df.style.applymap(highlight_signal_type, subset=['signal_type'])
        
        # Apply paired highlighting
        styled_df = styled_df.applymap(highlight_paired, subset=['paired'])
        
        # Set text alignment properties
        styled_df = styled_df.set_properties(**{'text-align': 'center'})
        
        # Define a custom formatter that handles both numeric and string values
        def safe_format(x, format_str):
            if isinstance(x, (int, float)):
                return format_str.format(x)
            return x
            
        # Apply formatting to numeric values only
        styled_df = styled_df.format({
            'price': lambda x: f"${safe_format(x, '{:.2f}')}" if pd.notnull(x) else "",
            'score': lambda x: safe_format(x, '{:.2f}') if pd.notnull(x) else "",
            'paired': lambda x: "✓" if x == True else ""
        })
        
        # Rename columns for display
        styled_df = styled_df.set_properties(subset=['paired'], **{'width': '60px'})
        
        # Display the table
        st.dataframe(styled_df, use_container_width=True)

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
    from signals.generator import analyze_exit_strategy
    
    # Display signals with our new comprehensive table
    display_signal_table(data, signals)
    
    return signals

def render_orb_signals(data, signals=None, symbol=None):
    """
    Render Opening Range Breakout (ORB) signals visualization
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        signals (pd.DataFrame, optional): DataFrame with signal data. If None, uses data or session state
        symbol (str, optional): Symbol being analyzed
    """
    # Use provided signals or get from other sources
    if signals is None:
        # Check if we have signals with ORB data in the data DataFrame
        if 'orb_signal' in data.columns:
            signals = data
        # Otherwise check if signals exist in session state
        elif hasattr(st.session_state, 'signals') and st.session_state.signals is not None:
            signals = st.session_state.signals
        else:
            st.warning("No Opening Range Breakout signals available")
            return None
    
    # Check if we have any ORB signals
    if 'orb_signal' not in signals.columns or signals['orb_signal'].sum() == 0:
        st.info("No Opening Range Breakout signals detected")
        return None
    
    # Display title with symbol if provided
    title = "Opening Range Breakout (ORB) Analysis"
    if symbol:
        title += f" - {symbol}"
    st.subheader(title)
    
    # Create a figure
    fig = go.Figure()
        
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        )
    )
    
    # Get ORB signals
    orb_signals = signals[signals['orb_signal'] == True]
    
    # Add ORB levels as horizontal lines
    for idx, row in orb_signals.iterrows():
        orb_level = row['orb_level']
        is_buy = row['buy_signal'] if 'buy_signal' in row else False
        color = 'green' if is_buy else 'red'
        
        # Add horizontal line for ORB level
        fig.add_shape(
            type="line",
            x0=data.index[0],
            y0=orb_level,
            x1=data.index[-1],
            y1=orb_level,
            line=dict(
                color=color,
                width=2,
                dash="dash",
            ),
            name=f"ORB {'Support' if is_buy else 'Resistance'}"
        )
        
        # Add annotation for ORB level
        fig.add_annotation(
            x=data.index[0],
            y=orb_level,
            text=f"ORB {'Support' if is_buy else 'Resistance'}: {orb_level:.2f}",
            showarrow=True,
            arrowhead=1,
            ax=-80,
            ay=0,
            font=dict(
                color=color
            )
        )
    
    # Add buy signals with ORB
    orb_buy_signals = orb_signals[orb_signals['buy_signal'] == True] if 'buy_signal' in orb_signals.columns else pd.DataFrame()
    if not orb_buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=orb_buy_signals.index,
                y=orb_buy_signals['close'] if 'close' in orb_buy_signals.columns else data.loc[orb_buy_signals.index, 'close'],
                mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                    size=15,
                            color='green',
                    line=dict(width=2, color='darkgreen')
                ),
                name='ORB Buy Signal'
            )
        )
    
    # Add sell signals with ORB
    orb_sell_signals = orb_signals[orb_signals['sell_signal'] == True] if 'sell_signal' in orb_signals.columns else pd.DataFrame()
    if not orb_sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=orb_sell_signals.index,
                y=orb_sell_signals['close'] if 'close' in orb_sell_signals.columns else data.loc[orb_sell_signals.index, 'close'],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=15,
                    color='red',
                    line=dict(width=2, color='darkred')
                ),
                name='ORB Sell Signal'
            )
        )
    
    # Update layout
    chart_title = 'Opening Range Breakout (ORB) Signals'
    if symbol:
        chart_title += f" - {symbol}"
        
    fig.update_layout(
        title=chart_title,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    return orb_signals

def render_advanced_signals(data, results=None, symbol=None):
    """
    Render advanced trading signals with market regime information
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        results (dict, optional): Dictionary with advanced analysis results or signals DataFrame
        symbol (str, optional): Symbol being analyzed
    """
    # Use provided results or try to get from session state or generate
    if results is None:
        if hasattr(st.session_state, 'advanced_results'):
            results = st.session_state.advanced_results
        elif hasattr(st.session_state, 'signals') and isinstance(st.session_state.signals, pd.DataFrame):
            # If results is a DataFrame (signals), wrap in a proper structure 
            signals = st.session_state.signals
            results = {'signals': signals}
        else:
            # Generate advanced signals
            with st.spinner("Generating advanced signals..."):
                try:
                    results = generate_signals_advanced(data)
                    st.session_state.advanced_results = results
                except Exception as e:
                    st.error(f"Error generating advanced signals: {str(e)}")
                    return None
    
    # If results is a DataFrame (signals), wrap in a proper structure
    if isinstance(results, pd.DataFrame):
        signals = results
        results = {'signals': signals}
    
    # Get filtered signals
    if 'signals' in results:
        signals = results['signals']
    else:
        st.warning("No advanced signals available")
        return None
    
    # Display title with symbol if provided
    title = "Advanced Signal Analysis"
    if symbol:
        title += f" - {symbol}"
    st.subheader(title)
    
    # Display market regime information
    if 'market_regime' in results:
        regime = results['market_regime'].replace('_', ' ').title()
        st.subheader(f"Market Regime: {regime}")
        
        # Determine market regime color
        regime_color = {
            'bull_trend': 'green',
            'bear_trend': 'red',
            'sideways': 'blue',
            'high_volatility': 'orange'
        }.get(results['market_regime'], 'gray')
        
        # Add regime description
        regime_descriptions = {
            'bull_trend': 'Strong upward momentum with higher highs and higher lows',
            'bear_trend': 'Strong downward momentum with lower highs and lower lows',
            'sideways': 'Price consolidation with limited directional movement',
            'high_volatility': 'Increased price swings with above-average volatility'
        }
        
        description = regime_descriptions.get(results['market_regime'], '')
        if description:
            st.markdown(f"<span style='color:{regime_color}'>{description}</span>", unsafe_allow_html=True)
    
    # Create plot with price data and enhanced signals
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        )
    )
    
    # Add buy signals
    buy_signals = signals[signals['buy_signal'] == True] if 'buy_signal' in signals.columns else pd.DataFrame()
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['close'] if 'close' in buy_signals.columns else data.loc[buy_signals.index, 'close'],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=15,
                    color='green',
                    line=dict(width=2, color='darkgreen')
                ),
                name='Buy Signal',
                hovertemplate='Buy: %{y:.2f}<br>Date: %{x}<extra></extra>'
            )
        )
    
    # Add sell signals
    sell_signals = signals[signals['sell_signal'] == True] if 'sell_signal' in signals.columns else pd.DataFrame()
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                        x=sell_signals.index,
                y=sell_signals['close'] if 'close' in sell_signals.columns else data.loc[sell_signals.index, 'close'],
                mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                    size=15,
                            color='red',
                    line=dict(width=2, color='darkred')
                ),
                name='Sell Signal',
                hovertemplate='Sell: %{y:.2f}<br>Date: %{x}<extra></extra>'
            )
        )
    
    # Update layout
    chart_title = 'Advanced Signals Analysis'
    if 'market_regime' in results:
        regime = results['market_regime'].replace('_', ' ').title()
        chart_title = f'Advanced Signals - {regime}'
    if symbol:
        chart_title += f" - {symbol}"
        
    fig.update_layout(
        title=chart_title,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display additional metrics if available
    if 'scores' in results:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Buy Score", f"{results['scores']['buy']:.2f}")
        with col2:
            st.metric("Sell Score", f"{results['scores']['sell']:.2f}")
    
    return signals

def render_exit_strategy_analysis(data, position=None):
    """
    Render exit strategy analysis for an open position
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        position (dict, optional): Dictionary with position information
            Must contain: 'type' ('long' or 'short'), 'entry_price', 'entry_time'
    """
    if position is None:
        # Check if position exists in session state
        if hasattr(st.session_state, 'current_position') and st.session_state.current_position is not None:
            position = st.session_state.current_position
        else:
            st.warning("No active position to analyze. Open a position first.")
            return None
    
    # Calculate exit strategy results
    with st.spinner("Analyzing exit strategy..."):
        try:
            exit_results = analyze_exit_strategy(data, position)
        except Exception as e:
            st.error(f"Error analyzing exit strategy: {str(e)}")
            traceback.print_exc()
            return None
    
    if not exit_results["success"]:
        st.error(f"Error analyzing exit strategy: {exit_results.get('error', 'Unknown error')}")
        return None
    
    # Display position information
    position_type = position["type"].upper()
    entry_price = position["entry_price"]
    entry_time = format_timestamp_as_et(position["entry_time"])
    current_price = data["close"].iloc[-1]
    
    # Calculate P&L
    if position_type == "LONG":
        pnl = (current_price - entry_price) / entry_price * 100
        pnl_color = "green" if pnl > 0 else "red"
    else:  # SHORT
        pnl = (entry_price - current_price) / entry_price * 100
        pnl_color = "green" if pnl > 0 else "red"
    
    # Position information
    st.subheader("Position Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Position Type", position_type)
    with col2:
        st.metric("Entry Price", f"${entry_price:.2f}")
    with col3:
        st.metric("Current Price", f"${current_price:.2f}")
    
    st.text(f"Entry Time: {entry_time}")
    
    # Position metrics
    st.subheader("Position Metrics")
    metrics = exit_results.get("metrics", {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**P&L**: <span style='color:{pnl_color}'>{pnl:.2f}%</span>", unsafe_allow_html=True)
                
        with col2:
            position_age = metrics.get("position_age_minutes", 0)
            st.metric("Position Age", f"{position_age:.1f} min")
    
    with col3:
        exit_prob = metrics.get("exit_probability", 0) * 100
        st.metric("Exit Probability", f"{exit_prob:.1f}%")
    
    # Exit recommendation
    action = exit_results.get("action", {})
    close_pct = action.get("close_percent", 0)
    exit_reason = action.get("reason", "unknown")
    
    # Show recommendation
    st.subheader("Exit Recommendation")
    if close_pct > 0:
        st.markdown(f"**Recommendation**: <span style='color:red'>EXIT position ({close_pct}%)</span>", unsafe_allow_html=True)
        st.markdown(f"**Reason**: {exit_reason.replace('_', ' ').title()}")
    else:
        st.markdown("**Recommendation**: <span style='color:green'>HOLD position</span>", unsafe_allow_html=True)
    
    # ATR stop level
    atr_stop = exit_results.get("atr_stop")
    if atr_stop is not None:
        atr_stop_distance = abs(current_price - atr_stop) / current_price * 100
        st.markdown(f"**ATR Stop Level**: ${atr_stop:.2f} ({atr_stop_distance:.2f}% away)")
    
    # Visualize exit signals
    st.subheader("Exit Signals Visualization")
    exit_signals = exit_results.get("signals")
    
    if exit_signals is not None and not exit_signals.empty:
        # Create figure
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
            name='Price'
            )
        )
        
        # Add entry point
        entry_idx = None
        if hasattr(position["entry_time"], "tzinfo"):
            # Find closest index to entry time
            entry_idx = data.index[data.index.get_indexer([position["entry_time"]], method='nearest')[0]]
        
        if entry_idx is not None:
            marker_color = "green" if position_type == "LONG" else "red"
            marker_symbol = "triangle-up" if position_type == "LONG" else "triangle-down"
            
            fig.add_trace(
                go.Scatter(
                    x=[entry_idx],
                    y=[entry_price],
                    mode='markers',
                    marker=dict(
                        symbol=marker_symbol,
                        size=15,
                        color=marker_color,
                        line=dict(width=2, color="black")
                    ),
                    name=f'{position_type} Entry'
                )
            )
        
        # Add ATR stop level
        if atr_stop is not None:
            fig.add_shape(
                type="line",
                x0=data.index[0],
                y0=atr_stop,
                x1=data.index[-1],
                y1=atr_stop,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                ),
                name="ATR Stop"
            )
            
            fig.add_annotation(
                x=data.index[-1],
                y=atr_stop,
                text=f"ATR Stop: {atr_stop:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=-80,
                ay=0
            )
        
        # Add exit signals
        if 'exit_buy' in exit_signals.columns and position_type == "LONG":
            exit_points = exit_signals[exit_signals['exit_buy'] == True]
            
            if not exit_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exit_points.index,
                        y=data.loc[exit_points.index, 'close'],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=10,
                            color='red',
                            line=dict(width=2, color='darkred')
                        ),
                        name='Exit Long Signal'
                    )
                )
        
        if 'exit_sell' in exit_signals.columns and position_type == "SHORT":
            exit_points = exit_signals[exit_signals['exit_sell'] == True]
            
            if not exit_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exit_points.index,
                        y=data.loc[exit_points.index, 'close'],
                    mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=10,
                            color='green',
                            line=dict(width=2, color='darkgreen')
                        ),
                        name='Exit Short Signal'
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f'Exit Strategy Analysis for {position_type} Position',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    return exit_results 