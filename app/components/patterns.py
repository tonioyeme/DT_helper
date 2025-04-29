import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from app.patterns.price_action import detect_all_patterns


def render_patterns_section(data):
    """
    Render a section showing detected price action patterns
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC and volume data
    """
    if data is None or len(data) == 0:
        st.warning("No data to analyze for patterns")
        return
    
    st.subheader("Price Action Patterns")
    
    # Set window size for pattern detection
    window = st.slider("Pattern detection window", 10, 50, 20, 
                     help="Window size for pattern detection. Larger windows may detect larger patterns.")
    
    # Detect all patterns
    patterns = detect_all_patterns(data, window)
    
    # Check if any patterns were detected
    any_patterns = False
    for col in patterns.columns:
        if patterns[col].any():
            any_patterns = True
            break
    
    if not any_patterns:
        st.info("No price action patterns detected in the current data")
        return
    
    # Group patterns by type
    reversal_patterns = ['HeadAndShoulders', 'InvertedHeadAndShoulders', 
                         'DoubleTop', 'DoubleBottom', 
                         'TripleTop', 'TripleBottom']
    
    continuation_patterns = ['Rectangle', 'Channel', 'Triangle', 'Flag']
    
    # Create tabs for different pattern types
    tab1, tab2 = st.tabs(["Reversal Patterns", "Continuation Patterns"])
    
    with tab1:
        _render_pattern_group(data, patterns, reversal_patterns)
        
    with tab2:
        _render_pattern_group(data, patterns, continuation_patterns)
    
    # Show most recent patterns
    st.subheader("Recent Patterns")
    
    # Get the last 10 candles
    recent_data = data.iloc[-10:]
    recent_patterns = []
    
    for col in patterns.columns:
        recent_pattern = patterns[col].iloc[-10:]
        if recent_pattern.any():
            dates = recent_data.index[recent_pattern].strftime('%Y-%m-%d %H:%M')
            for date in dates:
                recent_patterns.append({
                    'Date': date,
                    'Pattern': col,
                    'Type': 'Reversal' if col in reversal_patterns else 'Continuation',
                    'Bias': _get_pattern_bias(col)
                })
    
    if recent_patterns:
        recent_df = pd.DataFrame(recent_patterns)
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No patterns detected in the most recent data")


def _render_pattern_group(data, patterns, pattern_list):
    """Render a group of patterns"""
    # Count patterns by type
    pattern_counts = {}
    for pattern in pattern_list:
        if pattern in patterns.columns:
            count = patterns[pattern].sum()
            if count > 0:
                pattern_counts[pattern] = int(count)
    
    if not pattern_counts:
        st.info(f"No {pattern_list[0].split()[0]} patterns detected")
        return
    
    # Display pattern counts
    cols = st.columns(len(pattern_counts))
    for i, (pattern, count) in enumerate(pattern_counts.items()):
        with cols[i]:
            bias = _get_pattern_bias(pattern)
            color = "green" if bias == "Bullish" else "red" if bias == "Bearish" else "blue"
            st.metric(f"{pattern} ({bias})", count)
    
    # Create expanders for each pattern
    for pattern in pattern_list:
        if pattern in patterns.columns and patterns[pattern].sum() > 0:
            with st.expander(f"{pattern} Details ({int(patterns[pattern].sum())} occurrences)"):
                # Get dates where pattern occurred
                pattern_dates = data.index[patterns[pattern]]
                
                # Create a candlestick chart highlighting the pattern
                if len(pattern_dates) > 0:
                    # Get data around the most recent pattern
                    most_recent = pattern_dates[-1]
                    idx = data.index.get_loc(most_recent)
                    
                    # Get data from 10 candles before to 5 candles after the pattern
                    start_idx = max(0, idx - 10)
                    end_idx = min(len(data), idx + 5)
                    chart_data = data.iloc[start_idx:end_idx]
                    
                    # Create candlestick chart
                    fig = go.Figure()
                    
                    # Add candlestick
                    fig.add_trace(
                        go.Candlestick(
                            x=chart_data.index,
                            open=chart_data['open'],
                            high=chart_data['high'],
                            low=chart_data['low'],
                            close=chart_data['close'],
                            name='Price'
                        )
                    )
                    
                    # Add marker for pattern
                    pattern_data = chart_data.loc[chart_data.index.isin(pattern_dates)]
                    if not pattern_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=pattern_data.index,
                                y=pattern_data['high'] + (pattern_data['high'] * 0.005),  # Slightly above high
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-down',
                                    size=12,
                                    color='green' if 'Bullish' in pattern or 'Bottom' in pattern or 'Inverse' in pattern else 'red',
                                ),
                                name=pattern
                            )
                        )
                    
                    # Update layout
                    fig.update_layout(
                        title=f'Most recent {pattern} pattern',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        height=400,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show a list of dates when the pattern occurred
                    st.subheader(f"{pattern} occurrences")
                    date_df = pd.DataFrame({
                        'Date': pattern_dates.strftime('%Y-%m-%d %H:%M')
                    })
                    st.dataframe(date_df, use_container_width=True)


def _get_pattern_bias(pattern):
    """Get the bias (bullish/bearish) of a pattern"""
    bullish_patterns = ['InvertedHeadAndShoulders', 'DoubleBottom', 'TripleBottom']
    bearish_patterns = ['HeadAndShoulders', 'DoubleTop', 'TripleTop']
    
    if pattern in bullish_patterns:
        return "Bullish"
    elif pattern in bearish_patterns:
        return "Bearish"
    else:
        return "Neutral" 