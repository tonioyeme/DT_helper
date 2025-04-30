"""
UI components for displaying multi-timeframe analysis results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional

def render_multi_timeframe_analysis(data_dict: Dict[str, pd.DataFrame], 
                                   results: Dict[str, Any],
                                   symbol: str = None):
    """
    Render multi-timeframe analysis results in the Streamlit UI
    
    Args:
        data_dict: Dictionary of dataframes for each timeframe
        results: Results dictionary from the multi-timeframe analysis
        symbol: Trading symbol
    """
    if not results.get("success", False):
        st.error(f"Multi-timeframe analysis failed: {results.get('error', 'Unknown error')}")
        return
    
    # Show recommendation summary
    recommendation = results.get("recommendation", {})
    action = recommendation.get("action", "WAIT")
    
    # Use different colors for different actions
    if action == "BUY":
        color = "green"
    elif action == "SELL":
        color = "red"
    else:
        color = "gray"
    
    # Display the recommendation
    st.subheader("Trading Recommendation")
    
    # Create columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{action}</h1>", unsafe_allow_html=True)
    
    with col2:
        confidence = recommendation.get("confidence", 0) * 100
        st.metric("Confidence", f"{confidence:.1f}%")
    
    with col3:
        rr_ratio = recommendation.get("risk_reward", 0)
        st.metric("Risk/Reward", f"{rr_ratio:.2f}")
    
    # Show entry, target and stop levels
    if action != "WAIT":
        price_cols = st.columns(3)
        
        with price_cols[0]:
            st.metric("Entry Price", f"${recommendation.get('entry_price', 0):.2f}")
        
        with price_cols[1]:
            st.metric("Target", f"${recommendation.get('target', 0):.2f}")
        
        with price_cols[2]:
            st.metric("Stop Loss", f"${recommendation.get('stop', 0):.2f}")
    
    # Display timeframe analysis
    st.subheader("Timeframe Analysis")
    
    # Convert analysis results to a more display-friendly format
    analysis_data = []
    analysis_results = results.get("analysis", {})
    
    for tf_name, tf_analysis in analysis_results.items():
        if "error" in tf_analysis:
            continue
            
        # Convert trend enum to string
        trend = tf_analysis.get("trend", None)
        if hasattr(trend, "name"):
            trend_str = trend.name
        else:
            trend_str = "UNKNOWN"
            
        analysis_data.append({
            "Timeframe": tf_name,
            "Trend": trend_str,
            "Strength": tf_analysis.get("strength", 0),
            "Momentum": tf_analysis.get("momentum", 0),
            "Volume Ratio": tf_analysis.get("volume_ratio", 1.0)
        })
    
    # Convert to DataFrame for display
    if analysis_data:
        analysis_df = pd.DataFrame(analysis_data)
        
        # Color code the trends
        def color_trend(val):
            if val == "BULLISH":
                return 'background-color: green; color: white'
            elif val == "BEARISH":
                return 'background-color: red; color: white'
            else:
                return 'background-color: gray; color: white'
        
        # Apply styling
        styled_df = analysis_df.style.applymap(color_trend, subset=['Trend'])
        
        # Display the table
        st.dataframe(styled_df, use_container_width=True)
    
    # Show alignment information
    st.subheader("Timeframe Alignment")
    
    confirmation = results.get("recommendation", {})
    trend_aligned = confirmation.get("trend_aligned", False)
    momentum_confirmed = confirmation.get("momentum_confirmed", False)
    volume_confirmed = confirmation.get("volume_confirmed", False)
    
    alignment_cols = st.columns(3)
    
    with alignment_cols[0]:
        st.metric("Trend Alignment", "✅" if trend_aligned else "❌")
    
    with alignment_cols[1]:
        st.metric("Momentum Confirmation", "✅" if momentum_confirmed else "❌")
    
    with alignment_cols[2]:
        st.metric("Volume Confirmation", "✅" if volume_confirmed else "❌")
    
    # Visualize timeframes with plotly
    st.subheader("Multi-timeframe Chart")
    
    # Create separate charts for each timeframe
    timeframes = results.get("timeframes", [])
    
    if timeframes and len(timeframes) >= 3:
        # Sort timeframes by their typical duration
        timeframe_order = {
            "1m": 1, 
            "5m": 2, 
            "15m": 3, 
            "30m": 4, 
            "1h": 5,
            "4h": 6,
            "1d": 7
        }
        
        sorted_timeframes = sorted(timeframes, key=lambda x: timeframe_order.get(x, 99))
        
        # Limit to 3 timeframes for display
        display_timeframes = sorted_timeframes[:3]
        
        # Create subplots
        fig = make_subplots(
            rows=3, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[f"{tf.upper()} Timeframe" for tf in display_timeframes],
            row_heights=[0.40, 0.30, 0.30]
        )
        
        # Add candlesticks for each timeframe
        for i, tf in enumerate(display_timeframes):
            row = i + 1
            
            if tf not in data_dict:
                continue
                
            data = data_dict[tf]
            
            # Limit to last 100 candles for performance
            if len(data) > 100:
                data = data.iloc[-100:]
            
            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name=f"{tf} Price"
                ),
                row=row, col=1
            )
            
            # Add EMAs if available (using our analysis results)
            if "ema20" in data.columns and "ema50" in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['ema20'],
                        name=f"{tf} EMA20",
                        line=dict(color='blue', width=1)
                    ),
                    row=row, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['ema50'],
                        name=f"{tf} EMA50",
                        line=dict(color='orange', width=1)
                    ),
                    row=row, col=1
                )
            
            # Add volume as bar chart
            if 'volume' in data.columns:
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['volume'],
                        name=f"{tf} Volume",
                        marker=dict(color='rgba(0, 0, 255, 0.3)')
                    ),
                    row=row, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Multi-timeframe Analysis" if symbol else "Multi-timeframe Analysis",
            height=800,
            xaxis3_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            xaxis_rangeslider_visible=False
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation of multi-timeframe strategy
        with st.expander("About Multi-timeframe Trading"):
            st.markdown("""
            ## Multi-timeframe Trading Strategy
            
            **Three-Tier Timeframe Structure:**
            1. **Higher TF (15-min/1-hour):** Determines the primary trend direction
            2. **Middle TF (5-min):** Identifies trading opportunities
            3. **Lower TF (1-min):** Provides precise entry timing
            
            **Key Rules:**
            - Only take 5-min signals in the direction of the 1-hour trend
            - Use 1-min timeframe for entry refinement and stop placement
            - Confirm volume spikes across 2+ timeframes for stronger signals
            
            **Advantages:**
            - Reduces false signals by requiring alignment across timeframes
            - Improves timing of entries and exits
            - Provides better context for understanding price movements
            """)
    else:
        st.warning("Insufficient timeframes for multi-timeframe visualization. Need 3 timeframes.")
        
def render_multi_timeframe_signals(data_dict: Dict[str, pd.DataFrame], 
                                 results: Dict[str, Any],
                                 symbol: str = None):
    """
    Render a simplified version of the multi-timeframe signals in a compact UI
    for dashboard display.
    
    Args:
        data_dict: Dictionary of dataframes for each timeframe
        results: Results dictionary from the multi-timeframe analysis
        symbol: Trading symbol
    """
    if not results.get("success", False):
        st.warning("No multi-timeframe signals available")
        return
    
    # Get recommendation and signals
    recommendation = results.get("recommendation", {})
    action = recommendation.get("action", "WAIT")
    
    # Create display container
    st.subheader("Multi-timeframe Signal")
    
    # Action indicator
    if action == "BUY":
        st.success("🔼 BUY SIGNAL")
    elif action == "SELL":
        st.error("🔽 SELL SIGNAL")
    else:
        st.info("➡️ NEUTRAL")
    
    # Confidence and alignment indicators
    conf_col1, conf_col2, conf_col3 = st.columns(3)
    
    with conf_col1:
        confidence = recommendation.get("confidence", 0) * 100
        st.metric("Confidence", f"{confidence:.1f}%")
    
    with conf_col2:
        volume_confirmed = recommendation.get("volume_confirmed", False)
        st.metric("Volume", "✅" if volume_confirmed else "❌")
    
    with conf_col3:
        trend_aligned = recommendation.get("trend_aligned", False)
        st.metric("Alignment", "✅" if trend_aligned else "❌")
    
    # Display the most recent chart (signal timeframe)
    if "timeframes" in results and results["timeframes"]:
        timeframes = results["timeframes"]
        signal_tf = next((tf for tf in timeframes if tf == "5m"), timeframes[0])
        
        if signal_tf in data_dict:
            data = data_dict[signal_tf]
            
            # Limit to the most recent 30 candles
            if len(data) > 30:
                data = data.iloc[-30:]
            
            # Create a chart
            fig = go.Figure()
            
            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name="Price"
                )
            )
            
            # Add EMAs if available
            if "ema20" in data.columns and "ema50" in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['ema20'],
                        name="EMA20",
                        line=dict(color='blue', width=1)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['ema50'],
                        name="EMA50",
                        line=dict(color='orange', width=1)
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} {signal_tf}" if symbol else f"{signal_tf}",
                height=300,
                xaxis_rangeslider_visible=False
            )
            
            # Show the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Show entry/exit levels if available
            if action != "WAIT":
                levels_col1, levels_col2, levels_col3 = st.columns(3)
                
                with levels_col1:
                    st.metric("Entry", f"${recommendation.get('entry_price', 0):.2f}")
                
                with levels_col2:
                    st.metric("Target", f"${recommendation.get('target', 0):.2f}")
                
                with levels_col3:
                    st.metric("Stop", f"${recommendation.get('stop', 0):.2f}")
    
    # If there's still space, show a small table of timeframes and their trends
    analysis_results = results.get("analysis", {})
    
    if analysis_results:
        analysis_data = []
        
        for tf_name, tf_analysis in analysis_results.items():
            if "error" in tf_analysis:
                continue
                
            # Convert trend enum to string
            trend = tf_analysis.get("trend", None)
            if hasattr(trend, "name"):
                trend_str = trend.name
            else:
                trend_str = "UNKNOWN"
                
            analysis_data.append({
                "Timeframe": tf_name,
                "Trend": trend_str
            })
        
        if analysis_data:
            st.caption("Timeframe Trends:")
            st.dataframe(pd.DataFrame(analysis_data), use_container_width=True, hide_index=True) 