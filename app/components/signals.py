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
from app.signals.generator import create_default_signal_generator, generate_signals_advanced
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
    # If timestamp has no tzinfo, assume it's UTC and convert to Eastern
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
    
    # Check if during market hours (9:30 AM - 4:00 PM ET)
    market_open = time(9, 30)
    market_close = time(16, 0)
    current_time = timestamp.time()
    
    return market_open <= current_time <= market_close

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
    total_signals = buy_signals_count + sell_signals_count
    
    if has_advanced:
        regime = advanced_results.get('market_regime', 'unknown').replace('_', ' ').title()
        st.info(f"Found {total_signals} signals ({buy_signals_count} buy, {sell_signals_count} sell) - Market Regime: {regime}")
    else:
        st.info(f"Found {total_signals} signals ({buy_signals_count} buy, {sell_signals_count} sell)")
    
    # Get signals with actual buy or sell signals
    if 'buy_signal' in signals.columns and 'sell_signal' in signals.columns:
        buy_mask = signals['buy_signal'].astype(bool)
        sell_mask = signals['sell_signal'].astype(bool)
        valid_signals = signals[buy_mask | sell_mask]
    else:
        valid_signals = pd.DataFrame(index=signals.index)
    
    # If no valid signals were found after filtering, get more recent data
    if valid_signals.empty:
        recent_signals = signals.tail(10).copy()
    else:
        # Get the most recent signals (up to 20, with at least 5 signals if available)
        signal_count = min(max(5, valid_signals.shape[0]), 20)
        recent_signals = valid_signals.tail(signal_count).copy()
    
    # Display the signal table
    if recent_signals.empty:
        st.info("No signals generated for this data")
        return
        
    # Create a formatted table
    signal_data = []
    
    for idx, row in recent_signals.iterrows():
        try:
            # Format date/time in Eastern Time
            if hasattr(idx, 'tzinfo'):
                try:
                    # Convert to Eastern time if it's timezone-aware
                    eastern = pytz.timezone('US/Eastern')
                    date = idx.astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S ET')
                except:
                    date = str(idx)
            else:
                date = str(idx)
            
            # Determine signal type and explanation
            signal_type = ""
            explanation = ""
            target = ""
            stop_loss = ""
            strategy = ""
            
            if row['buy_signal']:
                signal_type = "Buy"
                price = data.loc[idx, 'close'] if idx in data.index else row.get('signal_price', 0)
                
                # Safely check for strategy indicators
                is_ema_vwap_bullish = row.get('ema_vwap_bullish', False) 
                is_mm_vol_bullish = row.get('mm_vol_bullish', False)
                is_price_cross_above_ema20 = row.get('price_cross_above_ema20', False)
                is_price_above_ema50 = row.get('price_above_ema50', False)
                is_macd_cross_above_signal = row.get('macd_cross_above_signal', False)
                is_ema_cloud_cross_bullish = row.get('ema_cloud_cross_bullish', False)
                is_stochastic_k_cross_above_d = row.get('stochastic_k_cross_above_d', False)
                is_stochastic_oversold = row.get('stochastic_oversold', False)
                
                # Determine if the signal is from a strategic combination
                if is_ema_vwap_bullish:
                    strategy = "EMA Cloud + VWAP"
                    explanation = "Price bounced off EMA cloud while above VWAP"
                elif is_mm_vol_bullish:
                    strategy = "Measured Move + Volume"
                    explanation = "Completed measured move pattern with strong volume"
                # Original signals logic
                elif is_price_cross_above_ema20 and is_price_above_ema50:
                    explanation = "Price crossed above EMA20 while above EMA50"
                elif is_macd_cross_above_signal:
                    explanation = "MACD crossed above signal line"
                elif is_ema_cloud_cross_bullish:
                    explanation = "EMA cloud turned bullish"
                elif is_stochastic_k_cross_above_d and is_stochastic_oversold:
                    explanation = "Stochastic K line crossed above D while oversold"
                else:
                    explanation = "Multiple indicators turned bullish"
                
                # Check if this is the latest signal and we have advanced results
                if has_advanced and idx == data.index[-1] and 'final_signal' in advanced_results:
                    final_signal = advanced_results['final_signal']
                    if final_signal.get('buy_signal', False):
                        explanation = "Advanced multi-timeframe analysis confirms bullish signal"
                        strategy = "Multi-Timeframe"
                
                # Add target price and stop loss if available
                target_price = row.get('target_price', np.nan)
                stop_loss_price = row.get('stop_loss', np.nan)
                
                if not pd.isna(target_price) and target_price is not None:
                    target = f"${target_price:.2f} ({((target_price / price) - 1) * 100:.1f}%)"
                
                if not pd.isna(stop_loss_price) and stop_loss_price is not None:
                    stop_loss = f"${stop_loss_price:.2f} ({((stop_loss_price / price) - 1) * 100:.1f}%)"
                    
            elif row['sell_signal']:
                signal_type = "Sell"
                price = data.loc[idx, 'close'] if idx in data.index else row.get('signal_price', 0)
                
                # Safely check for strategy indicators
                is_ema_vwap_bearish = row.get('ema_vwap_bearish', False)
                is_mm_vol_bearish = row.get('mm_vol_bearish', False)
                is_price_cross_below_ema20 = row.get('price_cross_below_ema20', False)
                is_price_above_ema50 = row.get('price_above_ema50', False)
                is_macd_cross_below_signal = row.get('macd_cross_below_signal', False)
                is_ema_cloud_cross_bearish = row.get('ema_cloud_cross_bearish', False)
                is_stochastic_k_cross_below_d = row.get('stochastic_k_cross_below_d', False)
                is_stochastic_overbought = row.get('stochastic_overbought', False)
                
                # Determine if the signal is from a strategic combination
                if is_ema_vwap_bearish:
                    strategy = "EMA Cloud + VWAP"
                    explanation = "Price bounced down from EMA cloud while below VWAP"
                elif is_mm_vol_bearish:
                    strategy = "Measured Move + Volume"
                    explanation = "Completed measured move pattern with strong volume"
                # Original signals logic
                elif is_price_cross_below_ema20 and not is_price_above_ema50:
                    explanation = "Price crossed below EMA20 while below EMA50"
                elif is_macd_cross_below_signal:
                    explanation = "MACD crossed below signal line"
                elif is_ema_cloud_cross_bearish:
                    explanation = "EMA cloud turned bearish"
                elif is_stochastic_k_cross_below_d and is_stochastic_overbought:
                    explanation = "Stochastic K line crossed below D while overbought"
                else:
                    explanation = "Multiple indicators turned bearish"
                
                # Check if this is the latest signal and we have advanced results
                if has_advanced and idx == data.index[-1] and 'final_signal' in advanced_results:
                    final_signal = advanced_results['final_signal']
                    if final_signal.get('sell_signal', False):
                        explanation = "Advanced multi-timeframe analysis confirms bearish signal"
                        strategy = "Multi-Timeframe"
                
                # Add target price and stop loss if available
                target_price = row.get('target_price', np.nan)
                stop_loss_price = row.get('stop_loss', np.nan)
                
                if not pd.isna(target_price) and target_price is not None:
                    target = f"${target_price:.2f} ({((target_price / price) - 1) * 100:.1f}%)"
                
                if not pd.isna(stop_loss_price) and stop_loss_price is not None:
                    stop_loss = f"${stop_loss_price:.2f} ({((stop_loss_price / price) - 1) * 100:.1f}%)"
            
            # Add signal counts for strength
            signal_count = 0
            if signal_type == "Buy":
                signal_count = row.get('buy_count', 0)
            elif signal_type == "Sell":
                signal_count = row.get('sell_count', 0)
                
            # Determine signal strength text
            signal_strength = 1
            
            # Get signal strength with proper error handling
            if 'signal_strength' in row and not pd.isna(row['signal_strength']):
                try:
                    signal_strength = int(row['signal_strength'])
                except:
                    pass
            elif signal_type == "Buy" and 'buy_strength' in row and not pd.isna(row['buy_strength']):
                try:
                    signal_strength = int(row['buy_strength'])
                except:
                    pass
            elif signal_type == "Sell" and 'sell_strength' in row and not pd.isna(row['sell_strength']):
                try:
                    signal_strength = int(row['sell_strength'])
                except:
                    pass
                    
            # Ensure strength is within range
            if signal_strength < 1:
                signal_strength = 1
            elif signal_strength > 4:
                signal_strength = 4
                
            # Map strength to text
            strength_map = {
                1: "Weak",
                2: "Moderate", 
                3: "Strong",
                4: "Very Strong"
            }
            strength = strength_map.get(signal_strength, "Weak")
                
            # Format as string for display
            confidence_text = ""
            if has_advanced and idx == data.index[-1] and 'final_signal' in advanced_results:
                confidence = advanced_results['final_signal'].get('confidence', None)
                if confidence is not None:
                    confidence_text = f" (Confidence: {int(confidence * 100)}%)"
                    
            if signal_count > 0:
                strength_text = f"{strength} ({signal_count} indicators){confidence_text}"
            else:
                strength_text = f"{strength}{confidence_text}"
            
            # Add to data table
            if signal_type:
                signal_data.append({
                    "Date": date,
                    "Signal": signal_type,
                    "Strategy": strategy if strategy else "Multi-Indicator",
                    "Price": f"${price:.2f}" if isinstance(price, (int, float)) else "N/A",
                    "Strength": strength_text,
                    "Target": target,
                    "Stop Loss": stop_loss,
                    "Explanation": explanation
                })
        except Exception as e:
            st.error(f"Error processing signal at {idx}: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            continue
    
    # Create DataFrame for display
    if signal_data:
        signal_df = pd.DataFrame(signal_data)
        
        # Highlight strong signals
        def highlight_strong_signals(row):
            if "Strong" in row['Strength'] or "VERY_STRONG" in row['Strength']:
                return ['background-color: #c6f6c6; font-weight: bold; color: #006400'] * len(row)
            return [''] * len(row)
        
        # Display styled dataframe
        st.dataframe(
            signal_df.style.apply(highlight_strong_signals, axis=1),
            use_container_width=True
        )
    else:
        st.warning("No valid signals could be processed from the data.")

def render_signals(data, signal_data=None, symbol=None):
    """
    Render signal data with visualization and detailed information
    
    Args:
        data: DataFrame with price data
        signal_data: Optional pre-calculated signal data (if None, will be calculated)
        symbol: Trading symbol being analyzed (optional)
    """
    st.header("Signal Analysis")
    
    if symbol:
        st.subheader(f"Symbol: {symbol}")
    
    if signal_data is None:
        if data is not None:
            signal_data = generate_signals(data)
        else:
            st.warning("No data available for signal generation")
            return
    
    # Check if we have signals
    if signal_data is None or not isinstance(signal_data, pd.DataFrame) or len(signal_data) == 0:
        st.warning("No signal data available")
        return
        
    # Filter to rows with actual signals - do this safely to avoid DataFrame boolean issues
    buy_signals_mask = signal_data['buy_signal'].astype(bool) if 'buy_signal' in signal_data.columns else pd.Series(False, index=signal_data.index)
    sell_signals_mask = signal_data['sell_signal'].astype(bool) if 'sell_signal' in signal_data.columns else pd.Series(False, index=signal_data.index)
    
    # Combine buy and sell signals
    all_signals = signal_data[buy_signals_mask | sell_signals_mask]
    
    # Create a price chart with signals marked
    st.subheader("Price Chart with Signals")
    
    try:
        # Create candlestick chart with signal points
        fig = go.Figure()
        
        # Add candlestick trace
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
        
        # Add signals as scatter points
        for idx, row in all_signals.iterrows():
            if idx not in data.index:
                continue
                
            if row.get('buy_signal', False):
                # Add buy signal marker
                fig.add_trace(
                    go.Scatter(
                        x=[idx],
                        y=[data.loc[idx, 'low'] * 0.995],  # Place slightly below
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green',
                        ),
                        name='Buy Signal',
                        hoverinfo='text',
                        text=f"Buy Signal: ${data.loc[idx, 'close']:.2f}"
                    )
                )
            
            if row.get('sell_signal', False):
                # Add sell signal marker
                fig.add_trace(
                    go.Scatter(
                        x=[idx],
                        y=[data.loc[idx, 'high'] * 1.005],  # Place slightly above
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red',
                        ),
                        name='Sell Signal',
                        hoverinfo='text',
                        text=f"Sell Signal: ${data.loc[idx, 'close']:.2f}"
                    )
                )
        
        # Set layout
        fig.update_layout(
            title=f"{symbol} Price with Signals",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        # Display plot
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering signal chart: {str(e)}")
        import traceback
        st.write(traceback.format_exc())

def render_orb_signals(data, signals, symbol):
    """
    Render Opening Range Breakout signals visualization
    
    Args:
        data (pd.DataFrame): Price data
        signals (pd.DataFrame): Signals data
        symbol (str): Trading symbol
    """
    st.subheader(f"Opening Range Breakout Analysis: {symbol}")
    
    # Debug check for signals
    st.write("Checking ORB signal data structure:")
    
    if 'orb_signal' not in signals.columns:
        st.warning("No 'orb_signal' column found in signals DataFrame")
        return
    
    orb_signal_count = signals['orb_signal'].sum() if 'orb_signal' in signals.columns else 0
    st.write(f"Found {orb_signal_count} ORB signals in data")
    
    # Check if we have any ORB signals
    if orb_signal_count == 0:
        st.warning("No Opening Range Breakout signals detected in this data.")
        return
    
    # Filter to get only the rows with ORB signals
    orb_data = signals[signals['orb_signal'] == True]
    
    # Try to calculate ORB levels directly to make sure we can find them
    from app.indicators import calculate_opening_range
    
    # Ensure we have data in Eastern time
    eastern = pytz.timezone('US/Eastern')
    chart_data = data.copy()
    if chart_data.index.tzinfo is not None:
        chart_data.index = chart_data.index.tz_convert(eastern)
    
    # Calculate opening range
    orb_high, orb_low = calculate_opening_range(chart_data, minutes=5)
    
    # Create a simplified ORB chart first to verify it works
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=chart_data.index,
        open=chart_data['open'],
        high=chart_data['high'],
        low=chart_data['low'],
        close=chart_data['close'],
        name='Price'
    ))
    
    # Add ORB levels if available
    if orb_high is not None:
        fig.add_shape(
            type="line",
            x0=chart_data.index[0],
            y0=orb_high,
            x1=chart_data.index[-1],
            y1=orb_high,
            line=dict(color="green", width=2, dash="dash"),
            name="ORB High"
        )
        
    if orb_low is not None:
        fig.add_shape(
            type="line",
            x0=chart_data.index[0],
            y0=orb_low,
            x1=chart_data.index[-1],
            y1=orb_low,
            line=dict(color="red", width=2, dash="dash"),
            name="ORB Low"
        )
    
    # Set chart title and labels
    fig.update_layout(
        title=f"Opening Range Breakout: {symbol}",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_advanced_signals(tf_data_dict, view_mode="primary", symbol=None):
    """
    Render advanced signal analysis with multi-timeframe data
    
    Args:
        tf_data_dict (dict): Dictionary containing multi-timeframe data
        view_mode (str): View mode, either "primary" or "all"
        symbol (str): Trading symbol being analyzed
    """
    if not tf_data_dict or not isinstance(tf_data_dict, dict):
        st.warning("No multi-timeframe data available")
        return
        
    st.header("Advanced Signal Analysis")
    
    if symbol:
        st.subheader(f"Symbol: {symbol}")
    
    # Display the weighted signals section if available
    if 'weighted_signals' in tf_data_dict:
        weighted_signals = tf_data_dict['weighted_signals']
        
        # Create a table of timeframe signals
        st.subheader("Multi-Timeframe Analysis")
        
        tf_data_rows = []
        for tf_name, tf_data in weighted_signals.items():
            signals_data = tf_data['signals']
            weight = tf_data['weight']
            
            # Extract signal direction
            signal_direction = "Neutral"
            buy_signal = signals_data.get('buy_signal', False)
            sell_signal = signals_data.get('sell_signal', False)
            
            # Make sure we're evaluating the signals safely
            if isinstance(buy_signal, bool) and buy_signal:
                signal_direction = "Buy"
            elif isinstance(sell_signal, bool) and sell_signal:
                signal_direction = "Sell"
            # Handle DataFrame/Series cases explicitly
            elif not isinstance(buy_signal, bool) and not isinstance(sell_signal, bool):
                # Convert to bool if it's a DataFrame/Series value
                if hasattr(buy_signal, 'item') and buy_signal:
                    signal_direction = "Buy"
                elif hasattr(sell_signal, 'item') and sell_signal:
                    signal_direction = "Sell"
            
            # Extract score (use appropriate field or fallback to default)
            score = signals_data.get('buy_score', 0) if signal_direction == "Buy" else signals_data.get('sell_score', 0) if signal_direction == "Sell" else 0
            
            tf_data_rows.append({
                "Timeframe": tf_name,
                "Signal": signal_direction,
                "Score": f"{int(score * 100)}%" if signal_direction != "Neutral" else "-",
                "Weight": f"{weight:.1f}"
            })
        
        # Ensure we have data before creating a DataFrame
        if len(tf_data_rows) > 0:
            tf_df = pd.DataFrame(tf_data_rows)
            
            # Add styling to the dataframe
            def highlight_signal(val):
                color = 'white'
                if val == 'Buy':
                    color = '#d4f7d4'  # Light green
                elif val == 'Sell':
                    color = '#f7d4d4'  # Light red
                return f'background-color: {color}'
            
            st.dataframe(
                tf_df.style.applymap(highlight_signal, subset=['Signal']),
                use_container_width=True
            )
    
    # If there's a final signal, display it prominently
    if 'final_signal' in tf_data_dict:
        final_signal = tf_data_dict['final_signal']
        
        col1, col2 = st.columns(2)
        
        with col1:
            signal_type = "BUY" if final_signal.get('buy_signal', False) else "SELL" if final_signal.get('sell_signal', False) else "NEUTRAL"
            signal_strength = final_signal.get('signal_strength', 1)
            
            # Convert signal strength to text
            strength_map = {1: "WEAK", 2: "MODERATE", 3: "STRONG", 4: "VERY STRONG"}
            strength_text = strength_map.get(signal_strength, "Unknown")
            
            # Display signal with appropriate styling
            if signal_type == "BUY":
                st.markdown(f"""
                <div style="background-color: #d4f7d4; padding: 10px; border-radius: 5px; border: 1px solid #28a745;">
                    <h2 style="color: #28a745; margin: 0;">BUY SIGNAL ({strength_text})</h2>
                </div>
                """, unsafe_allow_html=True)
            elif signal_type == "SELL":
                st.markdown(f"""
                <div style="background-color: #f7d4d4; padding: 10px; border-radius: 5px; border: 1px solid #dc3545;">
                    <h2 style="color: #dc3545; margin: 0;">SELL SIGNAL ({strength_text})</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #e9ecef; padding: 10px; border-radius: 5px; border: 1px solid #6c757d;">
                    <h2 style="color: #6c757d; margin: 0;">NEUTRAL</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Display confidence score if available
            confidence = final_signal.get('confidence', None)
            if confidence is not None:
                st.metric("Signal Confidence", f"{int(confidence * 100)}%")
                
            # Display other metrics if available
            score = final_signal.get('buy_score', 0) if signal_type == "BUY" else final_signal.get('sell_score', 0) if signal_type == "SELL" else 0
            if score:
                st.metric("Signal Score", f"{int(score * 100)}%") 