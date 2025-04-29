import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict
import plotly.express as px

from app.signals import generate_signals, SignalStrength
from app.signals.generator import create_default_signal_generator, generate_signals_advanced
from app.signals.timeframes import TimeFrame, TimeFramePriority

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
    from app.signals import generate_signals
    
    st.header("Trading Signals")
    
    # Generate signals from data
    with st.spinner("Generating signals..."):
        signals = generate_signals(data)
    
    # Display basic information about signals
    buy_signals = signals['buy_signal'].sum()
    sell_signals = signals['sell_signal'].sum()
    total_signals = buy_signals + sell_signals
    
    st.info(f"Found {total_signals} signals ({buy_signals} buy, {sell_signals} sell)")
    
    # Get signals with actual buy or sell signals
    valid_signals = signals[signals['buy_signal'] | signals['sell_signal']]
    
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
            date = format_timestamp_as_et(idx) if hasattr(idx, 'tzinfo') else str(idx)
            
            # Determine signal type and explanation
            signal_type = ""
            explanation = ""
            target = ""
            stop_loss = ""
            strategy = ""
            
            if row['buy_signal']:
                signal_type = "Buy"
                price = data.loc[idx, 'close']
                
                # Safely check for strategy indicators
                is_ema_vwap_bullish = row.get('ema_vwap_bullish', False) if 'ema_vwap_bullish' in row else False
                is_mm_vol_bullish = row.get('mm_vol_bullish', False) if 'mm_vol_bullish' in row else False
                is_price_cross_above_ema20 = row.get('price_cross_above_ema20', False) if 'price_cross_above_ema20' in row else False
                is_price_above_ema50 = row.get('price_above_ema50', False) if 'price_above_ema50' in row else False
                is_macd_cross_above_signal = row.get('macd_cross_above_signal', False) if 'macd_cross_above_signal' in row else False
                is_ema_cloud_cross_bullish = row.get('ema_cloud_cross_bullish', False) if 'ema_cloud_cross_bullish' in row else False
                is_stochastic_k_cross_above_d = row.get('stochastic_k_cross_above_d', False) if 'stochastic_k_cross_above_d' in row else False
                is_stochastic_oversold = row.get('stochastic_oversold', False) if 'stochastic_oversold' in row else False
                
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
                
                # Add target price and stop loss if available
                target_price = row.get('target_price', np.nan)
                stop_loss_price = row.get('stop_loss', np.nan)
                
                if not pd.isna(target_price):
                    target = f"${target_price:.2f} ({((target_price / price) - 1) * 100:.1f}%)"
                
                if not pd.isna(stop_loss_price):
                    stop_loss = f"${stop_loss_price:.2f} ({((stop_loss_price / price) - 1) * 100:.1f}%)"
                    
            elif row['sell_signal']:
                signal_type = "Sell"
                price = data.loc[idx, 'close']
                
                # Safely check for strategy indicators
                is_ema_vwap_bearish = row.get('ema_vwap_bearish', False) if 'ema_vwap_bearish' in row else False
                is_mm_vol_bearish = row.get('mm_vol_bearish', False) if 'mm_vol_bearish' in row else False
                is_price_cross_below_ema20 = row.get('price_cross_below_ema20', False) if 'price_cross_below_ema20' in row else False
                is_price_above_ema50 = row.get('price_above_ema50', False) if 'price_above_ema50' in row else False
                is_macd_cross_below_signal = row.get('macd_cross_below_signal', False) if 'macd_cross_below_signal' in row else False
                is_ema_cloud_cross_bearish = row.get('ema_cloud_cross_bearish', False) if 'ema_cloud_cross_bearish' in row else False
                is_stochastic_k_cross_below_d = row.get('stochastic_k_cross_below_d', False) if 'stochastic_k_cross_below_d' in row else False
                is_stochastic_overbought = row.get('stochastic_overbought', False) if 'stochastic_overbought' in row else False
                
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
                
                # Add target price and stop loss if available
                target_price = row.get('target_price', np.nan)
                stop_loss_price = row.get('stop_loss', np.nan)
                
                if not pd.isna(target_price):
                    target = f"${target_price:.2f} ({((target_price / price) - 1) * 100:.1f}%)"
                
                if not pd.isna(stop_loss_price):
                    stop_loss = f"${stop_loss_price:.2f} ({((stop_loss_price / price) - 1) * 100:.1f}%)"
            
            # Add signal counts for strength
            signal_count = 0
            if signal_type == "Buy":
                signal_count = row.get('buy_count', 0)
            elif signal_type == "Sell":
                signal_count = row.get('sell_count', 0)
                
            # Determine signal strength text
            signal_strength = row.get('signal_strength', 1)
            if signal_strength == 4:
                strength = "Very Strong"
            elif signal_strength == 3:
                strength = "Strong"
            elif signal_strength == 2:
                strength = "Moderate"
            else:
                strength = "Weak"
                
            # Format as string for display
            strength_text = f"{strength} ({signal_count} indicators)"
            
            # Add to data table
            if signal_type:
                signal_data.append({
                    "Date": date,
                    "Signal": signal_type,
                    "Strategy": strategy if strategy else "Multi-Indicator",
                    "Price": f"${data.loc[idx, 'close']:.2f}",
                    "Strength": strength_text,
                    "Target": target,
                    "Stop Loss": stop_loss,
                    "Explanation": explanation
                })
        except Exception as e:
            st.error(f"Error processing signal at {idx}: {str(e)}")
            continue
    
    # Create DataFrame for display
    if signal_data:
        signal_df = pd.DataFrame(signal_data)
        
        # Highlight strong signals
        def highlight_strong_signals(row):
            if "Strong" in row['Strength'] or "VERY_STRONG" in row['Strength']:
                return ['background-color: #c6f6c6; font-weight: bold; color: #006400'] * len(row)
            return [''] * len(row)
        
        # Apply styling
        styled_df = signal_df.style.apply(highlight_strong_signals, axis=1)
        
        st.dataframe(styled_df)
        
        # Add a download button for signals
        csv = signal_df.to_csv(index=False)
        st.download_button(
            label="Download Signals CSV",
            data=csv,
            file_name="trading_signals.csv",
            mime="text/csv"
        )
    else:
        st.warning("No trading signals could be displayed. There might be an issue with the data format or signal generation. Check the logs for errors.")
        
    # Display signals statistics
    st.subheader("Signal Statistics")
    
    # Use all signals for statistics
    signals_for_stats = signals
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        buy_count = signals_for_stats['buy_signal'].sum() if not signals_for_stats.empty else 0
        sell_count = signals_for_stats['sell_signal'].sum() if not signals_for_stats.empty else 0
        total_signals = buy_count + sell_count
        
        st.metric("Total Signals", total_signals)
        st.metric("Buy Signals", buy_count)
        st.metric("Sell Signals", sell_count)
        
    with col2:
        # Make sure signal_strength column exists
        if 'signal_strength' in signals_for_stats.columns and not signals_for_stats.empty:
            strong_buy = (signals_for_stats['buy_signal'] & (signals_for_stats['signal_strength'] >= 3)).sum()
            strong_sell = (signals_for_stats['sell_signal'] & (signals_for_stats['signal_strength'] >= 3)).sum()
        else:
            strong_buy = 0
            strong_sell = 0
            
        strong_signals = strong_buy + strong_sell
        
        st.metric("Strong Signals", strong_signals)
        st.metric("Strong Buy", strong_buy)
        st.metric("Strong Sell", strong_sell)
        
    with col3:
        # Show market hours status
        current_time_et = datetime.now(pytz.timezone('US/Eastern'))
        market_status = "OPEN" if is_market_hours(current_time_et) else "CLOSED"
        market_color = "#28a745" if market_status == "OPEN" else "#dc3545"
        
        st.markdown(f"""
        <div style="padding: 8px; border-radius: 5px; background-color: {market_color}; color: white; text-align: center; font-weight: bold;">
            Market: {market_status}
        </div>
        <div style="text-align: center; font-size: 0.9em; margin-top: 5px;">
            {current_time_et.strftime('%Y-%m-%d %H:%M:%S ET')}
        </div>
        """, unsafe_allow_html=True)
        
        # Safely check if strategy columns exist
        has_ema_vwap = 'ema_vwap_bullish' in signals_for_stats.columns and 'ema_vwap_bearish' in signals_for_stats.columns
        has_mm_vol = 'mm_vol_bullish' in signals_for_stats.columns and 'mm_vol_bearish' in signals_for_stats.columns
        
        # Calculate percentage of signals from strategic combinations
        if has_ema_vwap and has_mm_vol and not signals_for_stats.empty:
            strat_buy = (signals_for_stats['buy_signal'] & (signals_for_stats['ema_vwap_bullish'] | signals_for_stats['mm_vol_bullish'])).sum()
            strat_sell = (signals_for_stats['sell_signal'] & (signals_for_stats['ema_vwap_bearish'] | signals_for_stats['mm_vol_bearish'])).sum()
        else:
            strat_buy = 0
            strat_sell = 0
            
        strat_signals = strat_buy + strat_sell
        
        strat_pct = (strat_signals / total_signals * 100) if total_signals > 0 else 0
        
        st.metric("Strategic Signals", f"{strat_signals} ({strat_pct:.1f}%)")
        
        # Display individual strategy counts safely
        if has_ema_vwap and not signals_for_stats.empty:
            ema_vwap_count = (signals_for_stats['ema_vwap_bullish'] | signals_for_stats['ema_vwap_bearish']).sum()
        else:
            ema_vwap_count = 0
            
        if has_mm_vol and not signals_for_stats.empty:
            mm_vol_count = (signals_for_stats['mm_vol_bullish'] | signals_for_stats['mm_vol_bearish']).sum()
        else:
            mm_vol_count = 0
            
        st.metric("EMA+VWAP Signals", ema_vwap_count)
        st.metric("MM+Volume Signals", mm_vol_count)

def render_signals(data, signal_data=None):
    """
    Render signal data with visualization and detailed information
    
    Args:
        data: DataFrame with price data
        signal_data: Optional pre-calculated signal data (if None, will be calculated)
    """
    st.header("Signal Analysis")
    
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
        
    # Filter to rows with actual signals
    buy_signals = signal_data[signal_data['buy_signal']]
    sell_signals = signal_data[signal_data['sell_signal']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Buy Signals")
        if not buy_signals.empty:
            # Create a more readable table with key information
            buy_table = pd.DataFrame()
            buy_table['Time (ET)'] = buy_signals['signal_time_et']
            buy_table['Signal Price'] = buy_signals['signal_price'].map('${:.2f}'.format)
            
            # Add a "STRONG SIGNAL" badge for strong signals
            def format_strength(strength_value):
                strength_name = str(SignalStrength(strength_value).name)
                if strength_value >= 3:  # Strong or Very Strong
                    return f"{strength_name} 🔥"
                return strength_name
            
            buy_table['Strength'] = buy_signals['signal_strength'].map(format_strength)
            buy_table['Target'] = buy_signals['target_price'].map('${:.2f}'.format)
            buy_table['Stop Loss'] = buy_signals['stop_loss'].map('${:.2f}'.format)
            
            # Calculate reward-to-risk ratio
            if 'target_price' in buy_signals.columns and 'stop_loss' in buy_signals.columns:
                reward = buy_signals['target_price'] - buy_signals['signal_price']
                risk = buy_signals['signal_price'] - buy_signals['stop_loss']
                risk_reward = reward / risk
                buy_table['R:R Ratio'] = risk_reward.map('{:.2f}'.format)
            
            st.dataframe(buy_table, use_container_width=True)
        else:
            st.info("No buy signals identified in this period")
    
    with col2:
        st.subheader("Sell Signals")
        if not sell_signals.empty:
            # Create a more readable table with key information
            sell_table = pd.DataFrame()
            sell_table['Time (ET)'] = sell_signals['signal_time_et']
            sell_table['Signal Price'] = sell_signals['signal_price'].map('${:.2f}'.format)
            
            # Add a "STRONG SIGNAL" badge for strong signals
            def format_strength(strength_value):
                strength_name = str(SignalStrength(strength_value).name)
                if strength_value >= 3:  # Strong or Very Strong
                    return f"{strength_name} 🔥"
                return strength_name
            
            sell_table['Strength'] = sell_signals['signal_strength'].map(format_strength)
            sell_table['Target'] = sell_signals['target_price'].map('${:.2f}'.format)
            sell_table['Stop Loss'] = sell_signals['stop_loss'].map('${:.2f}'.format)
            
            # Calculate reward-to-risk ratio
            if 'target_price' in sell_signals.columns and 'stop_loss' in sell_signals.columns:
                reward = sell_signals['signal_price'] - sell_signals['target_price']
                risk = sell_signals['stop_loss'] - sell_signals['signal_price']
                risk_reward = reward / risk
                sell_table['R:R Ratio'] = risk_reward.map('{:.2f}'.format)
            
            st.dataframe(sell_table, use_container_width=True)
        else:
            st.info("No sell signals identified in this period")
            
    # Show signal metrics
    st.subheader("Signal Metrics")
    
    # Create a metrics row with summary information
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Buy Signals", f"{len(buy_signals)}")
    
    with metrics_col2:
        st.metric("Sell Signals", f"{len(sell_signals)}")
    
    with metrics_col3:
        if not buy_signals.empty:
            strong_buys = buy_signals[buy_signals['signal_strength'] >= 3]
            st.metric("Strong Buys", f"{len(strong_buys)}")
        else:
            st.metric("Strong Buys", "0")
    
    with metrics_col4:
        if not sell_signals.empty:
            strong_sells = sell_signals[sell_signals['signal_strength'] >= 3]
            st.metric("Strong Sells", f"{len(strong_sells)}")
        else:
            st.metric("Strong Sells", "0")
    
    # Display a horizontal rule to separate sections
    st.markdown("---")

def render_signal_chart(data, signals, symbol):
    """
    Render a price chart with signals overlaid
    
    Args:
        data: DataFrame with price data
        signals: Dictionary with signal information
        symbol: The symbol being analyzed
    """
    if data is None or data.empty or "error" in signals:
        return
        
    # Create figure with price chart
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume bars
    colors = ['green' if row['close'] >= row['open'] else 'red' for _, row in data.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Get indicator signals from primary timeframe
    indicator_signals = signals.get("indicator_signals", {})
    
    # Add indicator overlays if available
    if "Trend" in indicator_signals:
        trend_layer = indicator_signals["Trend"]
        
        # Check if we have moving averages
        for indicator_name, indicator_data in trend_layer.items():
            values = indicator_data.get("value", {})
            
            if "short_ma" in values and "long_ma" in values:
                # Convert dictionaries to series
                short_ma_dict = values["short_ma"]
                long_ma_dict = values["long_ma"]
                
                if short_ma_dict and long_ma_dict:
                    # Convert string indices to datetime
                    short_ma_series = pd.Series(short_ma_dict)
                    short_ma_series.index = pd.to_datetime(short_ma_series.index)
                    
                    long_ma_series = pd.Series(long_ma_dict)
                    long_ma_series.index = pd.to_datetime(long_ma_series.index)
                    
                    # Add to chart
                    fig.add_trace(
                        go.Scatter(
                            x=short_ma_series.index,
                            y=short_ma_series.values,
                            mode='lines',
                            name=f"{indicator_name} (Fast)",
                            line=dict(color='blue')
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=long_ma_series.index,
                            y=long_ma_series.values,
                            mode='lines',
                            name=f"{indicator_name} (Slow)",
                            line=dict(color='orange')
                        ),
                        row=1, col=1
                    )
    
    # Add buy/sell markers
    if signals.get("buy_signal", False) or signals.get("sell_signal", False):
        # Get the last date in the dataset
        last_date = data.index[-1]
        last_price = data['close'].iloc[-1]
        
        # Add marker for buy signal
        if signals.get("buy_signal", False):
            fig.add_trace(
                go.Scatter(
                    x=[last_date],
                    y=[last_price * 0.99],  # Slightly below the price
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green',
                        line=dict(color='darkgreen', width=2)
                    ),
                    name="Buy Signal",
                    showlegend=True
                ),
                row=1, col=1
            )
            
        # Add marker for sell signal
        if signals.get("sell_signal", False):
            fig.add_trace(
                go.Scatter(
                    x=[last_date],
                    y=[last_price * 1.01],  # Slightly above the price
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(color='darkred', width=2)
                    ),
                    name="Sell Signal",
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Price with Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    # Update Y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig 

def render_advanced_signals(data_dict: Dict[str, pd.DataFrame], primary_tf: str = None) -> None:
    """
    Render advanced signals visualization with component scores, market regime, and multi-timeframe confirmation
    
    Args:
        data_dict: Dictionary mapping timeframe names to price DataFrames
        primary_tf: Primary timeframe to use (defaults to shortest available)
    """
    if not data_dict or not isinstance(data_dict, dict):
        st.error("No data available for signal analysis")
        return
    
    try:
        # Import the advanced signal generation function
        from app.signals.generator import generate_signals_advanced
        
        # Generate advanced signals
        signal_data = generate_signals_advanced(data_dict, primary_tf)
        
        if not signal_data or 'signals' not in signal_data or signal_data['signals'].empty:
            if 'metrics' in signal_data and 'error' in signal_data['metrics']:
                error_msg = signal_data['metrics']['error']
                st.warning(f"{error_msg}. Try loading more historical data, using a shorter timeframe, or disabling market hours filtering.")
                
                # Show user guidance in an expandable section
                with st.expander("How to fix 'Insufficient Data' issues"):
                    st.markdown("""
                    ### Tips to fix insufficient data issues:
                    
                    1. **Load more historical data**
                       - Increase the 'Period' setting in the sidebar (e.g., change from 1d to 7d)
                    
                    2. **Choose a shorter timeframe**
                       - Switch from higher timeframes (1h, 4h) to lower ones (5m, 15m)
                       - Lower timeframes provide more data points for the same time period
                    
                    3. **Adjust market hours filtering**
                       - The advanced signal system only analyzes data during market hours
                       - For timeframes 1h and higher, you may not have enough data points during a single day's trading session
                    
                    4. **Select a more liquid symbol**
                       - Some symbols have limited trading activity
                       
                    The system requires at least 20 data points during market hours to calculate reliable signals.
                    """)
            else:
                st.warning("No signals generated. Try different data or parameters.")
            return
            
        signals = signal_data['signals']
        metrics = signal_data.get('metrics', {})
        primary_tf = signal_data.get('primary_timeframe', primary_tf)
        
        # Get multi-timeframe analysis if available
        multi_tf_data = signal_data.get('multi_timeframe_analysis', {})
        
        # Display primary timeframe and regime
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Primary Timeframe: {primary_tf}")
            
        with col2:
            # First try to get market regime from multi-timeframe analysis
            regime = None
            if 'market_regime' in multi_tf_data:
                regime = multi_tf_data['market_regime']
                if hasattr(regime, 'name'):
                    regime_name = regime.name
                elif hasattr(regime, 'value'):
                    regime_name = regime.value
                else:
                    regime_name = str(regime)
            # Fallback to signals dataframe
            elif 'market_regime' in signals.columns:
                regime = signals['market_regime'].iloc[-1]
                regime_name = regime
            else:
                regime_name = "Unknown"
                
            # Map regime name to display name and color
            regime_display = regime_name.replace('_', ' ').title() if regime_name else "Unknown"
            regime_color = {
                "bull_trend": "green",
                "bear_trend": "red",
                "sideways": "gray",
                "high_volatility": "orange",
                "low_volatility": "blue",
                "reversal": "purple",
                "BULL_TREND": "green",
                "BEAR_TREND": "red",
                "SIDEWAYS": "gray",
                "HIGH_VOLATILITY": "orange",
                "LOW_VOLATILITY": "blue",
                "REVERSAL": "purple",
            }.get(str(regime_name).lower(), "gray")
                
            st.markdown(f"### Market Regime: <span style='color:{regime_color}'>{regime_display}</span>", unsafe_allow_html=True)
        
        # Show latest signal with highlighted strength
        latest_idx = signals.index[-1]
        
        # Check if the score column exists, if not create it with default values
        if 'score' not in signals.columns:
            signals['score'] = 0.5  # Use neutral score as default
            st.warning("Signal score data not available. Using neutral values.")
        
        latest_score = signals['score'].iloc[-1]

        # Try to get signal status from multi-timeframe data first
        signal_type = "NEUTRAL"
        if multi_tf_data:
            if multi_tf_data.get('buy_signal', False):
                signal_type = "BUY"
            elif multi_tf_data.get('sell_signal', False):
                signal_type = "SELL"
        # Fallback to signals dataframe
        else:
            latest_buy = signals['buy_signal'].iloc[-1] if 'buy_signal' in signals.columns else False
            latest_sell = signals['sell_signal'].iloc[-1] if 'sell_signal' in signals.columns else False
            signal_type = "BUY" if latest_buy else "SELL" if latest_sell else "NEUTRAL"
        
        signal_color = {
            "BUY": "green",
            "SELL": "red",
            "NEUTRAL": "gray"
        }[signal_type]
        
        # Format the score as percentage
        score_pct = int(latest_score * 100)
        
        # Get signal strength if available
        strength_text = ""
        
        # Try to get signal strength from multi-tf data first
        if multi_tf_data and 'signal_strength' in multi_tf_data:
            strength_value = multi_tf_data['signal_strength']
            strength_map = {
                1: "WEAK",
                2: "MODERATE", 
                3: "STRONG",
                4: "VERY STRONG"
            }
            strength_text = f" ({strength_map.get(strength_value, '')})"
        # Fallback to signals dataframe
        elif 'signal_strength' in signals.columns and signals['signal_strength'].iloc[-1] is not None:
            strength_value = signals['signal_strength'].iloc[-1]
            strength_map = {
                1: "WEAK",
                2: "MODERATE", 
                3: "STRONG",
                4: "VERY STRONG"
            }
            strength_text = f" ({strength_map.get(strength_value, '')})"
        
        # Display the signal in a big, colored box
        st.markdown(
            f"""
            <div style="padding: 20px; background-color: {signal_color}; color: white; border-radius: 10px; text-align: center;">
                <h1>{signal_type}{strength_text}</h1>
                <h2>Score: {score_pct}%</h2>
                <p style="font-size: 0.8em; margin-top: 5px;">Signals only calculated during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Display multi-timeframe alignment and confidence
        if multi_tf_data:
            # Create columns for confidence and alignment
            col1, col2 = st.columns(2)
            
            # Display confidence metrics from multi-timeframe analysis
            with col1:
                # Check for regime-adjusted confidence first, fallback to regular confidence
                if 'regime_adjusted_confidence' in multi_tf_data:
                    confidence = multi_tf_data['regime_adjusted_confidence']
                elif 'confidence' in multi_tf_data:
                    confidence = multi_tf_data['confidence'] 
                else:
                    confidence = signals['confidence'].iloc[-1] if 'confidence' in signals.columns else 0.5
                
                conf_pct = int(confidence * 100)
                
                # Map confidence to color
                conf_color = "green" if conf_pct >= 70 else "orange" if conf_pct >= 40 else "red"
                
                st.markdown(
                    f"""
                    <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px; text-align: center;">
                        <h3>Confidence: <span style="color:{conf_color};">{conf_pct}%</span></h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            # Display timeframe alignment data
            with col2:
                if 'timeframe_alignment' in multi_tf_data:
                    alignment_score = multi_tf_data['timeframe_alignment']
                    align_pct = int(alignment_score * 100)
                    
                    # Map alignment to color
                    align_color = "green" if align_pct >= 70 else "orange" if align_pct >= 40 else "red"
                    
                    st.markdown(
                        f"""
                        <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px; text-align: center;">
                            <h3>Timeframe Alignment: <span style="color:{align_color};">{align_pct}%</span></h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Display multi-timeframe confirmation and conflicts
            if 'confirmed_by' in multi_tf_data and multi_tf_data['confirmed_by']:
                confirmations = [conf.get('timeframe') for conf in multi_tf_data['confirmed_by']]
                if confirmations:
                    st.markdown(
                        f"""
                        <div style="padding: 10px; background-color: #e6ffe6; border-radius: 5px; margin-top: 10px;">
                            <h4 style="margin: 0;">Confirmed by: <span style="color: green;">{', '.join(confirmations)}</span></h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            if 'conflicts_with' in multi_tf_data and multi_tf_data['conflicts_with']:
                conflicts = [conf.get('timeframe') for conf in multi_tf_data['conflicts_with']]
                if conflicts:
                    st.markdown(
                        f"""
                        <div style="padding: 10px; background-color: #ffe6e6; border-radius: 5px; margin-top: 10px;">
                            <h4 style="margin: 0;">Conflicts with: <span style="color: red;">{', '.join(conflicts)}</span></h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            # Fallback to basic confidence display
            if 'confidence' in signals.columns:
                confidence = signals['confidence'].iloc[-1]
                conf_pct = int(confidence * 100)
                
                st.markdown(
                    f"""
                    <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px;">
                        <h3>Confidence: {conf_pct}%</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Display multi-timeframe signals table if available
        if multi_tf_data and 'weighted_signals' in multi_tf_data:
            st.subheader("Multi-Timeframe Analysis")
            
            weighted_signals = multi_tf_data['weighted_signals']
            
            tf_data_rows = []
            for tf_name, tf_data in weighted_signals.items():
                signals_data = tf_data['signals']
                weight = tf_data['weight']
                
                # Extract signal direction
                signal_direction = "Neutral"
                if signals_data.get('buy_signal', False):
                    signal_direction = "Buy"
                elif signals_data.get('sell_signal', False):
                    signal_direction = "Sell"
                
                # Extract score (use appropriate field or fallback to default)
                score = signals_data.get('buy_score', 0) if signal_direction == "Buy" else signals_data.get('sell_score', 0) if signal_direction == "Sell" else 0
                
                tf_data_rows.append({
                    "Timeframe": tf_name,
                    "Signal": signal_direction,
                    "Score": f"{int(score * 100)}%" if signal_direction != "Neutral" else "-",
                    "Weight": f"{weight:.1f}"
                })
            
            if tf_data_rows:
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
        
        # Show signal components in a horizontal bar chart
        st.subheader("Signal Components")
        
        # First check for component scores in multi-timeframe data
        component_scores = {}
        if multi_tf_data and 'component_scores' in multi_tf_data:
            component_scores = multi_tf_data['component_scores']
        # Then fallback to checking the signals dataframe
        else:
            for component in ['trend_score', 'momentum_score', 'volume_score', 'volatility_score']:
                if component in signals.columns:
                    component_scores[component.replace('_score', '').title()] = signals[component].iloc[-1]
        
        if component_scores:
            # Create dataframe for visualization
            components = list(component_scores.keys())
            scores = list(component_scores.values())
            
            component_df = pd.DataFrame({
                'Component': [key.replace('_score', '').title() for key in components],
                'Score': scores
            })
            
            # Create bar chart with custom colors based on score
            fig = px.bar(
                component_df, 
                x='Score', 
                y='Component',
                orientation='h',
                color='Score',
                color_continuous_scale=[(0, "red"), (0.5, "yellow"), (1, "green")],
                range_color=[0, 1]
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                title_text="Component Contribution",
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show signal metrics over time
        st.subheader("Signal History")
        
        # Define date filtering for signal history
        col1, col2 = st.columns(2)
        with col1:
            last_n_days = st.selectbox(
                "Show signals from last:", 
                [1, 3, 5, 7, 14, 30, "All"], 
                index=2
            )
        
        with col2:
            signal_filter = st.multiselect(
                "Filter signals:", 
                ["Buy", "Sell", "Neutral"], 
                default=["Buy", "Sell"]
            )
            
        # Filter signals based on user selection
        filtered_signals = signals.copy()
        
        if last_n_days != "All":
            start_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=int(last_n_days))
            filtered_signals = filtered_signals[filtered_signals.index >= start_date]
            
        # Filter by signal type
        buy_filter = "Buy" in signal_filter
        sell_filter = "Sell" in signal_filter
        neutral_filter = "Neutral" in signal_filter
        
        signal_mask = pd.Series(False, index=filtered_signals.index)
        
        if buy_filter and 'buy_signal' in filtered_signals.columns:
            signal_mask = signal_mask | filtered_signals['buy_signal']
            
        if sell_filter and 'sell_signal' in filtered_signals.columns:
            signal_mask = signal_mask | filtered_signals['sell_signal']
            
        if neutral_filter:
            # Define neutral as neither buy nor sell
            if 'buy_signal' in filtered_signals.columns and 'sell_signal' in filtered_signals.columns:
                neutral_mask = ~(filtered_signals['buy_signal'] | filtered_signals['sell_signal'])
                signal_mask = signal_mask | neutral_mask
                
        filtered_signals = filtered_signals[signal_mask]
        
        # Display signal history visualization
        if not filtered_signals.empty:
            # Create a line chart with the signal score over time
            chart_data = pd.DataFrame(index=filtered_signals.index)
            
            # Add score column if it exists
            if 'score' in filtered_signals.columns:
                chart_data['score'] = filtered_signals['score']
            
            # Add component scores if available
            for component in ['trend_score', 'momentum_score', 'volume_score', 'volatility_score']:
                if component in filtered_signals.columns:
                    chart_data[component] = filtered_signals[component]
                    
            # Rename columns for better display
            chart_data.columns = [col.replace('_score', '').title() for col in chart_data.columns]
            
            # Create a Plotly figure
            fig = px.line(
                chart_data,
                labels={"value": "Score", "variable": "Component", "index": "Date"},
                title="Signal Score History"
            )
            
            # Add buy/sell signals as markers
            if 'buy_signal' in filtered_signals.columns:
                buy_points = filtered_signals[filtered_signals['buy_signal'] == True]
                if not buy_points.empty and 'score' in buy_points.columns:
                    fig.add_scatter(
                        x=buy_points.index,
                        y=buy_points['score'],
                        mode='markers',
                        marker=dict(color='green', size=12, symbol='triangle-up'),
                        name='Buy Signal'
                    )
                    
            if 'sell_signal' in filtered_signals.columns:
                sell_points = filtered_signals[filtered_signals['sell_signal'] == True]
                if not sell_points.empty and 'score' in sell_points.columns:
                    fig.add_scatter(
                        x=sell_points.index,
                        y=sell_points['score'],
                        mode='markers',
                        marker=dict(color='red', size=12, symbol='triangle-down'),
                        name='Sell Signal'
                    )
            
            # Add thresholds if possible to find them
            try:
                # Add buy threshold line
                buy_signals = filtered_signals[filtered_signals['buy_signal'] == True]
                if not buy_signals.empty and 'score' in buy_signals.columns:
                    min_buy_score = buy_signals['score'].min()
                    if min_buy_score > 0:
                        fig.add_hline(
                            y=min_buy_score,
                            line=dict(color="green", width=1, dash="dot"),
                            annotation_text="Buy Threshold"
                        )
                
                # Add sell threshold line
                sell_signals = filtered_signals[filtered_signals['sell_signal'] == True]
                if not sell_signals.empty and 'score' in sell_signals.columns:
                    max_sell_score = sell_signals['score'].max()
                    if max_sell_score < 1:
                        fig.add_hline(
                            y=max_sell_score,
                            line=dict(color="red", width=1, dash="dot"),
                            annotation_text="Sell Threshold"
                        )
            except Exception as e:
                pass  # Ignore threshold errors
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table with detailed signal information
            st.subheader("Signal Details")
            
            # Create a display DataFrame with key columns
            display_cols = ['signal_time_et', 'signal_price', 'score']
            if 'signal_strength' in filtered_signals.columns:
                display_cols.append('signal_strength')
            if 'market_regime' in filtered_signals.columns:
                display_cols.append('market_regime')
            
            display_df = filtered_signals[display_cols].copy()
            
            # Add signal type column
            if 'buy_signal' in filtered_signals.columns and 'sell_signal' in filtered_signals.columns:
                display_df['signal'] = 'Neutral'
                display_df.loc[filtered_signals['buy_signal'], 'signal'] = 'Buy'
                display_df.loc[filtered_signals['sell_signal'], 'signal'] = 'Sell'
            
            # Rename columns for display
            display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
            
            # Sort by time (most recent first)
            display_df = display_df.sort_index(ascending=False)
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No signals match the selected filters")
        
    except Exception as e:
        st.error(f"Error rendering advanced signals: {str(e)}") 