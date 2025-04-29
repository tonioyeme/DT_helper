import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.indicators import (
    calculate_ema,
    calculate_ema_cloud,
    calculate_macd,
    calculate_vwap,
    calculate_obv,
    calculate_ad_line,
    calculate_rsi,
    calculate_stochastic,
    calculate_fibonacci_sma,
    calculate_pain
)
from app.signals import generate_signals
from app.tradingview import (
    generate_tradingview_chart_url,
    generate_indicator_script
)

def render_chart(data, symbol=None):
    """
    Render a candlestick chart with selected indicators
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC and volume data
        symbol (str, optional): Symbol being displayed
    """
    if data is None or len(data) == 0:
        st.warning("No data to display")
        return
    
    if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        st.error("Data must contain OHLC and volume columns")
        return
    
    # Create figure with secondary y-axis (for volume)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.02, row_heights=[0.8, 0.2])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.3)'
        ),
        row=2, col=1
    )
    
    # Generate trading signals
    try:
        signals = generate_signals(data)
        
        # Add buy signals to chart
        buy_signals = signals[signals['buy_signal'] == True]
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=data.loc[buy_signals.index, 'low'] * 0.99,  # Place below candle
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Buy Signal',
                    hovertemplate='Buy Signal: %{x}<br>Price: $%{text:.2f}<extra></extra>',
                    text=data.loc[buy_signals.index, 'close']
                ),
                row=1, col=1
            )
        
        # Add sell signals to chart
        sell_signals = signals[signals['sell_signal'] == True]
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=data.loc[sell_signals.index, 'high'] * 1.01,  # Place above candle
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Sell Signal',
                    hovertemplate='Sell Signal: %{x}<br>Price: $%{text:.2f}<extra></extra>',
                    text=data.loc[sell_signals.index, 'close']
                ),
                row=1, col=1
            )
            
        # Highlight strong signals with annotations
        strong_buys = buy_signals[buy_signals['signal_strength'] >= 3]  # Strong or Very Strong
        strong_sells = sell_signals[sell_signals['signal_strength'] >= 3]  # Strong or Very Strong
        
        for idx in strong_buys.index:
            fig.add_annotation(
                x=idx,
                y=data.loc[idx, 'low'] * 0.97,
                text="STRONG BUY",
                showarrow=True,
                arrowhead=2,
                arrowcolor="green",
                arrowsize=1,
                arrowwidth=2,
                font=dict(color="green", size=10),
            )
            
        for idx in strong_sells.index:
            fig.add_annotation(
                x=idx,
                y=data.loc[idx, 'high'] * 1.03,
                text="STRONG SELL",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                arrowsize=1,
                arrowwidth=2,
                font=dict(color="red", size=10),
            )
    except Exception as e:
        st.error(f"Error adding signals to chart: {str(e)}")
        
    # Update layout
    title = f"{symbol} Chart" if symbol else "Price Chart"
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_rangeslider_visible=False
    )
    
    # Add volume title
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

def render_indicator_chart(data, indicator_type):
    """
    Render a separate chart for specific indicators
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC and volume data
        indicator_type (str): Type of indicator to render (e.g., 'macd')
    """
    if data is None or len(data) == 0:
        st.warning("No data to display")
        return
    
    if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        st.error("Data must contain OHLC and volume columns")
        return
        
    if indicator_type == 'macd':
        # Calculate MACD
        macd_line, signal_line, histogram = calculate_macd(data)
        
        # Create figure
        fig = go.Figure()
        
        # Add MACD line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=macd_line,
                name='MACD Line',
                line=dict(color='rgba(13, 71, 161, 0.8)', width=1.5)
            )
        )
        
        # Add signal line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=signal_line,
                name='Signal Line',
                line=dict(color='rgba(255, 0, 0, 0.8)', width=1.5)
            )
        )
        
        # Add histogram
        colors = ['green' if val >= 0 else 'red' for val in histogram]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=histogram,
                name='Histogram',
                marker_color=colors
            )
        )
        
        # Update layout
        fig.update_layout(
            title='MACD Indicator',
            xaxis_title='Date',
            yaxis_title='Value',
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    elif indicator_type == 'ema_cloud':
        # Calculate EMA cloud
        fast_ema, slow_ema = calculate_ema_cloud(data, 20, 50)
        
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
        
        # Add EMAs
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=fast_ema,
                name='Fast EMA',
                line=dict(color='rgba(13, 71, 161, 0.7)', width=1.5)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=slow_ema,
                name='Slow EMA',
                line=dict(color='rgba(255, 0, 0, 0.7)', width=1.5),
                fill='tonexty',
                fillcolor='rgba(0, 255, 0, 0.1)'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='EMA Cloud',
            xaxis_title='Date',
            yaxis_title='Price',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            xaxis_rangeslider_visible=False
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    elif indicator_type == 'volume_indicators':
        # Calculate OBV and A/D Line
        obv = calculate_obv(data)
        ad_line = calculate_ad_line(data)
        
        # Create figure with 2 subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1, subplot_titles=("On-Balance Volume", "Accumulation/Distribution Line"))
        
        # Add OBV
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=obv,
                name='OBV',
                line=dict(color='rgb(0, 128, 255)', width=1.5)
            ),
            row=1, col=1
        )
        
        # Add A/D Line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ad_line,
                name='A/D Line',
                line=dict(color='rgb(255, 128, 0)', width=1.5)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Volume-Based Indicators',
            xaxis_title='Date',
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    elif indicator_type == 'rsi':
        # Calculate RSI
        rsi = calculate_rsi(data)
        
        # Create figure
        fig = go.Figure()
        
        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=rsi,
                name='RSI',
                line=dict(color='rgb(153, 51, 255)', width=1.5)
            )
        )
        
        # Add overbought/oversold levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        
        # Update layout
        fig.update_layout(
            title='Relative Strength Index (RSI)',
            xaxis_title='Date',
            yaxis_title='RSI',
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            yaxis=dict(range=[0, 100])
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    elif indicator_type == 'stochastic':
        # Calculate Stochastic Oscillator
        stoch_k, stoch_d = calculate_stochastic(data)
        
        # Create figure
        fig = go.Figure()
        
        # Add Stochastic %K line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=stoch_k,
                name='%K',
                line=dict(color='rgb(0, 153, 255)', width=1.5)
            )
        )
        
        # Add Stochastic %D line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=stoch_d,
                name='%D',
                line=dict(color='rgb(255, 51, 0)', width=1.5)
            )
        )
        
        # Add overbought/oversold levels
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought (80)")
        fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold (20)")
        
        # Update layout
        fig.update_layout(
            title='Stochastic Oscillator',
            xaxis_title='Date',
            yaxis_title='Value',
            height=300,
            margin=dict(l=50, r=50, t=50, b=50),
            yaxis=dict(range=[0, 100])
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    elif indicator_type == 'fibonacci_sma':
        # Calculate Fibonacci SMAs (5-8-13)
        sma5, sma8, sma13 = calculate_fibonacci_sma(data)
        
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
        
        # Add Fibonacci SMA traces
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=sma5,
                name='SMA 5',
                line=dict(color='rgba(255, 152, 0, 0.7)', width=1.5)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=sma8,
                name='SMA 8',
                line=dict(color='rgba(76, 175, 80, 0.7)', width=1.5)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=sma13,
                name='SMA 13',
                line=dict(color='rgba(156, 39, 176, 0.7)', width=1.5)
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Fibonacci SMA Combination (5-8-13)',
            xaxis_title='Date',
            yaxis_title='Price',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            xaxis_rangeslider_visible=False
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    elif indicator_type == 'pain':
        # Calculate Price Action Indicator
        intraday_momentum, late_selling, late_buying = calculate_pain(data)
        
        # Create figure with 3 subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, 
                           subplot_titles=("Intraday Momentum", "Late Selling Pressure", "Late Buying Pressure"))
        
        # Create color scales based on values
        momentum_colors = ['red' if val < 0 else 'green' for val in intraday_momentum]
        selling_colors = ['green' if val > 0 else 'yellow' for val in late_selling]
        buying_colors = ['red' if val < 0 else 'blue' for val in late_buying]
        
        # Add Intraday Momentum
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=intraday_momentum,
                name='Intraday Momentum (Close - Open)',
                marker_color=momentum_colors
            ),
            row=1, col=1
        )
        
        # Add Late Selling Pressure
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=late_selling,
                name='Late Selling Pressure (Close - Low)',
                marker_color=selling_colors
            ),
            row=2, col=1
        )
        
        # Add Late Buying Pressure
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=late_buying,
                name='Late Buying Pressure (Close - High)',
                marker_color=buying_colors
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='Price Action Indicator (PAIN)',
            xaxis_title='Date',
            height=600,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

def render_tradingview_widget(symbol, exchange, timeframe, indicators=None):
    """
    Render a TradingView integration section with options to view chart and indicators
    
    Args:
        symbol (str): Trading symbol
        exchange (str): Exchange code
        timeframe (str): Chart timeframe
        indicators (list, optional): List of indicator names to include
    """
    st.subheader("TradingView Integration")
    
    # Generate TradingView chart URL
    chart_url = generate_tradingview_chart_url(symbol, exchange, timeframe, indicators)
    
    # Display link to TradingView chart
    st.markdown(f"[Open in TradingView]({chart_url})")
    
    # Generate Pine Script
    if indicators:
        with st.expander("Pine Script for Indicators"):
            pine_script = generate_indicator_script(indicators)
            st.code(pine_script, language="pine")
            st.info("Copy this script and paste it into TradingView Pine Editor to use these indicators.")
    
    # Display TradingView widget (iframe) if available
    try:
        # TradingView widget iframe HTML
        tv_widget_html = f"""
        <div class="tradingview-widget-container">
            <div id="tradingview_chart"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget(
            {{
                "width": "100%",
                "height": 500,
                "symbol": "{exchange}:{symbol}",
                "interval": "{timeframe}",
                "timezone": "exchange",
                "theme": "light",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "hide_top_toolbar": false,
                "studies": [
                    "MAExp@tv-basicstudies",
                    "MACD@tv-basicstudies",
                    "VWAP@tv-basicstudies"
                ],
                "container_id": "tradingview_chart"
            }}
            );
            </script>
        </div>
        """
        
        st.components.v1.html(tv_widget_html, height=550)
    except:
        st.warning("TradingView widget not available. Use the link above to open the chart in TradingView.") 