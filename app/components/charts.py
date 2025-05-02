import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import traceback

def render_indicator_chart(data, indicator_type, symbol=None):
    """
    Render an interactive chart for the selected technical indicator
    
    Args:
        data: DataFrame with OHLCV price data
        indicator_type: String identifying which indicator to display
        symbol: Optional trading symbol to show in chart title
        
    Returns:
        Plotly figure object
    """
    try:
        if data is None or data.empty:
            st.error("No data available to generate chart")
            return None
        
        # Set chart title
        title = f"{indicator_type} - {symbol}" if symbol else indicator_type
        
        # Create subplot structure based on indicator type
        if indicator_type in ["RSI", "Stochastic", "ROC"]:
            # Create figure with 2 rows for price and oscillator
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                subplot_titles=[title, f"{indicator_type} Values"]
            )
        else:
            # Create figure with single plot for indicators that overlay price
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=[title]
            )
        
        # Add price candlestick chart
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
        
        # Calculate and add the selected indicator
        if indicator_type == "EMA":
            # Add multiple EMAs
            ema_periods = [9, 21, 50, 200]
            colors = ['blue', 'green', 'orange', 'red']
            
            for period, color in zip(ema_periods, colors):
                ema = data['close'].ewm(span=period, adjust=False).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ema,
                        line=dict(color=color, width=1),
                        name=f'EMA {period}'
                    ),
                    row=1, col=1
                )
        
        elif indicator_type == "MACD":
            # Calculate MACD components
            ema12 = data['close'].ewm(span=12, adjust=False).mean()
            ema26 = data['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - signal_line
            
            # Add MACD components to second row
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=macd_line,
                    line=dict(color='blue', width=2),
                    name='MACD Line'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=signal_line,
                    line=dict(color='red', width=1),
                    name='Signal Line'
                ),
                row=2, col=1
            )
            
            # Add histogram
            colors = ['green' if val >= 0 else 'red' for val in macd_histogram]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=macd_histogram,
                    marker_color=colors,
                    name='Histogram'
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_trace(
                go.Scatter(
                    x=[data.index[0], data.index[-1]],
                    y=[0, 0],
                    line=dict(color='gray', width=1, dash='dash'),
                    name='Zero Line',
                    showlegend=False
                ),
                row=2, col=1
            )
            
        elif indicator_type == "RSI":
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = -loss
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Add RSI to second row
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=rsi,
                    line=dict(color='purple', width=2),
                    name='RSI (14)'
                ),
                row=2, col=1
            )
            
            # Add overbought/oversold lines
            for level, color in zip([30, 50, 70], ['green', 'gray', 'red']):
                fig.add_trace(
                    go.Scatter(
                        x=[data.index[0], data.index[-1]],
                        y=[level, level],
                        line=dict(color=color, width=1, dash='dash'),
                        name=f'Level {level}',
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
            # Set y-axis range
            fig.update_yaxes(range=[0, 100], row=2, col=1)
            
        elif indicator_type == "Stochastic":
            # Calculate Stochastic Oscillator
            k_period = 14
            d_period = 3
            
            low_min = data['low'].rolling(window=k_period).min()
            high_max = data['high'].rolling(window=k_period).max()
            
            # Calculate %K
            k = 100 * (data['close'] - low_min) / (high_max - low_min)
            
            # Calculate %D (3-period SMA of %K)
            d = k.rolling(window=d_period).mean()
            
            # Add Stochastic lines to second row
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=k,
                    line=dict(color='blue', width=2),
                    name='%K'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=d,
                    line=dict(color='red', width=1),
                    name='%D'
                ),
                row=2, col=1
            )
            
            # Add overbought/oversold lines
            for level, color in zip([20, 50, 80], ['green', 'gray', 'red']):
                fig.add_trace(
                    go.Scatter(
                        x=[data.index[0], data.index[-1]],
                        y=[level, level],
                        line=dict(color=color, width=1, dash='dash'),
                        name=f'Level {level}',
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
            # Set y-axis range
            fig.update_yaxes(range=[0, 100], row=2, col=1)
            
        elif indicator_type == "VWAP":
            # Calculate VWAP
            cumulative_tp_v = (((data['high'] + data['low'] + data['close']) / 3) * data['volume']).cumsum()
            cumulative_volume = data['volume'].cumsum()
            vwap = cumulative_tp_v / cumulative_volume
            
            # Add VWAP to main chart
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=vwap,
                    line=dict(color='purple', width=2),
                    name='VWAP'
                ),
                row=1, col=1
            )
            
        elif indicator_type == "Hull MA":
            # Calculate Hull Moving Average
            period = 20
            
            # Step 1: Calculate the weighted moving average with period/2
            half_length = int(period / 2)
            wma_half = data['close'].rolling(window=half_length).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1))
            )
            
            # Step 2: Calculate the weighted moving average with period
            wma_full = data['close'].rolling(window=period).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1))
            )
            
            # Step 3: Calculate 2 * WMA(period/2) - WMA(period)
            raw = 2 * wma_half - wma_full
            
            # Step 4: Calculate the weighted moving average of raw with sqrt(period)
            sqrt_period = int(np.sqrt(period))
            hull = raw.rolling(window=sqrt_period).apply(
                lambda x: np.sum(np.arange(1, len(x) + 1) * x) / np.sum(np.arange(1, len(x) + 1))
            )
            
            # Add Hull MA to main chart
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=hull,
                    line=dict(color='blue', width=2),
                    name=f'Hull MA ({period})'
                ),
                row=1, col=1
            )
            
        elif indicator_type == "ROC":
            # Calculate Rate of Change
            period = 14
            roc = data['close'].pct_change(period) * 100
            
            # Add ROC to second row
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=roc,
                    line=dict(color='blue', width=2),
                    name=f'ROC ({period})'
                ),
                row=2, col=1
            )
            
            # Add zero line
            fig.add_trace(
                go.Scatter(
                    x=[data.index[0], data.index[-1]],
                    y=[0, 0],
                    line=dict(color='gray', width=1, dash='dash'),
                    name='Zero Line',
                    showlegend=False
                ),
                row=2, col=1
            )
            
        elif indicator_type == "TTM Squeeze":
            # Calculate TTM Squeeze components
            # 1. Bollinger Bands
            bb_length = 20
            bb_mult = 2.0
            
            basis = data['close'].rolling(window=bb_length).mean()
            dev = bb_mult * data['close'].rolling(window=bb_length).std()
            upper_bb = basis + dev
            lower_bb = basis - dev
            
            # 2. Keltner Channels
            kc_length = 20
            kc_mult = 1.5
            
            tr1 = data['high'] - data['low']
            tr2 = abs(data['high'] - data['close'].shift())
            tr3 = abs(data['low'] - data['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=kc_length).mean()
            
            upper_kc = basis + kc_mult * atr
            lower_kc = basis - kc_mult * atr
            
            # 3. Determine squeeze
            squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
            
            # 4. Momentum
            mom_length = 12
            mom = data['close'] - data['close'].shift(mom_length)
            
            # Add BB and KC to main chart
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=upper_bb,
                    line=dict(color='rgba(0,128,0,0.5)', width=1),
                    name='Upper BB'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=lower_bb,
                    line=dict(color='rgba(0,128,0,0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(0,128,0,0.1)',
                    name='Lower BB'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=upper_kc,
                    line=dict(color='rgba(255,0,0,0.5)', width=1),
                    name='Upper KC'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=lower_kc,
                    line=dict(color='rgba(255,0,0,0.5)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    name='Lower KC'
                ),
                row=1, col=1
            )
            
            # Add basis
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=basis,
                    line=dict(color='black', width=1),
                    name='SMA'
                ),
                row=1, col=1
            )
            
            # Add momentum histogram with colors
            colors = ['green' if val >= 0 else 'red' for val in mom]
            
            # Add new row for momentum histogram
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=mom,
                    marker_color=colors,
                    name='Momentum'
                ),
                row=2, col=1
            )
            
            # Highlight squeeze periods
            for i in range(1, len(data)):
                if squeeze_on.iloc[i]:
                    fig.add_shape(
                        type="rect",
                        x0=data.index[i-1],
                        x1=data.index[i],
                        y0=data['low'].min() * 0.99,
                        y1=data['high'].max() * 1.01,
                        fillcolor="rgba(255,255,0,0.2)",
                        opacity=0.5,
                        line_width=0,
                        row=1, col=1
                    )
        
        # Update layout with range slider
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(
                l=60,
                r=60,
                b=80,
                t=100,
                pad=10
            )
        )
        
        # Set y-axis titles
        if indicator_type == "RSI":
            fig.update_yaxes(title_text="RSI Value", row=2, col=1)
        elif indicator_type == "Stochastic":
            fig.update_yaxes(title_text="Stochastic Value", row=2, col=1)
        elif indicator_type == "MACD":
            fig.update_yaxes(title_text="MACD Value", row=2, col=1)
        elif indicator_type == "ROC":
            fig.update_yaxes(title_text="ROC Value", row=2, col=1)
        elif indicator_type == "TTM Squeeze":
            fig.update_yaxes(title_text="Momentum", row=2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error rendering indicator chart: {str(e)}")
        traceback.print_exc()
        return None 