import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_trends(data, symbol=None):
    """
    Render trend analysis for price data
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV price data
        symbol (str, optional): Trading symbol
    """
    if data is None or data.empty:
        st.warning("No data available for trend analysis. Please load data first.")
        return
        
    st.header(f"Market Trend Analysis {f'- {symbol}' if symbol else ''}")
    
    # Calculate trend indicators
    df = data.copy()
    
    # Add some basic trend indicators
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Calculate ADX (Average Directional Index) for trend strength
    # High ADX (>25) indicates strong trend, low ADX (<20) indicates weak trend
    df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                            np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                             np.maximum(df['low'].shift(1) - df['low'], 0), 0)
    
    # Calculate true range
    df['tr'] = np.maximum(
        np.maximum(
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift(1))
        ),
        np.abs(df['low'] - df['close'].shift(1))
    )
    
    # Smooth with Wilder's smoothing
    period = 14
    df['plus_di'] = 100 * df['plus_dm'].rolling(window=period).sum() / df['tr'].rolling(window=period).sum()
    df['minus_di'] = 100 * df['minus_dm'].rolling(window=period).sum() / df['tr'].rolling(window=period).sum()
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=period).mean()
    
    # Determine current trend
    last_row = df.iloc[-1]
    current_close = last_row['close']
    ema20 = last_row['ema20']
    ema50 = last_row['ema50']
    ema200 = last_row['ema200']
    adx = last_row['adx']
    
    # Create tabs for different trend analysis views
    trend_tabs = st.tabs(["Trend Overview", "EMA Analysis", "Trend Strength"])
    
    # Tab 1: Trend Overview
    with trend_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Determine market trend
            if current_close > ema20 and ema20 > ema50 and ema50 > ema200:
                trend = "Strong Uptrend"
                trend_color = "green"
            elif current_close > ema20 and ema20 > ema50:
                trend = "Uptrend"
                trend_color = "green"
            elif current_close < ema20 and ema20 < ema50 and ema50 < ema200:
                trend = "Strong Downtrend"
                trend_color = "red"
            elif current_close < ema20 and ema20 < ema50:
                trend = "Downtrend"
                trend_color = "red"
            elif current_close > ema50:
                trend = "Bullish Bias"
                trend_color = "lightgreen"
            elif current_close < ema50:
                trend = "Bearish Bias"
                trend_color = "pink"
            else:
                trend = "Neutral"
                trend_color = "gray"
            
            # Display trend
            st.markdown(f"<h3 style='color:{trend_color}'>Current Trend: {trend}</h3>", unsafe_allow_html=True)
            
            # Display trend strength
            if adx > 25:
                strength = "Strong"
            elif adx > 20:
                strength = "Moderate"
            else:
                strength = "Weak"
                
            st.markdown(f"<h4>Trend Strength: {strength} (ADX: {adx:.2f})</h4>", unsafe_allow_html=True)
            
            # Trend duration
            # For simplicity, count how many consecutive days close has been above/below EMA20
            if current_close > ema20:
                trend_direction = 1  # Uptrend
                days = 0
                for i in range(len(df)-1, 0, -1):
                    if df.iloc[i]['close'] > df.iloc[i]['ema20']:
                        days += 1
                    else:
                        break
            else:
                trend_direction = -1  # Downtrend
                days = 0
                for i in range(len(df)-1, 0, -1):
                    if df.iloc[i]['close'] < df.iloc[i]['ema20']:
                        days += 1
                    else:
                        break
                        
            st.markdown(f"<h4>Trend Duration: {days} periods</h4>", unsafe_allow_html=True)
        
        with col2:
            # Display recent price action
            fig = go.Figure()
            
            # Add price bars
            fig.add_trace(go.Candlestick(
                x=df.index[-30:],
                open=df['open'][-30:],
                high=df['high'][-30:],
                low=df['low'][-30:],
                close=df['close'][-30:],
                name="Price"
            ))
            
            # Add EMAs
            fig.add_trace(go.Scatter(
                x=df.index[-30:],
                y=df['ema20'][-30:],
                name="EMA20",
                line=dict(color="blue", width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index[-30:],
                y=df['ema50'][-30:],
                name="EMA50",
                line=dict(color="orange", width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index[-30:],
                y=df['ema200'][-30:],
                name="EMA200",
                line=dict(color="purple", width=1)
            ))
            
            # Update layout
            fig.update_layout(
                title="Recent Price Action",
                xaxis_title="Date",
                yaxis_title="Price",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400,
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            # Show the chart
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: EMA Analysis
    with trend_tabs[1]:
        st.subheader("EMA Analysis")
        
        # Display EMA Crossovers
        ema_crossovers = []
        for i in range(1, len(df)):
            # EMA 20 and 50
            if df.iloc[i-1]['ema20'] <= df.iloc[i-1]['ema50'] and df.iloc[i]['ema20'] > df.iloc[i]['ema50']:
                ema_crossovers.append({
                    'date': df.index[i],
                    'type': 'EMA20 crossed above EMA50',
                    'signal': 'Bullish',
                    'price': df.iloc[i]['close']
                })
            elif df.iloc[i-1]['ema20'] >= df.iloc[i-1]['ema50'] and df.iloc[i]['ema20'] < df.iloc[i]['ema50']:
                ema_crossovers.append({
                    'date': df.index[i],
                    'type': 'EMA20 crossed below EMA50',
                    'signal': 'Bearish',
                    'price': df.iloc[i]['close']
                })
                
            # EMA 50 and 200
            if df.iloc[i-1]['ema50'] <= df.iloc[i-1]['ema200'] and df.iloc[i]['ema50'] > df.iloc[i]['ema200']:
                ema_crossovers.append({
                    'date': df.index[i],
                    'type': 'EMA50 crossed above EMA200 (Golden Cross)',
                    'signal': 'Strongly Bullish',
                    'price': df.iloc[i]['close']
                })
            elif df.iloc[i-1]['ema50'] >= df.iloc[i-1]['ema200'] and df.iloc[i]['ema50'] < df.iloc[i]['ema200']:
                ema_crossovers.append({
                    'date': df.index[i],
                    'type': 'EMA50 crossed below EMA200 (Death Cross)',
                    'signal': 'Strongly Bearish',
                    'price': df.iloc[i]['close']
                })
        
        # Display recent crossovers (if any)
        if ema_crossovers:
            crossovers_df = pd.DataFrame(ema_crossovers[-5:])  # Show last 5 crossovers
            if not crossovers_df.empty:
                # Format date and price columns if they exist
                if 'date' in crossovers_df.columns:
                    if hasattr(crossovers_df['date'].iloc[0], 'strftime'):
                        crossovers_df['date'] = crossovers_df['date'].dt.strftime('%Y-%m-%d %H:%M')
                if 'price' in crossovers_df.columns:
                    crossovers_df['price'] = crossovers_df['price'].map('${:,.2f}'.format)
                
                st.dataframe(crossovers_df.sort_values('date', ascending=False), use_container_width=True)
        else:
            st.info("No EMA crossovers found in the available data.")
            
        # Display full EMA chart
        fig = go.Figure()
        
        # Add price as area chart
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            name="Price",
            line=dict(color="gray", width=1)
        ))
        
        # Add EMAs
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ema20'],
            name="EMA20",
            line=dict(color="blue", width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ema50'],
            name="EMA50",
            line=dict(color="orange", width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ema200'],
            name="EMA200",
            line=dict(color="purple", width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title="Price with EMAs",
            xaxis_title="Date",
            yaxis_title="Price",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis tips
        st.subheader("EMA Trend Analysis Tips")
        st.markdown("""
        - **Short-term Trend**: Compare price to EMA20
        - **Medium-term Trend**: Compare price to EMA50
        - **Long-term Trend**: Compare price to EMA200
        - **Golden Cross** (EMA50 crosses above EMA200): Strong bullish signal
        - **Death Cross** (EMA50 crosses below EMA200): Strong bearish signal
        """)
    
    # Tab 3: Trend Strength
    with trend_tabs[2]:
        st.subheader("Trend Strength Analysis")
        
        # Create ADX chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                name="Price",
                line=dict(color="black", width=1)
            ),
            secondary_y=True
        )
        
        # Add ADX
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['adx'],
                name="ADX",
                line=dict(color="blue", width=2)
            ),
            secondary_y=False
        )
        
        # Add +DI and -DI
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['plus_di'],
                name="+DI",
                line=dict(color="green", width=1)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['minus_di'],
                name="-DI",
                line=dict(color="red", width=1)
            ),
            secondary_y=False
        )
        
        # Add horizontal lines for ADX levels
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=25,
            x1=df.index[-1],
            y1=25,
            line=dict(color="gray", width=1, dash="dash"),
            yref="y"
        )
        
        fig.add_shape(
            type="line",
            x0=df.index[0],
            y0=20,
            x1=df.index[-1],
            y1=20,
            line=dict(color="gray", width=1, dash="dash"),
            yref="y"
        )
        
        # Update layout
        fig.update_layout(
            title="ADX Trend Strength",
            xaxis_title="Date",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="ADX & DI Values", secondary_y=False)
        fig.update_yaxes(title_text="Price", secondary_y=True)
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend strength explanation
        st.subheader("Interpreting ADX Trend Strength")
        st.markdown("""
        - **ADX > 25**: Strong trend (bullish or bearish)
        - **ADX > 20**: Moderate trend
        - **ADX < 20**: Weak trend or ranging market
        - **+DI > -DI**: Bullish momentum
        - **-DI > +DI**: Bearish momentum
        - **ADX rising**: Increasing trend strength
        - **ADX falling**: Decreasing trend strength
        """)
        
        # Display current readings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "ADX",
                f"{last_row['adx']:.2f}",
                delta=f"{last_row['adx'] - df.iloc[-2]['adx']:.2f}",
                delta_color="normal" 
            )
            
        with col2:
            st.metric(
                "+DI",
                f"{last_row['plus_di']:.2f}",
                delta=f"{last_row['plus_di'] - df.iloc[-2]['plus_di']:.2f}",
                delta_color="normal"
            )
            
        with col3:
            st.metric(
                "-DI",
                f"{last_row['minus_di']:.2f}",
                delta=f"{last_row['minus_di'] - df.iloc[-2]['minus_di']:.2f}",
                delta_color="normal"
            ) 