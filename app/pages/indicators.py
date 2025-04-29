import streamlit as st
import pandas as pd
from app.components.charts import render_indicator_chart
from app.components.symbols import render_symbol_selection
from app.data.loader import load_market_data

def render_indicator_page():
    st.title("Technical Indicators")
    
    # Symbol selection
    symbol = render_symbol_selection()
    
    # Time frame selection
    time_frame_options = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    time_frame = st.selectbox("Select Time Frame", time_frame_options, index=4)  # Default to 1h
    
    # Indicator selection
    indicator_options = [
        "EMA", "MACD", "RSI", "Stochastic", "VWAP", 
        "Hull MA", "ROC", "TTM Squeeze"  # Advanced indicators
    ]
    indicator = st.selectbox("Select Indicator", indicator_options)
    
    # Time range selection
    date_options = ["Last 1 Day", "Last 3 Days", "Last 5 Days", "Last Week", "Last Month", "All Data"]
    date_range = st.selectbox("Select Date Range", date_options, index=3)  # Default to 1 week
    
    # Load data with selected parameters
    data = load_market_data(symbol, time_frame, date_range)
    
    if data is not None and not data.empty:
        with st.spinner("Generating indicator chart..."):
            # Create and render the indicator chart
            chart = render_indicator_chart(data, indicator, symbol)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
    else:
        st.error("No data available for the selected parameters.")
    
    # Add indicator description
    with st.expander("Indicator Description"):
        if indicator == "EMA":
            st.write("""
            ## Exponential Moving Average (EMA)
            
            The EMA is a type of moving average that places a greater weight and significance on the most recent data points. 
            This makes it more responsive to new information compared to a simple moving average (SMA).
            
            ### Trading Signals:
            - **Bullish Signal**: When price crosses above EMA or shorter-term EMA crosses above longer-term EMA
            - **Bearish Signal**: When price crosses below EMA or shorter-term EMA crosses below longer-term EMA
            
            ### Common EMA Periods:
            - 20 EMA: Short-term trend
            - 50 EMA: Medium-term trend
            - 200 EMA: Long-term trend
            """)
        
        elif indicator == "MACD":
            st.write("""
            ## Moving Average Convergence Divergence (MACD)
            
            The MACD is calculated by subtracting the 26-period EMA from the 12-period EMA. 
            The result is the MACD line. A 9-day EMA of the MACD, called the "signal line," is then plotted on top of the MACD line.
            
            ### Trading Signals:
            - **Bullish Signal**: When MACD crosses above its signal line
            - **Bearish Signal**: When MACD crosses below its signal line
            - **Histogram**: Shows the difference between MACD and its signal line
            
            ### Other Signals:
            - **Divergence**: When price makes a new high/low but MACD doesn't
            - **Zero Line Crossover**: When MACD crosses above/below zero
            """)
        
        elif indicator == "RSI":
            st.write("""
            ## Relative Strength Index (RSI)
            
            The RSI is a momentum oscillator that measures the speed and change of price movements. 
            It oscillates between 0 and 100 and is typically used to identify overbought or oversold conditions.
            
            ### Trading Signals:
            - **Overbought**: RSI > 70 (potential sell signal)
            - **Oversold**: RSI < 30 (potential buy signal)
            
            ### Other Signals:
            - **Divergence**: When price makes a new high/low but RSI doesn't
            - **Centerline Crossover**: When RSI crosses above/below 50
            """)
        
        elif indicator == "Stochastic":
            st.write("""
            ## Stochastic Oscillator
            
            The Stochastic Oscillator is a momentum indicator comparing a particular closing price to a range of prices over a certain period of time.
            It consists of two lines: %K (fast) and %D (slow).
            
            ### Trading Signals:
            - **Overbought**: When both %K and %D are above 80
            - **Oversold**: When both %K and %D are below 20
            - **Bullish Signal**: When %K crosses above %D
            - **Bearish Signal**: When %K crosses below %D
            
            ### Other Signals:
            - **Divergence**: When price makes a new high/low but Stochastic doesn't
            """)
        
        elif indicator == "VWAP":
            st.write("""
            ## Volume Weighted Average Price (VWAP)
            
            VWAP is a trading benchmark calculated by adding up the dollars traded for every transaction (price multiplied by the volume) 
            and dividing by the total shares traded.
            
            ### Trading Signals:
            - **Bullish Signal**: When price is trading above VWAP
            - **Bearish Signal**: When price is trading below VWAP
            
            ### Common Uses:
            - Institutional traders use VWAP to help ensure they're executing trades at good prices
            - Day traders use it as a trend confirmation tool
            - Often used to identify intraday support and resistance levels
            """)
        
        elif indicator == "Hull MA":
            st.write("""
            ## Hull Moving Average (HMA)
            
            The Hull Moving Average (HMA) is a directional trend indicator developed by Alan Hull to reduce lag in moving averages 
            while maintaining smoothness. It accomplishes this by using weighted moving averages and dampening the smoothing effect.
            
            ### Trading Signals:
            - **Bullish Signal**: When price crosses above HMA or HMA slope turns positive
            - **Bearish Signal**: When price crosses below HMA or HMA slope turns negative
            
            ### Advantages:
            - Responds more quickly to price changes than traditional moving averages
            - Reduces lag while maintaining smoothness
            - Eliminates noise while preserving trend definition
            """)
        
        elif indicator == "ROC":
            st.write("""
            ## Rate of Change (ROC)
            
            The Rate of Change (ROC) indicator, also known as momentum, measures the percentage change in price between the current price 
            and the price a certain number of periods ago. It is a pure momentum oscillator.
            
            ### Trading Signals:
            - **Bullish Signal**: When ROC crosses above zero (momentum turning positive)
            - **Bearish Signal**: When ROC crosses below zero (momentum turning negative)
            
            ### Other Signals:
            - **Divergence**: When price makes a new high/low but ROC doesn't
            - **Extreme Values**: Very high or low ROC values may indicate overbought/oversold conditions
            """)
        
        elif indicator == "TTM Squeeze":
            st.write("""
            ## TTM Squeeze Indicator
            
            Developed by John Carter, the TTM Squeeze identifies when price volatility is decreasing (the "squeeze") 
            and potentially ready for a significant move. It combines Bollinger Bands and Keltner Channels with a momentum oscillator.
            
            ### Components:
            - **Bollinger Bands**: Measure standard deviation around a moving average
            - **Keltner Channels**: Use Average True Range (ATR) to measure volatility
            - **Squeeze Momentum**: Histogram showing acceleration/deceleration of price
            
            ### Trading Signals:
            - **Squeeze On**: When Bollinger Bands are inside Keltner Channels (low volatility)
            - **Squeeze Fire**: When Bollinger Bands move outside Keltner Channels (volatility increasing)
            - **Momentum Direction**: Green = bullish momentum, Red = bearish momentum
            
            ### Strategy:
            - Look for squeeze conditions followed by high momentum in either direction
            - Stronger signals occur when momentum is aligned with the longer-term trend
            """)

if __name__ == "__main__":
    render_indicator_page() 