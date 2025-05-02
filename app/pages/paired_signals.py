import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import plotly.graph_objects as go

from app.signals.paired_signals import PairedSignalGenerator
from app.data.loader import load_days_data
from app.signals.paired_signal_examples import visualize_paired_signals

def app():
    st.title("Paired Entry-Exit Signals")
    st.write("Analyze complete trade pairs with entry and exit signals")
    
    # Sidebar options
    st.sidebar.subheader("Paired Signal Settings")
    symbol = st.sidebar.text_input("Symbol", value="SPY")
    days = st.sidebar.slider("Days to analyze", min_value=1, max_value=10, value=3)
    min_profit = st.sidebar.slider("Minimum profit threshold (%)", min_value=0.05, max_value=2.0, value=0.15, step=0.05)
    
    # Handle button click
    if st.sidebar.button("Generate Paired Signals"):
        with st.spinner("Loading data..."):
            # Load data
            data = load_days_data(symbol=symbol, days=days)
            
            if data is None or len(data) < 20:
                st.error(f"Insufficient data for {symbol}. Please try another symbol or timeframe.")
                return
                
            st.session_state.data = data
            st.session_state.symbol = symbol
            
            # Generate signals
            with st.spinner("Generating paired signals..."):
                generator = PairedSignalGenerator()
                # Set custom minimum profit if provided
                generator.position_manager.min_profit = min_profit / 100  # Convert from % to decimal
                
                results = generator.generate_paired_signals(data)
                st.session_state.paired_results = results
    
    # Display results if available
    if hasattr(st.session_state, 'paired_results') and st.session_state.paired_results is not None:
        results = st.session_state.paired_results
        paired_signals = results['paired_signals']
        
        # Display metrics
        metrics = results.get('metrics', {})
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Trades", f"{metrics.get('total_trades', 0)}")
        
        with col2:
            win_rate = metrics.get('win_rate', 0) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            avg_profit = metrics.get('avg_profit', 0) * 100
            st.metric("Avg Profit", f"{avg_profit:.2f}%")
            
        with col4:
            max_profit = metrics.get('max_profit', 0) * 100
            st.metric("Max Profit", f"{max_profit:.2f}%")
            
        with col5:
            max_loss = metrics.get('max_loss', 0) * 100
            st.metric("Max Loss", f"{max_loss:.2f}%")
        
        # Visualize signals
        st.subheader("Signal Visualization")
        fig = visualize_paired_signals(results, st.session_state.symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display paired trades table
        st.subheader("Paired Trades")
        if not paired_signals.empty:
            # Convert to display format
            display_df = paired_signals.copy()
            
            # Format timestamps
            eastern = pytz.timezone('US/Eastern')
            display_df['entry_time'] = [pd.Timestamp(t).astimezone(eastern).strftime('%Y-%m-%d %H:%M ET') 
                                      if hasattr(t, 'astimezone') else str(t) 
                                      for t in display_df['entry_time']]
            display_df['exit_time'] = [pd.Timestamp(t).astimezone(eastern).strftime('%Y-%m-%d %H:%M ET') 
                                     if hasattr(t, 'astimezone') else str(t)
                                     for t in display_df['exit_time']]
            
            # Format price and profit columns
            display_df['entry_price'] = display_df['entry_price'].map('${:.2f}'.format)
            display_df['exit_price'] = display_df['exit_price'].map('${:.2f}'.format)
            display_df['profit_pct'] = (display_df['profit_pct'] * 100).map('{:.2f}%'.format)
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'entry_time': 'Entry Time',
                'exit_time': 'Exit Time',
                'type': 'Position',
                'entry_price': 'Entry Price',
                'exit_price': 'Exit Price',
                'profit_pct': 'Profit'
            })
            
            # Display the table
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No valid paired trades found with the current settings.")
    
    # Information about the paired signal approach
    with st.expander("About Paired Signals"):
        st.markdown("""
        ### Entry-Exit Signal Pairing System
        
        This system pairs entry signals with their corresponding exit signals to analyze complete trades.
        Key features:
        
        - **Complete Trades**: Only considers trades with both entry and exit signals
        - **Profit Validation**: Ensures trades meet minimum profit threshold (default 0.15%)
        - **Performance Metrics**: Calculates win rate, average profit, and more
        - **Visual Analysis**: Shows entry/exit points and profit for each trade
        
        Ideal for:
        - Backtesting trading strategies
        - Optimizing entry/exit parameters
        - Understanding trade performance patterns
        
        Adjust the minimum profit threshold to filter for more significant trading opportunities.
        """)

if __name__ == "__main__":
    app() 