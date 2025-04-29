import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app.risk import (
    calculate_risk_reward_ratio,
    calculate_win_probability,
    calculate_position_size,
    calculate_expected_return
)
from app.signals import SignalStrength

def render_risk_analysis(account_size=10000, risk_percentage=1.0, data=None, signals=None):
    """
    Render risk analysis tools
    
    Args:
        account_size (float): Account size in dollars
        risk_percentage (float): Risk per trade percentage
        data (pd.DataFrame, optional): DataFrame with OHLC data
        signals (pd.DataFrame, optional): DataFrame with signals data
    """
    st.subheader("Risk Analysis")
    
    # Initialize session state for entry price if not exists
    if 'entry_price' not in st.session_state and data is not None and len(data) > 0:
        st.session_state.entry_price = data['close'].iloc[-1]
    
    # Trade direction
    direction = st.radio("Trade Direction", ["Long", "Short"], horizontal=True)
    
    # Entry price
    cols = st.columns(3)
    with cols[0]:
        entry_price = st.number_input("Entry Price", min_value=0.01, value=st.session_state.get('entry_price', 100.0), step=0.01)
    
    # Target price and stop loss based on direction
    with cols[1]:
        if direction == "Long":
            target_price = st.number_input("Target Price", min_value=entry_price + 0.01, value=entry_price * 1.05, step=0.01)
        else:
            target_price = st.number_input("Target Price", max_value=entry_price - 0.01, value=entry_price * 0.95, step=0.01)
    
    with cols[2]:
        if direction == "Long":
            stop_loss = st.number_input("Stop Loss", max_value=entry_price - 0.01, value=entry_price * 0.98, step=0.01)
        else:
            stop_loss = st.number_input("Stop Loss", min_value=entry_price + 0.01, value=entry_price * 1.02, step=0.01)
    
    # Signal strength
    signal_strength = st.select_slider(
        "Signal Strength",
        options=[SignalStrength.WEAK.value, SignalStrength.MODERATE.value, 
                SignalStrength.STRONG.value, SignalStrength.VERY_STRONG.value],
        value=SignalStrength.MODERATE.value,
        format_func=lambda x: {1: "Weak", 2: "Moderate", 3: "Strong", 4: "Very Strong"}[x]
    )
    
    # Calculate risk metrics
    try:
        # Risk-to-reward ratio
        risk_reward_ratio = calculate_risk_reward_ratio(entry_price, target_price, stop_loss)
        
        # Win probability
        win_probability = calculate_win_probability(signal_strength, signals)
        
        # Position size
        position_size = calculate_position_size(account_size, risk_percentage, entry_price, stop_loss)
        
        # Expected return
        expected_return, expectancy_ratio = calculate_expected_return(
            entry_price, target_price, stop_loss, win_probability
        )
        
        # Display metrics
        st.subheader("Trade Metrics")
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Risk:Reward Ratio", f"1:{risk_reward_ratio:.2f}")
        with cols[1]:
            st.metric("Win Probability", f"{win_probability*100:.1f}%")
        with cols[2]:
            st.metric("Position Size", f"{int(position_size)} shares")
        with cols[3]:
            st.metric("Expected Return", f"{expected_return:.2f}%", 
                     delta=f"{expectancy_ratio:.2f} expectancy")
        
        # Risk management summary
        st.subheader("Risk Management Summary")
        
        # Calculate dollar amounts
        risk_amount = account_size * (risk_percentage / 100)
        reward_amount = risk_amount * risk_reward_ratio
        expected_amount = (win_probability * reward_amount) - ((1 - win_probability) * risk_amount)
        
        # Create table data
        summary_data = [
            ["Account Size", f"${account_size:,.2f}"],
            ["Risk Per Trade", f"${risk_amount:.2f} ({risk_percentage:.1f}%)"],
            ["Potential Loss", f"${risk_amount:.2f}"],
            ["Potential Gain", f"${reward_amount:.2f}"],
            ["Expected Value", f"${expected_amount:.2f}"],
            ["Required Win Rate", f"{(1/(1+risk_reward_ratio))*100:.1f}%"],
            ["Estimated Win Rate", f"{win_probability*100:.1f}%"]
        ]
        
        # Create and display table
        fig = go.Figure(data=[go.Table(
            header=dict(values=["Metric", "Value"],
                        fill_color='darkblue',
                        font=dict(color='white', size=12),
                        align='left'),
            cells=dict(values=list(zip(*summary_data)),
                      fill_color='lavender',
                      align='left'))
        ])
        
        fig.update_layout(margin=dict(l=5, r=5, b=10, t=10), height=230)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk assessment
        if expectancy_ratio >= 1.5:
            risk_assessment = "Excellent trade opportunity with strong positive expectancy."
            emoji = "🟢"
        elif expectancy_ratio >= 1.0:
            risk_assessment = "Good trade opportunity with positive expectancy."
            emoji = "🟢"
        elif expectancy_ratio >= 0.5:
            risk_assessment = "Average trade opportunity with slightly positive expectancy."
            emoji = "🟡"
        else:
            risk_assessment = "Poor trade opportunity with negative expectancy."
            emoji = "🔴"
        
        st.info(f"{emoji} **Risk Assessment:** {risk_assessment}")
        
        # Additional tips based on risk characteristics
        if risk_reward_ratio < 1.0:
            st.warning("⚠️ Risk-reward ratio is less than 1:1. Consider finding a better entry or target price.")
        
        if win_probability < 0.4:
            st.warning("⚠️ Win probability is low. Wait for a stronger signal or consider reducing position size.")
        
        if risk_percentage > 2:
            st.warning("⚠️ Risk percentage is high. Consider reducing position size to protect account.")
            
    except Exception as e:
        st.error(f"Error calculating risk metrics: {str(e)}")

# Alias for compatibility
render_risk = render_risk_analysis 