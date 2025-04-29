import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import traceback

# Print module search path for debugging
print("Python path:", sys.path)

try:
    from app.risk import (
        calculate_risk_reward_ratio,
        calculate_win_probability,
        calculate_position_size,
        calculate_expected_return,
        calculate_dynamic_position_size
    )
    print("Successfully imported risk calculation functions")
except ImportError as e:
    print(f"Error importing risk functions: {e}")
    traceback.print_exc()
    # Define fallback functions if imports fail
    def calculate_dynamic_position_size(account_size, risk_percentage, entry_price, stop_loss_price, 
                                       volatility, signal_strength=None, market_conditions=None):
        print("Using fallback dynamic position size calculation")
        # Basic calculation without adjustments
        risk_fraction = risk_percentage / 100
        risk_amount = account_size * risk_fraction
        price_diff = abs(entry_price - stop_loss_price)
        if price_diff <= 0:
            return 0
        return risk_amount / price_diff

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
    
    # Dynamic position sizing toggle
    use_dynamic_sizing = st.checkbox("Use Dynamic Position Sizing", value=True, 
                                    help="Adjusts position size based on volatility, signal strength, and market conditions")
    
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
    
    # Market conditions (only if dynamic sizing is enabled)
    if use_dynamic_sizing:
        market_condition = st.selectbox(
            "Market Condition",
            options=["Sideways", "Bull", "Bear", "Volatile"],
            index=0,
            help="Current market regime affects position sizing"
        )
    
    # Calculate risk metrics
    try:
        # Risk-to-reward ratio
        risk_reward_ratio = calculate_risk_reward_ratio(entry_price, target_price, stop_loss)
        
        # Win probability
        win_probability = calculate_win_probability(signal_strength, signals)
        
        # Calculate estimated ATR for volatility (simplified)
        if data is not None and len(data) > 0:
            # Calculate simple ATR from recent data
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            volatility = atr / entry_price
        else:
            # Fallback if no data provided (assumes 2% volatility)
            volatility = 0.02
            atr = entry_price * volatility
        
        # Position size
        try:
            if use_dynamic_sizing:
                print(f"Calculating dynamic position size with params: account_size={account_size}, risk_percentage={risk_percentage}, entry_price={entry_price}, stop_loss_price={stop_loss}, volatility={volatility}, signal_strength={signal_strength}")
                market_cond = market_condition.lower() if 'market_condition' in locals() else None
                position_size = calculate_dynamic_position_size(
                    account_size=account_size,
                    risk_percentage=risk_percentage,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    volatility=volatility,
                    signal_strength=signal_strength,
                    market_conditions=market_cond
                )
                print(f"Dynamic position size calculated: {position_size}")
            else:
                position_size = calculate_position_size(
                    account_size=account_size, 
                    risk_percentage=risk_percentage, 
                    entry_price=entry_price, 
                    stop_loss_price=stop_loss
                )
        except Exception as pos_ex:
            print(f"Error calculating position size: {pos_ex}")
            traceback.print_exc()
            # Fallback to basic calculation
            risk_amount = account_size * (risk_percentage / 100)
            risk_per_share = abs(entry_price - stop_loss)
            position_size = risk_amount / risk_per_share if risk_per_share > 0 else 0
            st.warning(f"Error in position sizing calculation. Using basic formula.")
        
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
        
        # Display dynamic position sizing explanation if enabled
        if use_dynamic_sizing:
            try:
                # Calculate standard position size for comparison
                standard_size = calculate_position_size(
                    account_size=account_size,
                    risk_percentage=risk_percentage,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss
                )
                
                adjustment_pct = ((position_size / standard_size) - 1) * 100
                
                st.info(f"**Dynamic Position Sizing:** {int(position_size)} shares ({adjustment_pct:+.1f}% from standard {int(standard_size)} shares)")
            except Exception as e:
                print(f"Error calculating standard position size: {e}")
                st.info(f"**Dynamic Position Size:** {int(position_size)} shares")
        
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

def render_risk_calculator(current_price, risk_percentage=1.0, stop_loss_atr=2.0, take_profit_ratio=2.0):
    """
    Render a risk calculator for position sizing and risk management
    
    Args:
        current_price (float): Current price of the security
        risk_percentage (float): Risk per trade as percentage of account
        stop_loss_atr (float): Stop loss multiple of ATR
        take_profit_ratio (float): Risk:reward ratio for take profit
    """
    st.subheader("Position Size Calculator")
    
    # Dynamic sizing toggle
    use_dynamic_sizing = st.checkbox("Use Dynamic Position Sizing", value=True, 
                                    help="Adjusts position size based on volatility, signal strength, and market conditions")
    
    # Account size input
    account_size = st.number_input(
        "Account Size ($)",
        min_value=100.0,
        max_value=10000000.0,
        value=10000.0,
        step=1000.0,
        format="%.2f"
    )
    
    # Current market price
    price = st.number_input(
        "Current Price ($)",
        min_value=0.01,
        value=float(current_price),
        step=0.01,
        format="%.2f"
    )
    
    # Create columns for input parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Risk percentage
        risk_pct = st.number_input(
            "Risk per Trade (%)",
            min_value=0.1,
            max_value=10.0,
            value=float(risk_percentage),
            step=0.1,
            format="%.1f"
        )
    
    with col2:
        # Trade direction
        direction = st.radio(
            "Trade Direction",
            options=["Long", "Short"],
            horizontal=True
        )
    
    with col3:
        # ATR period
        atr_period = st.selectbox(
            "ATR Period",
            options=[14, 7, 21],
            index=0
        )
    
    # Calculate ATR (normally would be from price data, using mock value here)
    # In real implementation, this would use actual price data
    atr_value = price * 0.015
    
    # Create columns for risk parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate stop loss based on ATR
        atr_multiple = st.number_input(
            "Stop Loss (ATR Multiple)",
            min_value=0.5,
            max_value=5.0,
            value=float(stop_loss_atr),
            step=0.5,
            format="%.1f"
        )
        
        # Calculate stop loss price
        atr_stop = atr_value * atr_multiple
        if direction == "Long":
            stop_loss = price - atr_stop
        else:
            stop_loss = price + atr_stop
        
        st.text(f"Stop Loss: ${stop_loss:.2f}")
        
    with col2:
        # Risk to reward ratio for take profit
        risk_reward = st.number_input(
            "Risk:Reward Ratio",
            min_value=1.0,
            max_value=10.0,
            value=float(take_profit_ratio),
            step=0.5,
            format="%.1f"
        )
        
        # Calculate take profit based on risk:reward ratio
        price_risk = abs(price - stop_loss)
        price_reward = price_risk * risk_reward
        
        if direction == "Long":
            take_profit = price + price_reward
        else:
            take_profit = price - price_reward
            
        st.text(f"Take Profit: ${take_profit:.2f}")
    
    with col3:
        # Signal strength
        signal_strength = st.select_slider(
            "Signal Strength",
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda x: {1: "Weak", 2: "Moderate", 3: "Strong", 4: "Very Strong"}[x]
        )
        
        # Market condition
        market_condition = st.selectbox(
            "Market Condition",
            options=["Sideways", "Bull", "Bear", "Volatile"],
            index=0
        )
    
    # Calculate risk amount
    risk_amount = account_size * (risk_pct / 100)
    
    # Risk per share
    risk_per_share = abs(price - stop_loss)
    
    if risk_per_share <= 0:
        st.error("Invalid stop loss. Please adjust to create a valid risk level.")
        position_size = 0
        total_position = 0
    else:
        # Calculate volatility as ATR/price
        volatility = atr_value / price
        
        try:
            if use_dynamic_sizing:
                print(f"Risk Calculator - Calculating dynamic position size with params: account_size={account_size}, risk_percentage={risk_pct}, entry_price={price}, stop_loss_price={stop_loss}, volatility={volatility}, signal_strength={signal_strength}, market_conditions={market_condition.lower()}")
                position_size = int(calculate_dynamic_position_size(
                    account_size=account_size,
                    risk_percentage=risk_pct,
                    entry_price=price,
                    stop_loss_price=stop_loss,
                    volatility=volatility,
                    signal_strength=signal_strength,
                    market_conditions=market_condition.lower()
                ))
                print(f"Risk Calculator - Dynamic position size calculated: {position_size}")
            else:
                position_size = int(calculate_position_size(
                    account_size=account_size,
                    risk_percentage=risk_pct,
                    entry_price=price,
                    stop_loss_price=stop_loss
                ))
        except Exception as e:
            print(f"Risk Calculator - Error calculating position size: {e}")
            traceback.print_exc()
            # Fallback to basic calculation
            position_size = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            st.warning(f"Error in position sizing calculation. Using basic formula.")
            
        total_position = position_size * price
    
    # Calculate key metrics
    if position_size > 0:
        reward_amount = position_size * abs(take_profit - price)
        risk_reward = reward_amount / risk_amount
        
        # Calculate required win rate for breakeven
        required_win_rate = 1 / (1 + risk_reward)
    else:
        reward_amount = 0
        risk_reward = 0
        required_win_rate = 0
    
    # Display dynamic sizing explanation if enabled
    if use_dynamic_sizing and position_size > 0:
        try:
            # Calculate standard position size for comparison
            standard_size = int(calculate_position_size(
                account_size=account_size,
                risk_percentage=risk_pct,
                entry_price=price,
                stop_loss_price=stop_loss
            ))
            
            # Calculate adjustment factors
            volatility_factor = np.clip(0.02 / max(volatility, 0.001), 0.5, 1.5)
            signal_factor = 0.8 + (signal_strength - 1) * 0.1
            
            market_factor = 1.0
            if market_condition.lower() == "bull":
                market_factor = 1.1
            elif market_condition.lower() == "bear":
                market_factor = 0.9
            elif market_condition.lower() == "volatile":
                market_factor = 0.8
            
            adjustment_pct = ((position_size / standard_size) - 1) * 100
            
            st.info(f"""
            **Dynamic Sizing Adjustments:**
            - Standard Position Size: {standard_size} shares
            - Volatility Factor: {volatility_factor:.2f}x ({"Lower" if volatility_factor < 1 else "Higher"} volatility)
            - Signal Strength Factor: {signal_factor:.2f}x ({signal_strength}/4)
            - Market Condition Factor: {market_factor:.2f}x ({market_condition})
            - **Total Adjustment: {adjustment_pct:+.1f}%**
            """)
        except Exception as e:
            print(f"Risk Calculator - Error calculating adjustment factors: {e}")
            st.info(f"Dynamic Position Size: {position_size} shares")
    
    # Display results in a box
    st.markdown("### Trade Plan")
    
    results_html = f"""
    <div style="background-color:#f0f2f6; padding:10px; border-radius:10px;">
        <table width="100%">
            <tr>
                <td><b>Position Size:</b></td>
                <td>{position_size} shares (${total_position:.2f})</td>
                <td><b>Account Utilization:</b></td>
                <td>{(total_position/account_size)*100:.1f}%</td>
            </tr>
            <tr>
                <td><b>Max Risk:</b></td>
                <td>${risk_amount:.2f}</td>
                <td><b>Potential Reward:</b></td>
                <td>${reward_amount:.2f}</td>
            </tr>
            <tr>
                <td><b>Risk:Reward Ratio:</b></td>
                <td>1:{risk_reward:.2f}</td>
                <td><b>Required Win Rate:</b></td>
                <td>{required_win_rate*100:.1f}%</td>
            </tr>
        </table>
    </div>
    """
    
    st.markdown(results_html, unsafe_allow_html=True)
    
    # Trade instruction
    if position_size > 0:
        instruction = f"{'BUY' if direction == 'Long' else 'SELL'} {position_size} shares at ${price:.2f}"
        instruction += f", stop loss at ${stop_loss:.2f}, target at ${take_profit:.2f}"
        
        if direction == "Long":
            color = "green" if risk_reward >= 2 else "orange"
        else:
            color = "red" if risk_reward >= 2 else "orange"
            
        st.markdown(f"<div style='background-color:{color}; padding:10px; border-radius:5px; color:white;'><b>{instruction}</b></div>", unsafe_allow_html=True)
    
    # Risk management tips
    with st.expander("Risk Management Tips"):
        st.markdown("""
        - **2% Rule**: Never risk more than 2% of your account on a single trade
        - **Position Sizing**: Adjust position size based on volatility (ATR)
        - **Risk:Reward**: Aim for at least 1:2 risk-to-reward ratio
        - **Stop Loss**: Always use a stop loss to protect your capital
        - **Correlation**: Be aware of correlated positions that increase overall risk
        - **Psychological Risk**: Never take a position so large it affects your emotions
        - **Dynamic Sizing**: Scale position size based on volatility, signal strength, and market conditions
        - **High Volatility**: Reduce position size in high volatility environments
        - **Strong Signals**: Consider larger positions for stronger, higher-confidence signals
        """)

# Alias for compatibility
render_risk = render_risk_analysis 