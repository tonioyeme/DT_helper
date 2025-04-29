import pandas as pd
import numpy as np
from app.signals import SignalStrength

def calculate_risk_reward_ratio(entry_price, target_price, stop_loss_price):
    """
    Calculate risk-to-reward ratio for a trade
    
    Args:
        entry_price (float): Entry price of the trade
        target_price (float): Target price (take profit)
        stop_loss_price (float): Stop loss price
        
    Returns:
        float: Risk-to-reward ratio
    """
    if entry_price <= 0 or target_price <= 0 or stop_loss_price <= 0:
        raise ValueError("All prices must be positive")
    
    # For long positions
    if target_price > entry_price:
        reward = target_price - entry_price
        risk = entry_price - stop_loss_price
    # For short positions
    else:
        reward = entry_price - target_price
        risk = stop_loss_price - entry_price
    
    if risk <= 0:
        raise ValueError("Risk must be positive (stop loss must be below entry for longs, above for shorts)")
    
    return reward / risk

def calculate_win_probability(signal_strength, historical_signals=None):
    """
    Calculate probability of winning based on signal strength and historical performance
    
    Args:
        signal_strength (SignalStrength or int): Strength of the current signal
        historical_signals (pd.DataFrame, optional): DataFrame with historical signals and outcomes
        
    Returns:
        float: Estimated probability of winning (0.0 to 1.0)
    """
    # Base probabilities by signal strength if no historical data
    base_probabilities = {
        SignalStrength.WEAK: 0.35,
        SignalStrength.MODERATE: 0.50,
        SignalStrength.STRONG: 0.65,
        SignalStrength.VERY_STRONG: 0.75
    }
    
    # Convert int to enum if needed
    if isinstance(signal_strength, int):
        if signal_strength == 1:
            signal_strength = SignalStrength.WEAK
        elif signal_strength == 2:
            signal_strength = SignalStrength.MODERATE
        elif signal_strength == 3:
            signal_strength = SignalStrength.STRONG
        elif signal_strength == 4:
            signal_strength = SignalStrength.VERY_STRONG
        else:
            raise ValueError("Invalid signal strength value")
    
    # If no historical data provided, return base probability
    if historical_signals is None:
        return base_probabilities[signal_strength]
    
    # If historical data available, calculate win probability based on past performance
    if not isinstance(historical_signals, pd.DataFrame):
        raise ValueError("historical_signals must be a DataFrame")
    
    if 'signal_strength' not in historical_signals.columns or 'win' not in historical_signals.columns:
        raise ValueError("historical_signals must contain 'signal_strength' and 'win' columns")
    
    # Filter historical signals with the same strength
    matching_signals = historical_signals[historical_signals['signal_strength'] == signal_strength.value]
    
    # If not enough historical data, use the base probability
    if len(matching_signals) < 10:
        return base_probabilities[signal_strength]
    
    # Calculate win probability from historical data
    win_probability = matching_signals['win'].mean()
    
    # Ensure we're returning a float, not a Series or DataFrame
    if isinstance(win_probability, pd.Series) or isinstance(win_probability, pd.DataFrame):
        win_probability = float(win_probability.iloc[0])
    
    return win_probability

def calculate_position_size(account_size, risk_percentage, entry_price, stop_loss_price):
    """
    Calculate optimal position size based on risk management rules
    
    Args:
        account_size (float): Total account balance
        risk_percentage (float): Percentage of account to risk (0-100)
        entry_price (float): Entry price of the trade
        stop_loss_price (float): Stop loss price
        
    Returns:
        float: Recommended position size (number of shares/contracts)
    """
    if account_size <= 0:
        raise ValueError("Account size must be positive")
    
    if risk_percentage <= 0 or risk_percentage > 100:
        raise ValueError("Risk percentage must be between 0 and 100")
    
    # Convert percentage to decimal
    risk_fraction = risk_percentage / 100
    
    # Calculate amount willing to risk
    risk_amount = account_size * risk_fraction
    
    # Calculate risk per share/contract
    risk_per_unit = abs(entry_price - stop_loss_price)
    
    if risk_per_unit <= 0:
        raise ValueError("Risk per unit must be positive")
    
    # Calculate position size
    position_size = risk_amount / risk_per_unit
    
    return position_size

def calculate_expected_return(entry_price, target_price, stop_loss_price, win_probability):
    """
    Calculate expected return of a trade
    
    Args:
        entry_price (float): Entry price of the trade
        target_price (float): Target price (take profit)
        stop_loss_price (float): Stop loss price
        win_probability (float): Probability of winning (0.0 to 1.0)
        
    Returns:
        tuple: (Expected return percentage, Expectancy ratio)
    """
    if win_probability < 0 or win_probability > 1:
        raise ValueError("Win probability must be between 0 and 1")
    
    # Calculate reward and risk
    if target_price > entry_price:  # Long position
        reward = target_price - entry_price
        risk = entry_price - stop_loss_price
    else:  # Short position
        reward = entry_price - target_price
        risk = stop_loss_price - entry_price
    
    # Calculate reward and risk percentages
    reward_percent = (reward / entry_price) * 100
    risk_percent = (risk / entry_price) * 100
    
    # Calculate expected return percentage
    expected_return_percent = (win_probability * reward_percent) - ((1 - win_probability) * risk_percent)
    
    # Calculate expectancy ratio (average amount won per trade / average amount lost per trade)
    expectancy_ratio = (win_probability * reward) / ((1 - win_probability) * risk)
    
    return expected_return_percent, expectancy_ratio 