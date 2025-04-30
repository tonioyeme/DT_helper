import sys
import traceback

try:
    from app.risk.calculator import (
        calculate_risk_reward_ratio,
        calculate_win_probability,
        calculate_position_size,
        calculate_expected_return,
        calculate_dynamic_position_size
    )
    print("Successfully imported all risk calculation functions")
except ImportError as e:
    print(f"Error importing from calculator module: {e}")
    traceback.print_exc()
    
    # Define fallback functions if imports fail
    def calculate_risk_reward_ratio(entry_price, target_price, stop_loss_price):
        print("Using fallback risk reward ratio calculation")
        if entry_price <= 0 or target_price <= 0 or stop_loss_price <= 0:
            return 1.0
        
        # For long positions
        if target_price > entry_price:
            reward = target_price - entry_price
            risk = entry_price - stop_loss_price
        # For short positions
        else:
            reward = entry_price - target_price
            risk = stop_loss_price - entry_price
        
        if risk <= 0:
            return 1.0
        
        return reward / risk
        
    def calculate_win_probability(signal_strength, historical_signals=None):
        print("Using fallback win probability calculation")
        # Base probabilities by signal strength
        if isinstance(signal_strength, int):
            if signal_strength == 1:
                return 0.35
            elif signal_strength == 2:
                return 0.50
            elif signal_strength == 3:
                return 0.65
            elif signal_strength == 4:
                return 0.75
        return 0.50
        
    def calculate_position_size(account_size, risk_percentage, entry_price, stop_loss_price):
        print("Using fallback position size calculation")
        if account_size <= 0 or risk_percentage <= 0 or risk_percentage > 100:
            return 0
            
        # Convert percentage to decimal
        risk_fraction = risk_percentage / 100
        
        # Calculate amount willing to risk
        risk_amount = account_size * risk_fraction
        
        # Calculate risk per share/contract
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            return 0
        
        # Calculate position size
        return risk_amount / risk_per_unit
        
    def calculate_expected_return(entry_price, target_price, stop_loss_price, win_probability):
        print("Using fallback expected return calculation")
        # Calculate risk and reward in percentage terms
        if entry_price <= 0:
            return 0, 0
            
        # For long positions
        if target_price > entry_price:
            reward_pct = (target_price - entry_price) / entry_price * 100
            risk_pct = (entry_price - stop_loss_price) / entry_price * 100
        # For short positions
        else:
            reward_pct = (entry_price - target_price) / entry_price * 100
            risk_pct = (stop_loss_price - entry_price) / entry_price * 100
            
        if risk_pct <= 0:
            return 0, 0
            
        # Calculate expected return
        expected_return = (win_probability * reward_pct) - ((1 - win_probability) * risk_pct)
        
        # Calculate expectancy ratio
        expectancy_ratio = (win_probability * reward_pct / risk_pct) - (1 - win_probability)
        
        return expected_return, expectancy_ratio
        
    def calculate_dynamic_position_size(account_size, risk_percentage, entry_price, stop_loss_price, volatility, signal_strength=None, market_conditions=None):
        print("Using fallback dynamic position size calculation")
        if account_size <= 0 or risk_percentage <= 0 or risk_percentage > 100:
            return 0
            
        # Convert percentage to decimal
        risk_fraction = risk_percentage / 100
        
        # Calculate base risk amount
        risk_amount = account_size * risk_fraction
        
        # Calculate price difference for risk
        price_diff = abs(entry_price - stop_loss_price)
        
        if price_diff <= 0:
            return 0
            
        # Calculate base position size
        base_size = risk_amount / price_diff
        
        # Apply simple adjustments
        if signal_strength is not None and isinstance(signal_strength, int) and signal_strength >= 1 and signal_strength <= 4:
            # Scale between 0.8-1.2 based on signal strength (1-4)
            signal_factor = 0.8 + (signal_strength - 1) * 0.1
            base_size *= signal_factor
            
        if market_conditions is not None and isinstance(market_conditions, str):
            market_conditions = market_conditions.lower()
            if market_conditions == "bull":
                base_size *= 1.1  # More aggressive in bull market
            elif market_conditions == "bear":
                base_size *= 0.9  # More conservative in bear market
            elif market_conditions == "volatile":
                base_size *= 0.8  # More conservative in volatile market
        
        import math
        return math.floor(base_size)

__all__ = [
    'calculate_risk_reward_ratio',
    'calculate_win_probability',
    'calculate_position_size',
    'calculate_expected_return',
    'calculate_dynamic_position_size'
] 