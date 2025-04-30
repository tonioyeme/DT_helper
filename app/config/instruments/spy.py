"""
SPY configuration with optimized parameters for day trading.
This configuration is tailored for intraday trading of SPY, focusing on
key technical indicators, support/resistance levels, and volatility-based adjustments.
"""

from typing import Dict, Any, List

SPY_CONFIG: Dict[str, Any] = {
    "meta": {
        "name": "SPY Day Trading Configuration",
        "description": "Optimized parameters for intraday trading of SPY",
        "version": "1.0.0",
        "timeframes": ["1m", "5m", "15m", "1h", "4h"],
        "primary_timeframe": "5m",
    },
    
    "indicators": {
        "ema": {
            "fast_period": 8,
            "slow_period": 21,
            "signal_period": 5,
            "weight": 1.2,
            "cloud_periods": [5, 13],
        },
        "rsi": {
            "period": 14,
            "overbought": 70,
            "oversold": 30,
            "weight": 1.0,
            "adaptive": True,
            "adaptive_period": 14,
        },
        "vwap": {
            "enabled": True,
            "bands": {
                "enabled": True,
                "std_dev": 1.5,
                "weight": 1.2,
            },
            "weight": 1.5,
        },
        "bollinger_bands": {
            "period": 20,
            "std_dev": 2.0,
            "weight": 1.0,
        },
        "volume": {
            "sma_period": 20,
            "weight": 1.1,
            "threshold": 1.5,  # Volume threshold multiplier
        },
        "atr": {
            "period": 14,
            "multiplier": 1.5,
            "weight": 1.0,
        },
        "macd": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "weight": 0.8,
        },
        "stochastic": {
            "k_period": 14,
            "d_period": 3,
            "smooth_k": 3,
            "weight": 0.7,
        },
        "support_resistance": {
            "enabled": True,
            "lookback_period": 5,  # days
            "strength_threshold": 3,
            "proximity_threshold": 0.5,  # percentage
            "weight": 1.3,
        },
    },
    
    "strategies": {
        "orb": {  # Opening Range Breakout
            "enabled": True,
            "minutes": 15,
            "threshold": 0.1,  # percentage breakout
            "weight": 1.4,
        },
        "vwap_bb": {  # VWAP and Bollinger Band strategy
            "enabled": True,
            "band_penetration": 0.3,  # percentage penetration for signal
            "weight": 1.2,
        },
        "ema_vwap_cross": {  # EMA and VWAP crossover strategy
            "enabled": True,
            "weight": 1.3,
        },
        "volume_price_divergence": {
            "enabled": True,
            "lookback": 5,
            "threshold": 0.2,
            "weight": 1.1,
        },
    },
    
    "signals": {
        "threshold": 0.65,  # Score threshold for signal generation
        "time_decay": {
            "enabled": True,
            "factor": 0.95,  # Decay factor for older signals
            "interval": "1h",  # Time interval for decay
        },
        "session_weights": {
            "pre_market": 0.7,
            "regular_hours": 1.0,
            "post_market": 0.6,
        },
    },
    
    "market_conditions": {
        "high_volatility": {
            "threshold": 1.5,  # ATR multiple for high volatility
            "signal_adjustment": 0.8,  # Reduce signal weight in high volatility
            "stop_loss_multiplier": 1.5,  # Increase stop loss in high volatility
        },
        "low_volatility": {
            "threshold": 0.7,  # ATR multiple for low volatility
            "signal_adjustment": 1.2,  # Increase signal weight in low volatility
            "stop_loss_multiplier": 0.8,  # Decrease stop loss in low volatility
        },
        "trend_following": {
            "enabled": True,
            "weight_adjustment": 1.3,  # Increase trend signals in trending market
        },
        "mean_reversion": {
            "enabled": True,
            "weight_adjustment": 1.1,  # Increase reversal signals in choppy market
        },
    },
    
    "time_filters": {
        "avoid_fomc": True,  # Avoid trading during FOMC announcements
        "avoid_first_minutes": 5,  # Avoid trading in first X minutes
        "avoid_last_minutes": 5,  # Avoid trading in last X minutes
    },
    
    "position_sizing": {
        "method": "atr",  # Options: fixed, percentage, atr
        "risk_per_trade": 1.0,  # Percentage of account
        "max_position_size": 5.0,  # Maximum position size as percentage of account
        "atr_multiplier": 2.0,  # For ATR-based position sizing
    },
    
    "risk_management": {
        "stop_loss": {
            "method": "atr",  # Options: fixed, percentage, atr
            "atr_multiplier": 2.0,
            "fixed_percentage": 1.0,
        },
        "take_profit": {
            "method": "risk_reward",  # Options: fixed, percentage, risk_reward
            "risk_reward_ratio": 2.0,
            "fixed_percentage": 2.0,
        },
        "trailing_stop": {
            "enabled": True,
            "activation_percentage": 1.0,  # Profit percentage to activate trailing stop
            "trail_percentage": 0.5,  # Percentage to trail by
        },
    },
} 