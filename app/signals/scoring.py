import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import logging
from datetime import datetime, timedelta

class SignalStrength(Enum):
    """Signal strength classifications"""
    WEAK = 1      # Low significance, requires additional confirmation
    MODERATE = 2  # Moderate significance, reasonable confidence
    STRONG = 3    # Strong significance, high confidence
    VERY_STRONG = 4  # Very strong significance, highest confidence

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_TREND = "bull_trend"              # Established uptrend
    BEAR_TREND = "bear_trend"              # Established downtrend
    RANGE_BOUND = "range_bound"            # Price moving in a range
    SIDEWAYS = "sideways"                  # No clear direction
    HIGH_VOLATILITY = "high_volatility"    # Increased volatility
    LOW_VOLATILITY = "low_volatility"      # Decreased volatility
    BREAKOUT = "breakout"                  # Breaking from a range
    REVERSAL = "reversal"                  # Potential trend reversal

class DynamicSignalScorer:
    """
    Dynamic signal scoring system that adjusts thresholds based on market conditions,
    volatility, and indicator performance history
    """
    
    def __init__(self):
        """Initialize the dynamic scorer"""
        self.logger = logging.getLogger(__name__)
        
        # Default thresholds for buy/sell signals
        self.buy_threshold = 5.0  # Minimum score for a buy signal
        self.sell_threshold = 5.0  # Minimum score for a sell signal
        
        # Store registered indicators with their weights
        self.indicators = {}
        
        # Store indicator performance metrics
        self.performance_metrics = {}
        
        # Market regime weights for different conditions
        self.regime_adjustments = {
            MarketRegime.BULL_TREND: {"buy": 0.9, "sell": 1.2},
            MarketRegime.BEAR_TREND: {"buy": 1.2, "sell": 0.9},
            MarketRegime.RANGE_BOUND: {"buy": 1.0, "sell": 1.0},
            MarketRegime.SIDEWAYS: {"buy": 1.1, "sell": 1.1},
            MarketRegime.HIGH_VOLATILITY: {"buy": 1.2, "sell": 1.2},
            MarketRegime.LOW_VOLATILITY: {"buy": 0.9, "sell": 0.9},
            MarketRegime.BREAKOUT: {"buy": 0.8, "sell": 1.1},
            MarketRegime.REVERSAL: {"buy": 0.7, "sell": 0.7}
        }
        
    def set_baseline_thresholds(self, buy_threshold: float = 5.0, sell_threshold: float = 5.0):
        """
        Set baseline thresholds for buy/sell signals
        
        Args:
            buy_threshold: Base threshold for buy signals
            sell_threshold: Base threshold for sell signals
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        
    def register_indicator(self, name: str, category: Any, base_weight: float = 1.0, 
                          condition_weights: Dict[MarketRegime, float] = None):
        """
        Register an indicator with its base weight and condition-specific weights
        
        Args:
            name: Name of the indicator
            category: Category the indicator belongs to
            base_weight: Base weight for this indicator
            condition_weights: Dictionary mapping market regimes to weight multipliers
        """
        if condition_weights is None:
            condition_weights = {}
            
        # Store indicator configuration
        self.indicators[name] = {
            'name': name,
            'category': category,
            'base_weight': base_weight,
            'condition_weights': condition_weights
        }
        
        # Initialize performance metrics
        if name not in self.performance_metrics:
            self.performance_metrics[name] = {
                'accuracy': 0.75,  # Initial assumed accuracy (75%)
                'total_signals': 0,
                'correct_signals': 0,
                'recent_signals': []  # List of recent signal results
            }
            
    def update_indicator_performance(self, indicator_name: str, was_correct: bool):
        """
        Update performance metrics for an indicator based on signal accuracy
        
        Args:
            indicator_name: Name of the indicator
            was_correct: Whether the signal was correct
        """
        if indicator_name not in self.performance_metrics:
            self.logger.warning(f"Indicator {indicator_name} not found in performance metrics")
            return
            
        metrics = self.performance_metrics[indicator_name]
        
        # Update signal counts
        metrics['total_signals'] += 1
        if was_correct:
            metrics['correct_signals'] += 1
            
        # Update recent signals list (keep last 10)
        metrics['recent_signals'].append(was_correct)
        if len(metrics['recent_signals']) > 10:
            metrics['recent_signals'].pop(0)
            
        # Recalculate accuracy
        metrics['accuracy'] = metrics['correct_signals'] / max(1, metrics['total_signals'])
        
        # If we have recent signals, weight them more heavily
        if metrics['recent_signals']:
            recent_accuracy = sum(metrics['recent_signals']) / len(metrics['recent_signals'])
            
            # Blend overall accuracy with recent accuracy (weighted toward recent)
            metrics['accuracy'] = (metrics['accuracy'] * 0.3) + (recent_accuracy * 0.7)
            
    def calculate_volatility_factor(self, data: pd.DataFrame) -> float:
        """
        Calculate a volatility factor based on recent price action
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Volatility factor (multiplier between 0.5 and 1.5)
        """
        if data is None or data.empty:
            return 1.0
            
        try:
            # Use a simple volatility measure: ATR relative to price
            # First, calculate TR (True Range)
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate ATR (Average True Range) - 14 period
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            # Calculate ATR as a percentage of price
            atr_pct = atr / data['close'].iloc[-1]
            
            # Normalize to a factor between 0.5 and 1.5
            # Higher volatility = higher factor (more conservative thresholds)
            if atr_pct < 0.005:  # Less than 0.5% volatility
                return 0.5  # Low volatility
            elif atr_pct > 0.03:  # More than 3% volatility
                return 1.5  # High volatility
            else:
                # Scale between 0.5 and 1.5
                return 0.5 + (atr_pct - 0.005) * (1.0 / 0.025)
                
        except Exception as e:
            self.logger.error(f"Error calculating volatility factor: {str(e)}")
            return 1.0  # Default to neutral
            
    def detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Detect the current market regime based on price data
        
        Args:
            data: DataFrame with price data
            
        Returns:
            MarketRegime enum value
        """
        if data is None or data.empty or len(data) < 30:
            return MarketRegime.SIDEWAYS  # Default to sideways if insufficient data
            
        try:
            # Calculate some basic indicators to determine regime
            
            # 1. Moving averages for trend direction
            ma_short = data['close'].rolling(window=20).mean()
            ma_long = data['close'].rolling(window=50).mean()
            
            # 2. ATR for volatility
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            # 3. Normalized ATR (ATR as percentage of price)
            normalized_atr = atr / data['close']
            
            # 4. ADX for trend strength
            plus_dm = data['high'].diff()
            minus_dm = data['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = abs(minus_dm)
            
            # Get the true range
            tr_adx = tr
            
            # Smooth DM and TR using Wilder's smoothing method
            period = 14
            plus_di = 100 * plus_dm.rolling(window=period).sum() / tr_adx.rolling(window=period).sum()
            minus_di = 100 * minus_dm.rolling(window=period).sum() / tr_adx.rolling(window=period).sum()
            
            # Calculate DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            # Get current values
            current_adx = adx.iloc[-1]
            trend_direction = 1 if ma_short.iloc[-1] > ma_long.iloc[-1] else -1
            current_volatility = normalized_atr.iloc[-1]
            
            # Calculate historical volatility average
            historical_volatility_avg = normalized_atr.iloc[-30:].mean()
            volatility_ratio = current_volatility / historical_volatility_avg if historical_volatility_avg > 0 else 1
            
            # Detect breakout: significant price movement beyond recent range
            price_range_20d = data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
            avg_range = price_range_20d.iloc[-5:].mean()
            latest_move = abs(data['close'].iloc[-1] - data['close'].iloc[-5])
            
            is_breakout = latest_move > (0.7 * avg_range)
            
            # Detect reversal: price crossing over medium-term moving average
            price_above_ma = data['close'] > ma_long
            
            # Check for recent crossing
            crossed_ma = False
            if len(price_above_ma) > 5:
                crossed_ma = price_above_ma.iloc[-1] != price_above_ma.iloc[-5]
                
            # Determine regime based on indicators
            if current_adx > 25:  # Strong trend
                if trend_direction > 0:
                    regime = MarketRegime.BULL_TREND
                else:
                    regime = MarketRegime.BEAR_TREND
            elif current_adx < 15:  # Weak trend
                if volatility_ratio > 1.5:  # Higher than normal volatility
                    regime = MarketRegime.HIGH_VOLATILITY
                elif volatility_ratio < 0.5:  # Lower than normal volatility
                    regime = MarketRegime.LOW_VOLATILITY
                else:
                    regime = MarketRegime.SIDEWAYS
            else:  # Moderate trend
                if abs(ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1] < 0.01:
                    # Moving averages are very close = range bound
                    regime = MarketRegime.RANGE_BOUND
                elif is_breakout:
                    regime = MarketRegime.BREAKOUT
                elif crossed_ma:
                    regime = MarketRegime.REVERSAL
                else:
                    # Default to sideways if nothing else matches
                    regime = MarketRegime.SIDEWAYS
                    
            return regime
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime.SIDEWAYS  # Default to sideways on error
            
    def get_adjusted_thresholds(self, market_regime: MarketRegime, volatility_factor: float) -> Tuple[float, float]:
        """
        Get adjusted buy and sell thresholds based on market conditions and volatility
        
        Args:
            market_regime: Current market regime
            volatility_factor: Volatility factor
            
        Returns:
            Tuple of (buy_threshold, sell_threshold)
        """
        # Get regime adjustments
        regime_adj = self.regime_adjustments.get(market_regime, {"buy": 1.0, "sell": 1.0})
        
        # Apply adjustments to base thresholds
        adjusted_buy = self.buy_threshold * regime_adj["buy"] * volatility_factor
        adjusted_sell = self.sell_threshold * regime_adj["sell"] * volatility_factor
        
        return (adjusted_buy, adjusted_sell)
        
    def score_signal(self, signal_data: Dict[str, Any], market_regime: MarketRegime = None) -> Dict[str, Any]:
        """
        Score a set of indicator signals and determine buy/sell actions
        
        Args:
            signal_data: Dictionary with indicator signal data
            market_regime: Optional market regime (if not provided, will be neutral)
            
        Returns:
            Dictionary with scored signals and actions
        """
        if market_regime is None:
            market_regime = MarketRegime.SIDEWAYS
            
        # Get volatility factor
        volatility_factor = 1.0  # Default if not calculated
        
        # Get adjusted thresholds
        buy_threshold, sell_threshold = self.get_adjusted_thresholds(market_regime, volatility_factor)
        
        result = signal_data.copy()
        
        # Add market regime and threshold info
        result['market_regime'] = market_regime.value
        result['buy_threshold'] = buy_threshold
        result['sell_threshold'] = sell_threshold
        result['volatility_factor'] = volatility_factor
        
        # Add scores from weighted indicators
        if 'buy_signals' in signal_data and 'sell_signals' in signal_data:
            # Apply indicator-specific weights based on performance and market regime
            for signal_type in ['buy_signals', 'sell_signals']:
                for i, signal in enumerate(result[signal_type]):
                    indicator_name = signal.get('indicator', '')
                    
                    if indicator_name in self.indicators and indicator_name in self.performance_metrics:
                        # Get indicator config
                        indicator = self.indicators[indicator_name]
                        
                        # Get performance metrics
                        metrics = self.performance_metrics[indicator_name]
                        
                        # Calculate weight adjustments
                        performance_factor = metrics['accuracy'] * 1.5  # 1.0 to 1.5 scale
                        
                        # Get regime-specific weight if available
                        regime_factor = indicator['condition_weights'].get(market_regime, 1.0)
                        
                        # Apply adjustments to signal weight
                        original_weight = signal.get('effective_weight', 1.0)
                        adjusted_weight = original_weight * performance_factor * regime_factor
                        
                        # Update signal with new weight
                        result[signal_type][i]['effective_weight'] = adjusted_weight
                        result[signal_type][i]['performance_factor'] = performance_factor
                        result[signal_type][i]['regime_factor'] = regime_factor
        
        # Recalculate buy/sell scores based on adjusted weights
        buy_score = 0
        for signal in result.get('buy_signals', []):
            buy_score += signal.get('effective_weight', 0)
            
        sell_score = 0
        for signal in result.get('sell_signals', []):
            sell_score += signal.get('effective_weight', 0)
            
        # Update scores in result
        result['buy_score'] = buy_score
        result['sell_score'] = sell_score
        
        # Determine final signal based on thresholds
        result['buy_signal'] = buy_score >= buy_threshold and buy_score > sell_score
        result['sell_signal'] = sell_score >= sell_threshold and sell_score > buy_score
        result['neutral_signal'] = not (result['buy_signal'] or result['sell_signal'])
        
        return result 