import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import logging
from datetime import datetime, timedelta

# Import indicators if needed
from indicators import calculate_atr
from indicators.momentum import calculate_rsi, calculate_macd
from .exit_signals import get_vix_data

class SignalStrength(Enum):
    """Signal strength classifications"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

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
    UNKNOWN = "unknown"

class IndicatorCategory(Enum):
    """Indicator categories for weighting in the scoring system"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"
    PATTERN = "pattern"
    SUPPORT_RESISTANCE = "support_resistance"

def calculate_dynamic_signal_score(indicators, market_regime):
    """
    Calculate a dynamic signal score based on current market conditions
    
    Args:
        indicators (dict): Dictionary of indicator signals and values
        market_regime (MarketRegime): Current market regime
        
    Returns:
        float: Signal score between 0 and 1
    """
    # Base weights for different indicator categories
    category_weights = {
        IndicatorCategory.TREND: 1.0,
        IndicatorCategory.MOMENTUM: 0.8,
        IndicatorCategory.VOLATILITY: 0.6,
        IndicatorCategory.VOLUME: 0.7,
        IndicatorCategory.OSCILLATOR: 0.8,
        IndicatorCategory.PATTERN: 0.9,
        IndicatorCategory.SUPPORT_RESISTANCE: 1.1
    }
    
    # Adjust weights based on market regime
    if market_regime == MarketRegime.BULL_TREND:
        category_weights[IndicatorCategory.TREND] *= 1.3
        category_weights[IndicatorCategory.MOMENTUM] *= 1.2
        category_weights[IndicatorCategory.PATTERN] *= 1.1
    elif market_regime == MarketRegime.BEAR_TREND:
        category_weights[IndicatorCategory.VOLATILITY] *= 1.2
        category_weights[IndicatorCategory.VOLUME] *= 1.2
        category_weights[IndicatorCategory.SUPPORT_RESISTANCE] *= 1.3
    elif market_regime == MarketRegime.SIDEWAYS or market_regime == MarketRegime.RANGE_BOUND:
        category_weights[IndicatorCategory.OSCILLATOR] *= 1.3
        category_weights[IndicatorCategory.SUPPORT_RESISTANCE] *= 1.2
        category_weights[IndicatorCategory.TREND] *= 0.7
    elif market_regime == MarketRegime.HIGH_VOLATILITY:
        category_weights[IndicatorCategory.VOLATILITY] *= 1.4
        category_weights[IndicatorCategory.VOLUME] *= 1.3
        category_weights[IndicatorCategory.PATTERN] *= 0.8
    elif market_regime == MarketRegime.LOW_VOLATILITY:
        category_weights[IndicatorCategory.PATTERN] *= 1.2
        category_weights[IndicatorCategory.OSCILLATOR] *= 1.2
        category_weights[IndicatorCategory.VOLATILITY] *= 0.7
    elif market_regime == MarketRegime.BREAKOUT:
        category_weights[IndicatorCategory.VOLUME] *= 1.4
        category_weights[IndicatorCategory.MOMENTUM] *= 1.3
        category_weights[IndicatorCategory.OSCILLATOR] *= 0.7
    elif market_regime == MarketRegime.REVERSAL:
        category_weights[IndicatorCategory.PATTERN] *= 1.4
        category_weights[IndicatorCategory.MOMENTUM] *= 1.2
        category_weights[IndicatorCategory.TREND] *= 0.6
    
    # Calculate weighted score for each indicator
    total_weight = 0
    total_score = 0
    
    # Keep track of categories with signals
    categories_with_signals = set()
    
    for indicator, signal in indicators.items():
        weight = category_weights.get(indicator.category, 1.0) * indicator.base_weight
        total_weight += weight
        signal_strength = getattr(signal, 'strength', 0.5)  # Default to 0.5 if strength not available
        total_score += signal_strength * weight
        categories_with_signals.add(indicator.category)
    
    # Normalize score between 0 and 1
    if total_weight == 0:
        return 0
    
    # Base final score
    final_score = total_score / total_weight
    
    # Bonus for having signals from multiple categories (signal confirmation)
    category_count_bonus = min(0.2, len(categories_with_signals) * 0.05)  # Up to 0.2 bonus
    
    # Apply bonus but keep maximum score at 1.0
    final_score = min(1.0, final_score + category_count_bonus)
    
    return final_score

class DynamicSignalScorer:
    """
    Enhanced signal scoring system with regime-specific validation,
    volatility-adaptive scoring, and multi-layer validation
    """
    def __init__(self, symbol=None, config=None):
        self.symbol = symbol
        self.config = config or {}
        self.logger = logging.getLogger("DynamicSignalScorer")
        self.prev_scores = {}
        self.score_history = []
        
        # Configure SPY-specific optimizations
        if symbol and symbol.upper() == "SPY":
            self.use_spy_optimizations = True
            self.spy_config = self._get_spy_config()
        else:
            self.use_spy_optimizations = False
            
    def _get_spy_config(self) -> Dict[str, Any]:
        """Get SPY-specific scoring configuration"""
        return {
            'volatility_regimes': {
                'low': {
                    'atr_multiplier': 1.8,
                    'score_threshold': 4.2,
                    'position_size': 1.2
                },
                'normal': {
                    'atr_multiplier': 2.2,
                    'score_threshold': 5.0,
                    'position_size': 1.0
                },
                'high': {
                    'atr_multiplier': 3.0,
                    'score_threshold': 6.5,
                    'position_size': 0.7
                }
            },
            'session_adjustments': {
                'pre_open': {'threshold_adj': 1.3, 'weight_adj': 0.7},
                'power_hour': {'threshold_adj': 0.8, 'weight_adj': 1.2},
                'close': {'threshold_adj': 1.1, 'weight_adj': 0.9}
            },
            'weight_profile': {
                'technical': 0.45,
                'regime': 0.35,
                'volatility': 0.20
            }
        }
        
    def detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime based on multiple indicators
        
        Args:
            data (pd.DataFrame): OHLCV price data with indicators
            
        Returns:
            MarketRegime: Detected market regime
        """
        if len(data) < 20:
            return MarketRegime.UNKNOWN
        
        # Calculate needed indicators if not present
        if 'rsi' not in data.columns:
            data['rsi'] = calculate_rsi(data)
            
        if 'atr' not in data.columns:
            data['atr'] = calculate_atr(data)
            
        # Calculate intermediate metrics
        # 1. Trend detection
        if 'ema20' not in data.columns:
            data['ema20'] = data['close'].ewm(span=20).mean()
            
        if 'ema50' not in data.columns:
            data['ema50'] = data['close'].ewm(span=50).mean()
            
        # 2. Volatility metrics
        if 'atr_ratio' not in data.columns:
            data['atr_ratio'] = data['atr'] / data['close']
            data['atr_norm'] = data['atr_ratio'] / data['atr_ratio'].rolling(100).mean()
            
        # Extract latest values
        latest = data.iloc[-1]
        
        # Get VIX data or use ATR-derived volatility
        vix_value = data['vix'].iloc[-1] if 'vix' in data.columns else get_vix_data(data)
        
        # 1. Determine if trending
        trend_score = 0
        
        # Check price relative to EMAs
        if latest['close'] > latest['ema20'] > latest['ema50']:
            trend_score += 2  # Strong uptrend
        elif latest['close'] > latest['ema20']:
            trend_score += 1  # Moderate uptrend
        elif latest['close'] < latest['ema20'] < latest['ema50']:
            trend_score -= 2  # Strong downtrend
        elif latest['close'] < latest['ema20']:
            trend_score -= 1  # Moderate downtrend
            
        # Check RSI for trend confirmation
        if latest['rsi'] > 60:
            trend_score += 1  # Bullish
        elif latest['rsi'] < 40:
            trend_score -= 1  # Bearish
        
        # 2. Determine volatility regime
        is_high_volatility = (vix_value > 25) or (latest['atr_norm'] > 1.5)
        is_low_volatility = (vix_value < 15) or (latest['atr_norm'] < 0.7)
        
        # Determine regime based on trend and volatility
        if trend_score >= 2 and not is_high_volatility:
            return MarketRegime.BULL_TREND
        elif trend_score <= -2 and not is_high_volatility:
            return MarketRegime.BEAR_TREND
        elif is_high_volatility:
            return MarketRegime.HIGH_VOLATILITY
        elif is_low_volatility:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.SIDEWAYS
            
    def validate_regime_signals(self, data: pd.DataFrame, signals: pd.DataFrame, 
                               regime: Optional[MarketRegime] = None) -> pd.DataFrame:
        """
        Validate signals based on the current market regime
        
        Args:
            data (pd.DataFrame): OHLCV price data
            signals (pd.DataFrame): Signal dataframe
            regime (MarketRegime, optional): Market regime, if None will be detected
            
        Returns:
            pd.DataFrame: Validated signals
        """
        # Detect regime if not provided
        if regime is None:
            regime = self.detect_market_regime(data)
            
        # Initialize validated signals
        validated = signals.copy()
        
        # Add regime column
        validated['market_regime'] = regime.value
        
        # Define regime indicators that need to confirm
        regime_indicators = {
            'trend': ['ema_cross', 'adx', 'price_velocity'],
            'volatility': ['atr', 'bb_width', 'vix'],
            'volume': ['obv', 'relative_volume', 'vwap']
        }
        
        # Set confirmation thresholds by regime
        confirmation_thresholds = {
            MarketRegime.BULL_TREND: {
                'buy': 0.6,   # 60% must confirm for buy signals
                'sell': 0.8   # 80% must confirm for sell signals (more strict)
            },
            MarketRegime.BEAR_TREND: {
                'buy': 0.8,   # 80% must confirm for buy signals (more strict)
                'sell': 0.6   # 60% must confirm for sell signals
            },
            MarketRegime.HIGH_VOLATILITY: {
                'buy': 0.7,   # 70% must confirm for buy signals
                'sell': 0.7   # 70% must confirm for sell signals
            },
            MarketRegime.LOW_VOLATILITY: {
                'buy': 0.5,   # 50% must confirm for buy signals (less strict)
                'sell': 0.5   # 50% must confirm for sell signals (less strict)
            },
            MarketRegime.SIDEWAYS: {
                'buy': 0.65,  # 65% must confirm for buy signals
                'sell': 0.65  # 65% must confirm for sell signals
            },
            MarketRegime.UNKNOWN: {
                'buy': 0.75,  # 75% must confirm for buy signals (conservative)
                'sell': 0.75  # 75% must confirm for sell signals (conservative)
            }
        }
        
        # Get thresholds for current regime
        thresholds = confirmation_thresholds.get(regime, confirmation_thresholds[MarketRegime.UNKNOWN])
        
        # Validate each signal
        validated['regime_validated_buy'] = False
        validated['regime_validated_sell'] = False
        
        # Function to check if enough indicators confirm the signal
        def check_confirmation(row, signal_type):
            if not row.get(f'{signal_type}_signal', False):
                return False
                
            # Count available indicators
            available_indicators = 0
            confirmed_indicators = 0
            
            # Check trend indicators
            for ind in regime_indicators['trend']:
                if f'{ind}_{signal_type}' in row:
                    available_indicators += 1
                    if row[f'{ind}_{signal_type}']:
                        confirmed_indicators += 1
                        
            # Check volatility indicators
            for ind in regime_indicators['volatility']:
                if f'{ind}_{signal_type}' in row:
                    available_indicators += 1
                    if row[f'{ind}_{signal_type}']:
                        confirmed_indicators += 1
                        
            # Check volume indicators
            for ind in regime_indicators['volume']:
                if f'{ind}_{signal_type}' in row:
                    available_indicators += 1
                    if row[f'{ind}_{signal_type}']:
                        confirmed_indicators += 1
            
            # Calculate confirmation ratio
            if available_indicators == 0:
                return True  # No indicators to check
                
            confirmation_ratio = confirmed_indicators / available_indicators
            return confirmation_ratio >= thresholds[signal_type]
        
        # Apply validation
        for idx in validated.index:
            validated.at[idx, 'regime_validated_buy'] = check_confirmation(validated.loc[idx], 'buy')
            validated.at[idx, 'regime_validated_sell'] = check_confirmation(validated.loc[idx], 'sell')
        
        return validated
    
    def dynamic_score_adjustment(self, base_score: float, data: pd.DataFrame) -> float:
        """
        Adjust signal score based on volatility state and other factors
        
        Args:
            base_score (float): Initial signal score
            data (pd.DataFrame): OHLCV price data
            
        Returns:
            float: Adjusted score
        """
        # Get volatility state
        vix_value = data['vix'].iloc[-1] if 'vix' in data.columns else get_vix_data(data)
        
        # Determine volatility regime
        if vix_value > 25:
            volatility_state = "high"
        elif vix_value < 15:
            volatility_state = "low"
        else:
            volatility_state = "normal"
            
        # Adjust score based on volatility regime
        if volatility_state == "high":
            score = base_score * 0.7  # Be more conservative in high volatility
        elif volatility_state == "low":
            score = base_score * 1.3  # More aggressive in low volatility
        else:
            score = base_score
            
        # Apply time-based session adjustments
        session_adj = self._get_session_adjustment(data)
        score = score * session_adj['weight_adj']
        
        # Apply performance-based decay if we have history
        if len(self.score_history) > 0:
            decay_factor = self._calculate_performance_decay()
            score = score * decay_factor
            
        return score
    
    def _get_session_adjustment(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get time-based session adjustments"""
        session_type = 'regular'
        result = {'threshold_adj': 1.0, 'weight_adj': 1.0}
        
        if len(data) == 0 or not isinstance(data.index, pd.DatetimeIndex):
            return result
            
        # Get current session time
        try:
            import pytz
            from datetime import time
            
            # Convert to Eastern time if timezone info is available
            current_time = data.index[-1].time()
            if hasattr(data.index[-1], 'tzinfo') and data.index[-1].tzinfo is not None:
                eastern = pytz.timezone('US/Eastern')
                current_time = data.index[-1].astimezone(eastern).time()
                
            # Market open (9:30-10:30): Higher volatility, lower threshold
            if time(9, 30) <= current_time < time(10, 30):
                session_type = 'pre_open'
            # Mid-day (10:30-14:30): Normal parameters
            elif time(10, 30) <= current_time < time(14, 30):
                session_type = 'regular'
            # Power hour (14:30-16:00): Higher volatility, higher threshold
            elif time(14, 30) <= current_time <= time(16, 0):
                session_type = 'power_hour'
        except:
            # In case of any error, return default values
            return result
            
        # Return session adjustments based on configuration
        if self.use_spy_optimizations and session_type in self.spy_config['session_adjustments']:
            return self.spy_config['session_adjustments'][session_type]
        else:
            # Default session adjustments if not using SPY config
            session_adjustments = {
                'pre_open': {'threshold_adj': 1.2, 'weight_adj': 0.8},
                'power_hour': {'threshold_adj': 0.9, 'weight_adj': 1.1},
                'regular': {'threshold_adj': 1.0, 'weight_adj': 1.0}
            }
            return session_adjustments.get(session_type, result)
    
    def _calculate_performance_decay(self) -> float:
        """Calculate performance-based decay factor"""
        # Default return if no history
        if len(self.score_history) < 5:
            return 1.0
            
        # Calculate performance metrics
        win_rate = sum(1 for s in self.score_history if s['outcome'] == 'win') / len(self.score_history)
        
        # Calculate decay factor
        half_life = 90  # Days
        decay_factor = 0.5 ** (1 / half_life)
        
        # Adjust decay based on win rate
        if win_rate > 0.6:  # Good performance
            return 1.0 + (1.0 - decay_factor) * 0.5  # Strengthen signals
        elif win_rate < 0.4:  # Poor performance
            return decay_factor  # Weaken signals
        
        return 1.0  # Neutral
    
    def calculate_composite_strength(self, signals: List[Dict]) -> float:
        """
        Calculate composite signal strength from multiple signals
        
        Args:
            signals (List[Dict]): List of signal dictionaries with strength
            
        Returns:
            float: Composite signal strength
        """
        if not signals:
            return 0.0
            
        return sum([
            1.5 if s.get('strength') == SignalStrength.VERY_STRONG else
            1.2 if s.get('strength') == SignalStrength.STRONG else
            1.0 if s.get('strength') == SignalStrength.MODERATE else
            0.7 if s.get('strength') == SignalStrength.WEAK else
            0.5 for s in signals
        ]) / len(signals)
    
    def dynamic_thresholds(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate dynamic signal thresholds based on market conditions
        
        Args:
            data (pd.DataFrame): OHLCV price data
            
        Returns:
            Dict[str, float]: Dynamic thresholds for signals
        """
        # Base threshold values
        base = 5.0
        
        # Volatility adjustment
        if 'atr' in data.columns:
            volatility_adj = data['atr'].iloc[-1] / data['atr'].rolling(100).mean().iloc[-1]
        else:
            volatility_adj = 1.0
            
        # Regime adjustment
        regime = self.detect_market_regime(data)
        if regime == MarketRegime.BULL_TREND:
            regime_adj_buy = 0.9  # Lower threshold for buy in bull trend
            regime_adj_sell = 1.2  # Higher threshold for sell in bull trend
        elif regime == MarketRegime.BEAR_TREND:
            regime_adj_buy = 1.2  # Higher threshold for buy in bear trend
            regime_adj_sell = 0.9  # Lower threshold for sell in bear trend
        elif regime == MarketRegime.HIGH_VOLATILITY:
            regime_adj_buy = 1.3  # Higher threshold in high volatility
            regime_adj_sell = 1.3
        else:
            regime_adj_buy = 1.0  # Default values
            regime_adj_sell = 1.0
            
        # Session adjustment
        session_adj = self._get_session_adjustment(data)
        
        # SPY-specific adjustments if enabled
        if self.use_spy_optimizations:
            vix_value = data['vix'].iloc[-1] if 'vix' in data.columns else get_vix_data(data)
            
            # Determine volatility regime for SPY
            vol_regime = 'normal'
            if vix_value > 25:
                vol_regime = 'high'
            elif vix_value < 15:
                vol_regime = 'low'
                
            # Apply SPY-specific threshold based on volatility regime
            base = self.spy_config['volatility_regimes'][vol_regime]['score_threshold']
            
        # Calculate final thresholds
        return {
            'buy': base * volatility_adj * regime_adj_buy * session_adj['threshold_adj'],
            'sell': base * volatility_adj * regime_adj_sell * session_adj['threshold_adj']
        }
    
    def generate_signal(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate final signal with multi-layer validation
        
        Args:
            data (pd.DataFrame): OHLCV price data
            signals (pd.DataFrame): Signal indicators
            
        Returns:
            Dict[str, Any]: Processed signal result with scores
        """
        # Skip if data is too short
        if len(data) < 20:
            return {
                'buy_score': 0,
                'sell_score': 0,
                'thresholds': {'buy': 5.0, 'sell': 5.0},
                'signals': signals,
                'regime': MarketRegime.UNKNOWN.value,
                'position_size': 0.5  # Conservative default
            }
            
        # 1. Detect market regime
        regime = self.detect_market_regime(data)
        
        # 2. Validate signals based on regime
        validated_signals = self.validate_regime_signals(data, signals, regime)
        
        # 3. Calculate technical score (40-45% weight)
        buy_tech_score = self._calculate_technical_score(data, validated_signals, 'buy')
        sell_tech_score = self._calculate_technical_score(data, validated_signals, 'sell')
        
        # 4. Calculate regime score (30-35% weight)
        buy_regime_score = self._calculate_regime_score(data, regime, 'buy')
        sell_regime_score = self._calculate_regime_score(data, regime, 'sell')
        
        # 5. Calculate volatility adjustment (20% weight)
        volatility_score = self._calculate_volatility_score(data)
        
        # 6. Define weights based on configuration
        if self.use_spy_optimizations:
            weights = self.spy_config['weight_profile']
        else:
            weights = {
                'technical': 0.4,
                'regime': 0.3,
                'volatility': 0.3
            }
            
        # 7. Calculate composite scores
        buy_score = (
            weights['technical'] * buy_tech_score +
            weights['regime'] * buy_regime_score +
            weights['volatility'] * volatility_score
        )
        
        sell_score = (
            weights['technical'] * sell_tech_score +
            weights['regime'] * sell_regime_score +
            weights['volatility'] * volatility_score
        )
        
        # 8. Apply dynamic adjustment
        buy_score = self.dynamic_score_adjustment(buy_score, data)
        sell_score = self.dynamic_score_adjustment(sell_score, data)
        
        # 9. Calculate dynamic thresholds
        thresholds = self.dynamic_thresholds(data)
        
        # 10. Determine position sizing
        position_size = self._calculate_position_size(data, regime)
        
        # Store for history tracking
        self.prev_scores = {
            'buy': buy_score,
            'sell': sell_score,
            'thresholds': thresholds,
            'timestamp': data.index[-1]
        }
        
        return {
            'buy_score': buy_score,
            'sell_score': sell_score,
            'thresholds': thresholds,
            'signals': validated_signals,
            'regime': regime.value,
            'position_size': position_size
        }
    
    def _calculate_technical_score(self, data: pd.DataFrame, signals: pd.DataFrame, signal_type: str) -> float:
        """Calculate technical analysis score component"""
        # Get relevant signals
        relevant_cols = [col for col in signals.columns if signal_type in col and signals[col].dtype == bool]
        
        if not relevant_cols:
            return 0.0
            
        # Calculate average signal confirmation
        signal_count = 0
        confirmed_count = 0
        
        for col in relevant_cols:
            if col == f'{signal_type}_signal':  # Skip the main signal
                continue
                
            # Count valid signals
            signal_count += 1
            if signals[col].iloc[-1]:
                confirmed_count += 1
                
        # Calculate confirmation ratio
        if signal_count == 0:
            return 0.0
            
        confirmation_ratio = confirmed_count / signal_count
        
        # Base technical score (0-10 scale)
        score = confirmation_ratio * 10
        
        # Boost if regime validated
        if f'regime_validated_{signal_type}' in signals.columns and signals[f'regime_validated_{signal_type}'].iloc[-1]:
            score *= 1.2
            
        return score
    
    def _calculate_regime_score(self, data: pd.DataFrame, regime: MarketRegime, signal_type: str) -> float:
        """Calculate regime alignment score component"""
        # Different regimes favor different signals
        regime_alignment = {
            MarketRegime.BULL_TREND: {'buy': 10.0, 'sell': 4.0},
            MarketRegime.BEAR_TREND: {'buy': 4.0, 'sell': 10.0},
            MarketRegime.SIDEWAYS: {'buy': 6.0, 'sell': 6.0},
            MarketRegime.HIGH_VOLATILITY: {'buy': 5.0, 'sell': 5.0},
            MarketRegime.LOW_VOLATILITY: {'buy': 7.0, 'sell': 7.0},
            MarketRegime.UNKNOWN: {'buy': 5.0, 'sell': 5.0}
        }
        
        # Get base score from alignment table
        base_score = regime_alignment.get(regime, regime_alignment[MarketRegime.UNKNOWN])[signal_type]
        
        # Adjust by recent price action
        if signal_type == 'buy':
            # For buy signals, stronger in uptrend
            if data['close'].iloc[-1] > data['close'].iloc[-5]:
                base_score *= 1.1
        else:  # sell signal
            # For sell signals, stronger in downtrend
            if data['close'].iloc[-1] < data['close'].iloc[-5]:
                base_score *= 1.1
                
        return base_score
    
    def _calculate_volatility_score(self, data: pd.DataFrame) -> float:
        """Calculate volatility component score"""
        # Get VIX value
        vix_value = data['vix'].iloc[-1] if 'vix' in data.columns else get_vix_data(data)
        
        # Calculate ATR ratio if available
        if 'atr' in data.columns:
            atr_ratio = data['atr'].iloc[-1] / data['close'].iloc[-1]
            atr_normalized = atr_ratio / data['atr'].rolling(100).mean().iloc[-1]
        else:
            atr_normalized = 1.0
            
        # Calculate volatility score
        if vix_value > 30 or atr_normalized > 2.0:
            # Very high volatility: reduce score
            return 3.0
        elif vix_value > 20 or atr_normalized > 1.5:
            # High volatility: slightly reduce score
            return 5.0
        elif vix_value < 12 or atr_normalized < 0.5:
            # Very low volatility: may be misleading
            return 6.0
        else:
            # Normal volatility: neutral
            return 8.0
    
    def _calculate_position_size(self, data: pd.DataFrame, regime: MarketRegime) -> float:
        """Calculate optimal position size based on conditions"""
        # Default position size
        position_size = 1.0
        
        if self.use_spy_optimizations:
            # Use SPY-specific position sizing
            vix_value = data['vix'].iloc[-1] if 'vix' in data.columns else get_vix_data(data)
            
            # Determine volatility regime
            vol_regime = 'normal'
            if vix_value > 25:
                vol_regime = 'high'
            elif vix_value < 15:
                vol_regime = 'low'
                
            # Get from config
            position_size = self.spy_config['volatility_regimes'][vol_regime]['position_size']
        else:
            # General position sizing based on regime
            if regime == MarketRegime.HIGH_VOLATILITY:
                position_size = 0.7  # Reduce position size in high volatility
            elif regime == MarketRegime.LOW_VOLATILITY:
                position_size = 1.2  # Increase position size in low volatility
                
        # Cap position size
        return min(position_size, 1.5)
        
    def update_performance(self, signal_result: Dict[str, Any], outcome: str, profit_pct: float):
        """
        Update performance history for dynamic adjustments
        
        Args:
            signal_result (Dict[str, Any]): Signal result dictionary
            outcome (str): 'win' or 'loss'
            profit_pct (float): Profit/loss percentage
        """
        self.score_history.append({
            'timestamp': signal_result.get('timestamp', pd.Timestamp.now()),
            'buy_score': signal_result.get('buy_score', 0),
            'sell_score': signal_result.get('sell_score', 0),
            'regime': signal_result.get('regime', 'unknown'),
            'outcome': outcome,
            'profit_pct': profit_pct
        })
        
        # Keep last 100 results
        if len(self.score_history) > 100:
            self.score_history = self.score_history[-100:]


# Factory function to create a signal scorer
def create_signal_scorer(symbol=None, config=None) -> DynamicSignalScorer:
    """
    Create a signal scorer instance
    
    Args:
        symbol (str, optional): Trading symbol
        config (dict, optional): Custom configuration
        
    Returns:
        DynamicSignalScorer: Signal scorer instance
    """
    return DynamicSignalScorer(symbol=symbol, config=config)


def score_signals(data: pd.DataFrame, signals: pd.DataFrame, symbol=None) -> Dict[str, Any]:
    """
    Score signals using the dynamic signal scorer
    
    Args:
        data (pd.DataFrame): OHLCV price data
        signals (pd.DataFrame): Signal dataframe
        symbol (str, optional): Trading symbol
        
    Returns:
        Dict[str, Any]: Scored signals result
    """
    # Create scorer
    scorer = create_signal_scorer(symbol=symbol)
    
    # Generate scored signals
    return scorer.generate_signal(data, signals) 