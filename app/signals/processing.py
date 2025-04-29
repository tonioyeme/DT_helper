import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

# Import from scoring.py to ensure we're using the same enums
from app.signals.scoring import MarketRegime, SignalStrength

# These enum classes are now imported from scoring.py
"""
class MarketRegime(Enum):
    BULL_TREND = "bull_trend"       # Strong uptrend
    BEAR_TREND = "bear_trend"       # Strong downtrend
    SIDEWAYS = "sideways"           # Sideways/choppy market
    HIGH_VOLATILITY = "high_volatility"  # High volatility period
    LOW_VOLATILITY = "low_volatility"    # Low volatility period
    REVERSAL = "reversal"           # Potential trend reversal
"""

class AdvancedSignalProcessor:
    """
    Advanced signal processing system that dynamically adjusts weights based on
    market regimes, performs multi-timeframe confirmation, and provides confidence metrics.
    """
    
    def __init__(self, logger=None):
        """Initialize the signal processor with default settings"""
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize base weights
        self._base_weights = {
            'trend': 0.35,      # Trend indicators (EMAs, MAs, etc.)
            'momentum': 0.25,   # Momentum indicators (RSI, MACD, etc.)
            'volume': 0.20,     # Volume indicators (OBV, Volume Profile)
            'volatility': 0.20  # Volatility indicators (ATR, Bollinger Bands)
        }
        
        # Regime-specific weight modifiers
        self._regime_modifiers = {
            MarketRegime.BULL_TREND: {
                'trend': 1.2,      # Increase trend importance in bull markets
                'momentum': 1.1,   # Slightly increase momentum importance
                'volume': 0.9,     # Slightly decrease volume importance
                'volatility': 0.8  # Decrease volatility importance
            },
            MarketRegime.BEAR_TREND: {
                'trend': 1.2,      # Increase trend importance in bear markets
                'momentum': 1.0,   # Normal momentum importance
                'volume': 1.1,     # Slightly increase volume importance
                'volatility': 0.9  # Slightly decrease volatility importance
            },
            MarketRegime.SIDEWAYS: {
                'trend': 0.7,      # Decrease trend importance in sideways markets
                'momentum': 1.1,   # Increase momentum importance
                'volume': 1.0,     # Normal volume importance
                'volatility': 1.3  # Increase volatility importance
            },
            MarketRegime.HIGH_VOLATILITY: {
                'trend': 0.8,      # Decrease trend importance in volatile markets
                'momentum': 0.9,   # Decrease momentum importance
                'volume': 1.1,     # Increase volume importance
                'volatility': 1.5  # Significantly increase volatility importance
            },
            MarketRegime.LOW_VOLATILITY: {
                'trend': 1.3,      # Increase trend importance in low volatility
                'momentum': 1.2,   # Increase momentum importance
                'volume': 0.8,     # Decrease volume importance
                'volatility': 0.5  # Significantly decrease volatility importance
            },
            MarketRegime.REVERSAL: {
                'trend': 0.8,      # Decrease trend importance during reversals
                'momentum': 1.3,   # Significantly increase momentum importance
                'volume': 1.2,     # Increase volume importance
                'volatility': 1.1  # Slightly increase volatility importance
            }
        }
        
        # Threshold values for signal strength classification
        self._strength_thresholds = {
            SignalStrength.WEAK: 0.3,       # Minimum threshold for a weak signal
            SignalStrength.MODERATE: 0.5,   # Threshold for moderate signal
            SignalStrength.STRONG: 0.7,     # Threshold for strong signal
            SignalStrength.VERY_STRONG: 0.85 # Threshold for very strong signal
        }
        
        # Performance tracking for adaptive weight adjustments
        self._performance_history = {
            'trend': {'correct': 0, 'total': 0},
            'momentum': {'correct': 0, 'total': 0},
            'volume': {'correct': 0, 'total': 0},
            'volatility': {'correct': 0, 'total': 0}
        }
        
        # Time decay factor for performance history (recent performance matters more)
        self._time_decay = 0.95
    
    def detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime based on price action, volatility, and trend strength
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            MarketRegime: Detected market regime
        """
        if data is None or len(data) < 20:
            return MarketRegime.SIDEWAYS  # Default to sideways if insufficient data
        
        try:
            # Calculate key metrics for regime detection
            
            # 1. Trend strength using ADX if available, otherwise use EMA relationships
            if 'adx' in data.columns:
                adx_value = data['adx'].iloc[-1]
                trend_strength = adx_value / 50.0  # Normalize to 0-1 range (ADX max is ~100)
            else:
                # Use EMA relationships as a proxy for trend strength
                ema_short = data['ema20'].iloc[-10:] if 'ema20' in data.columns else data['close'].rolling(20).mean().iloc[-10:]
                ema_long = data['ema50'].iloc[-10:] if 'ema50' in data.columns else data['close'].rolling(50).mean().iloc[-10:]
                
                # Calculate average distance between EMAs as percentage of price
                avg_distance = abs(ema_short.mean() - ema_long.mean()) / ema_long.mean()
                trend_strength = min(1.0, avg_distance * 20)  # Scale to 0-1 range
            
            # 2. Volatility assessment
            if 'atr' in data.columns:
                # Use ATR relative to its moving average
                recent_atr = data['atr'].iloc[-5:].mean()
                historical_atr = data['atr'].iloc[-20:].mean()
                volatility_ratio = recent_atr / historical_atr if historical_atr > 0 else 1.0
            else:
                # Calculate simple volatility measure using price range
                recent_ranges = (data['high'].iloc[-5:] - data['low'].iloc[-5:]) / data['close'].iloc[-5:]
                historical_ranges = (data['high'].iloc[-20:] - data['low'].iloc[-20:]) / data['close'].iloc[-20:]
                volatility_ratio = recent_ranges.mean() / historical_ranges.mean() if historical_ranges.mean() > 0 else 1.0
            
            # 3. Trend direction
            if 'ema20' in data.columns and 'ema50' in data.columns:
                bullish = data['ema20'].iloc[-1] > data['ema50'].iloc[-1]
            else:
                # Use price relative to moving averages
                sma20 = data['close'].rolling(20).mean().iloc[-1]
                sma50 = data['close'].rolling(50).mean().iloc[-1]
                bullish = sma20 > sma50
            
            # 4. Detect potential reversal
            if 'rsi' in data.columns:
                overbought = data['rsi'].iloc[-1] > 70
                oversold = data['rsi'].iloc[-1] < 30
                
                # Divergence (simplified)
                price_higher_high = data['high'].iloc[-5:].max() > data['high'].iloc[-10:-5].max()
                price_lower_low = data['low'].iloc[-5:].min() < data['low'].iloc[-10:-5].min()
                
                rsi_higher_high = data['rsi'].iloc[-5:].max() > data['rsi'].iloc[-10:-5].max()
                rsi_lower_low = data['rsi'].iloc[-5:].min() < data['rsi'].iloc[-10:-5].min()
                
                bullish_divergence = price_lower_low and not rsi_lower_low
                bearish_divergence = price_higher_high and not rsi_higher_high
                
                potential_reversal = (bullish and overbought and bearish_divergence) or \
                                    (not bullish and oversold and bullish_divergence)
            else:
                # Simplified reversal detection without RSI
                potential_reversal = False
            
            # Determine regime based on the metrics calculated
            if potential_reversal:
                return MarketRegime.REVERSAL
            elif volatility_ratio > 1.5:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility_ratio < 0.7:
                return MarketRegime.LOW_VOLATILITY
            elif trend_strength > 0.6:  # Strong trend
                return MarketRegime.BULL_TREND if bullish else MarketRegime.BEAR_TREND
            else:  # Weak trend
                return MarketRegime.SIDEWAYS
                
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime.SIDEWAYS
    
    def update_performance_metrics(self, 
                                category: str, 
                                correct: bool,
                                apply_decay: bool = True) -> None:
        """
        Update performance metrics for a signal category
        
        Args:
            category: Signal category ('trend', 'momentum', 'volume', 'volatility')
            correct: Whether the signal was correct
            apply_decay: Whether to apply time decay to past performance
        """
        if category not in self._performance_history:
            return
            
        metrics = self._performance_history[category]
        
        # Apply time decay to past counts if requested
        if apply_decay:
            metrics['correct'] *= self._time_decay
            metrics['total'] *= self._time_decay
            
        # Update with new data
        metrics['correct'] += 1 if correct else 0
        metrics['total'] += 1
    
    def get_adjusted_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get weights adjusted for current market regime and historical performance
        
        Args:
            regime: Detected market regime
            
        Returns:
            Dictionary with adjusted weights for each signal category
        """
        adjusted_weights = {}
        
        # Get regime-specific modifiers
        modifiers = self._regime_modifiers.get(regime, {
            'trend': 1.0,
            'momentum': 1.0,
            'volume': 1.0,
            'volatility': 1.0
        })
        
        # Calculate performance adjustments
        performance_adjustments = {}
        for category, metrics in self._performance_history.items():
            if metrics['total'] > 10:  # Only adjust if we have sufficient data
                accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0.5
                # Scale to range 0.8-1.2 centered around 0.5 accuracy
                performance_adjustments[category] = 0.8 + (accuracy * 0.8)
            else:
                performance_adjustments[category] = 1.0
        
        # Apply both regime and performance adjustments to base weights
        for category, base_weight in self._base_weights.items():
            regime_modifier = modifiers.get(category, 1.0)
            perf_modifier = performance_adjustments.get(category, 1.0)
            
            adjusted_weights[category] = base_weight * regime_modifier * perf_modifier
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(adjusted_weights.values())
        if weight_sum > 0:
            for category in adjusted_weights:
                adjusted_weights[category] /= weight_sum
        
        return adjusted_weights
    
    def calculate_trend_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate trend score component using multiple trend indicators
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Series with trend scores from 0 to 1
        """
        trend_scores = pd.Series(0.0, index=data.index)
        
        try:
            components = []
            
            # EMA relationships (multiple timeframes)
            if all(col in data.columns for col in ['ema20', 'ema50', 'ema200']):
                # Strong uptrend: EMA20 > EMA50 > EMA200
                strong_uptrend = (data['ema20'] > data['ema50']) & (data['ema50'] > data['ema200'])
                components.append(strong_uptrend.astype(float))
                
                # Moderate uptrend: EMA20 > EMA50, but EMA50 < EMA200 (early trend)
                moderate_uptrend = (data['ema20'] > data['ema50']) & (data['ema50'] <= data['ema200'])
                components.append(moderate_uptrend.astype(float) * 0.6)
                
                # Weak uptrend: Only EMA20 rising, others flat or down
                ema20_rising = data['ema20'] > data['ema20'].shift(3)
                weak_uptrend = ema20_rising & ~strong_uptrend & ~moderate_uptrend
                components.append(weak_uptrend.astype(float) * 0.3)
            else:
                # Fallback if EMAs aren't available - use SMA crossovers
                sma20 = data['close'].rolling(20).mean()
                sma50 = data['close'].rolling(50).mean()
                sma200 = data['close'].rolling(200).mean() if len(data) >= 200 else sma50 * 0.9  # Fallback
                
                strong_uptrend = (sma20 > sma50) & (sma50 > sma200)
                components.append(strong_uptrend.astype(float))
                
                moderate_uptrend = (sma20 > sma50) & (sma50 <= sma200)
                components.append(moderate_uptrend.astype(float) * 0.6)
            
            # Price in relation to EMAs/SMAs
            if 'ema200' in data.columns:
                price_above_ema200 = data['close'] > data['ema200']
                components.append(price_above_ema200.astype(float) * 0.5)
            
            # ADX trend strength if available
            if 'adx' in data.columns:
                # ADX > 25 indicates strong trend, normalize to 0-1 range
                adx_strength = np.clip((data['adx'] - 15) / 35, 0, 1)
                components.append(adx_strength)
            
            # Supertrend if available
            if 'supertrend' in data.columns:
                supertrend_bullish = data['close'] > data['supertrend']
                components.append(supertrend_bullish.astype(float) * 0.7)
            
            # Combine all components with equal weighting
            if components:
                combined_score = sum(components) / len(components)
                trend_scores = combined_score
            
            return trend_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating trend score: {str(e)}")
            return trend_scores
    
    def calculate_momentum_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum score component using multiple momentum indicators
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Series with momentum scores from 0 to 1
        """
        momentum_scores = pd.Series(0.0, index=data.index)
        
        try:
            components = []
            
            # RSI
            if 'rsi' in data.columns:
                # Map RSI to a 0-1 score with centerpoint at 50
                # Below 30: 0 to 0.3 (oversold, bullish)
                # 30-50: 0.3 to 0.5 (neutral to bullish)
                # 50-70: 0.5 to 0.7 (neutral to bearish)
                # 70+: 0.7 to 1.0 (overbought, bearish)
                rsi_score = np.clip((data['rsi'] - 30) / 40, 0, 1)
                
                # For buy signals, we want RSI coming OUT of oversold (30-50 range)
                rsi_bullish = ((data['rsi'] > 30) & (data['rsi'] < 50) & 
                              (data['rsi'] > data['rsi'].shift(1)))
                
                # For this component, remap to: 0 = not bullish, 1 = bullish
                components.append(rsi_bullish.astype(float))
                
                # Add RSI value component (0.3 weight)
                components.append(1 - rsi_score * 0.3)  # Lower RSI = higher buy score
            
            # MACD
            if all(col in data.columns for col in ['macd', 'macd_signal']):
                # MACD crossing above signal line (bullish)
                macd_cross_up = (data['macd'] > data['macd_signal']) & (data['macd'].shift(1) <= data['macd_signal'].shift(1))
                components.append(macd_cross_up.astype(float) * 0.8)
                
                # MACD above signal line (bullish, but less strong than crossing)
                macd_above = (data['macd'] > data['macd_signal'])
                components.append(macd_above.astype(float) * 0.4)
                
                # MACD histogram increasing (bullish momentum)
                if 'macd_hist' in data.columns:
                    macd_hist_rising = (data['macd_hist'] > 0) & (data['macd_hist'] > data['macd_hist'].shift(1))
                    components.append(macd_hist_rising.astype(float) * 0.5)
            
            # Stochastic
            if all(col in data.columns for col in ['stoch_k', 'stoch_d']):
                # Stochastic %K crossing above %D from below 20 (strong bullish)
                stoch_cross_up_oversold = ((data['stoch_k'] > data['stoch_d']) & 
                                         (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1)) &
                                         (data['stoch_k'].shift(1) < 20))
                components.append(stoch_cross_up_oversold.astype(float) * 0.9)
                
                # Stochastic %K crossing above %D (moderate bullish)
                stoch_cross_up = ((data['stoch_k'] > data['stoch_d']) & 
                                (data['stoch_k'].shift(1) <= data['stoch_d'].shift(1)))
                components.append(stoch_cross_up.astype(float) * 0.5)
                
                # Stochastic coming out of oversold (< 20 to > 20)
                stoch_exit_oversold = (data['stoch_k'] > 20) & (data['stoch_k'].shift(1) <= 20)
                components.append(stoch_exit_oversold.astype(float) * 0.7)
            
            # Combine all components with equal weighting
            if components:
                combined_score = sum(components) / len(components)
                momentum_scores = combined_score
            
            return momentum_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {str(e)}")
            return momentum_scores
    
    def calculate_volume_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volume score component using volume indicators and patterns
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Series with volume scores from 0 to 1
        """
        volume_scores = pd.Series(0.0, index=data.index)
        
        try:
            components = []
            
            # Basic volume surge (volume significantly above average)
            vol_ratio = data['volume'] / data['volume'].rolling(20).mean()
            vol_surge = vol_ratio > 1.5
            components.append(vol_surge.astype(float) * 0.7)
            
            # Volume increasing trend (smoothed)
            vol_5d_avg = data['volume'].rolling(5).mean()
            vol_20d_avg = data['volume'].rolling(20).mean()
            vol_trend_up = vol_5d_avg > vol_20d_avg
            components.append(vol_trend_up.astype(float) * 0.5)
            
            # OBV trend
            if 'obv' in data.columns:
                obv_increasing = data['obv'] > data['obv'].shift(3)
                components.append(obv_increasing.astype(float) * 0.8)
                
                # OBV divergence with price (bullish)
                price_down = data['close'].rolling(5).apply(lambda x: x[0] > x[-1], raw=True)
                obv_up = data['obv'].rolling(5).apply(lambda x: x[0] < x[-1], raw=True)
                bullish_divergence = price_down & obv_up
                components.append(bullish_divergence.astype(float) * 0.9)
            
            # Money Flow Index (if available)
            if 'mfi' in data.columns:
                # MFI coming out of oversold (bullish)
                mfi_exit_oversold = (data['mfi'] > 20) & (data['mfi'].shift(1) <= 20)
                components.append(mfi_exit_oversold.astype(float) * 0.8)
                
                # MFI bullish level
                mfi_bullish = (data['mfi'] > 20) & (data['mfi'] < 50) & (data['mfi'] > data['mfi'].shift(1))
                components.append(mfi_bullish.astype(float) * 0.6)
            
            # Volume-price relationship
            up_days = data['close'] > data['close'].shift(1)
            down_days = data['close'] < data['close'].shift(1)
            
            # Bullish: Volume higher on up days
            up_vol = data['volume'].where(up_days, 0).rolling(5).sum()
            down_vol = data['volume'].where(down_days, 0).rolling(5).sum()
            
            # Ratio of up-day volume to down-day volume (higher is more bullish)
            with np.errstate(divide='ignore', invalid='ignore'):
                vol_ratio = up_vol / down_vol
            vol_ratio = vol_ratio.fillna(1.0)  # Default to neutral
            bullish_vol_trend = vol_ratio > 1.2  # At least 20% more volume on up days
            components.append(bullish_vol_trend.astype(float) * 0.7)
            
            # Combine all components with equal weighting
            if components:
                combined_score = sum(components) / len(components)
                volume_scores = combined_score
            
            return volume_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating volume score: {str(e)}")
            return volume_scores
    
    def calculate_volatility_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volatility score component using volatility indicators
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Series with volatility scores from 0 to 1 (higher = more favorable volatility)
        """
        volatility_scores = pd.Series(0.0, index=data.index)
        
        try:
            components = []
            
            # ATR analysis
            if 'atr' in data.columns:
                # Normalize ATR by price level
                norm_atr = data['atr'] / data['close']
                
                # Compare to historical ATR levels
                atr_ratio = norm_atr / norm_atr.rolling(20).mean()
                
                # Score based on ATR ratio (0.8-1.2 range is ideal, not too volatile, not too quiet)
                moderate_volatility = (atr_ratio >= 0.8) & (atr_ratio <= 1.2)
                components.append(moderate_volatility.astype(float) * 0.7)
                
                # Decreasing volatility can be bullish in uptrends
                decreasing_volatility = norm_atr < norm_atr.shift(3)
                components.append(decreasing_volatility.astype(float) * 0.4)
            
            # Bollinger Bands
            if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                # Bollinger Band width (normalized)
                bb_width = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
                
                # Contracting bands (often precedes breakout)
                contracting_bands = bb_width < bb_width.shift(5)
                components.append(contracting_bands.astype(float) * 0.5)
                
                # Price near middle band (stable)
                price_near_middle = abs(data['close'] - data['bb_middle']) < (data['bb_upper'] - data['bb_middle']) * 0.3
                components.append(price_near_middle.astype(float) * 0.6)
                
                # Price bouncing off lower band (bullish)
                lower_band_bounce = (data['close'] > data['bb_lower'] * 1.01) & (data['close'].shift(1) <= data['bb_lower'] * 1.01)
                components.append(lower_band_bounce.astype(float) * 0.8)
            
            # Keltner Channels (if available)
            if all(col in data.columns for col in ['kc_upper', 'kc_lower']):
                # Price within Keltner Channels (stable)
                price_within_kc = (data['close'] > data['kc_lower']) & (data['close'] < data['kc_upper'])
                components.append(price_within_kc.astype(float) * 0.5)
            
            # Combine all components with equal weighting
            if components:
                combined_score = sum(components) / len(components)
                volatility_scores = combined_score
            
            return volatility_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility score: {str(e)}")
            return volatility_scores
    
    def calculate_signal_score(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate comprehensive signal scores with dynamic weights
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            Dictionary with signal scores and components
        """
        # Initialize result dictionary
        result = {}
        
        # Skip if we don't have sufficient data
        if data is None or len(data) < 20:
            self.logger.warning("Insufficient data for signal scoring")
            return result
        
        try:
            # Detect market regime
            regime = self.detect_market_regime(data)
            result['market_regime'] = regime
            
            # Get dynamically adjusted weights for the current regime
            adjusted_weights = self.get_adjusted_weights(regime)
            result['weights'] = adjusted_weights
            
            # Calculate component scores
            trend_scores = self.calculate_trend_score(data)
            momentum_scores = self.calculate_momentum_score(data)
            volume_scores = self.calculate_volume_score(data)
            volatility_scores = self.calculate_volatility_score(data)
            
            # Store component scores
            result['trend_score'] = trend_scores
            result['momentum_score'] = momentum_scores
            result['volume_score'] = volume_scores
            result['volatility_score'] = volatility_scores
            
            # Calculate final weighted score
            final_score = (
                trend_scores * adjusted_weights['trend'] +
                momentum_scores * adjusted_weights['momentum'] +
                volume_scores * adjusted_weights['volume'] +
                volatility_scores * adjusted_weights['volatility']
            )
            
            # Normalize to 0-1 range
            final_score = np.clip(final_score, 0, 1)
            result['signal_score'] = final_score
            
            # Classify signal strength
            strength_series = pd.Series(index=data.index, dtype='object')
            
            for idx, score in final_score.items():
                if score >= self._strength_thresholds[SignalStrength.VERY_STRONG]:
                    strength_series[idx] = SignalStrength.VERY_STRONG
                elif score >= self._strength_thresholds[SignalStrength.STRONG]:
                    strength_series[idx] = SignalStrength.STRONG
                elif score >= self._strength_thresholds[SignalStrength.MODERATE]:
                    strength_series[idx] = SignalStrength.MODERATE
                elif score >= self._strength_thresholds[SignalStrength.WEAK]:
                    strength_series[idx] = SignalStrength.WEAK
                else:
                    strength_series[idx] = None
            
            result['signal_strength'] = strength_series
            
            # Calculate confidence metric (volatility-adjusted consistency)
            confidence = 0.7 + (volatility_scores * 0.3)  # Base confidence + volatility adjustment
            
            # Reduce confidence for mixed signals
            signal_agreement = (
                (trend_scores > 0.5) & 
                (momentum_scores > 0.5) & 
                (volume_scores > 0.5)
            ).astype(float)
            
            # Final confidence score
            confidence = confidence * (0.7 + 0.3 * signal_agreement)
            result['confidence'] = confidence
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating signal score: {str(e)}")
            return result

    def calculate_multi_timeframe_score(self, 
                                        data_dict: Dict[str, pd.DataFrame], 
                                        primary_tf: str = None) -> Dict[str, Any]:
        """
        Calculate signal scores across multiple timeframes with weighted consensus
        
        Args:
            data_dict: Dictionary mapping timeframe names to DataFrames
            primary_tf: Primary timeframe to use (defaults to shortest available)
            
        Returns:
            Dictionary with multi-timeframe signal scores and metrics
        """
        result = {}
        
        if not data_dict:
            return result
            
        try:
            # Timeframe weights (longer timeframes have more weight)
            tf_weights = {
                '1m': 0.5,   # 1-minute
                '5m': 0.6,   # 5-minute
                '15m': 0.7,  # 15-minute
                '30m': 0.8,  # 30-minute
                '1h': 0.9,   # 1-hour
                '4h': 1.0,   # 4-hour
                '1d': 1.1,   # Daily
                '1w': 1.2    # Weekly
            }
            
            # Select primary timeframe if not specified
            if primary_tf is None or primary_tf not in data_dict:
                # Default to shortest available timeframe
                timeframes = list(data_dict.keys())
                primary_tf = timeframes[0] if timeframes else None
            
            if not primary_tf:
                return result
                
            # Calculate signal scores for each timeframe
            tf_scores = {}
            for tf, data in data_dict.items():
                tf_scores[tf] = self.calculate_signal_score(data)
            
            # Get primary timeframe scores
            primary_scores = tf_scores.get(primary_tf, {})
            if not primary_scores:
                return result
                
            # Copy primary timeframe results as base
            result = primary_scores.copy()
            
            # Calculate consensus score across timeframes
            signal_scores = []
            weights = []
            
            for tf, scores in tf_scores.items():
                if 'signal_score' in scores:
                    signal = scores['signal_score'].iloc[-1] if len(scores['signal_score']) > 0 else 0
                    weight = tf_weights.get(tf, 0.8)  # Default weight if timeframe not in predefined list
                    
                    signal_scores.append(signal)
                    weights.append(weight)
            
            # Calculate weighted average if we have scores
            if signal_scores and weights:
                weighted_sum = sum(s * w for s, w in zip(signal_scores, weights))
                weight_sum = sum(weights)
                consensus_score = weighted_sum / weight_sum if weight_sum > 0 else 0
                
                result['consensus_score'] = consensus_score
                
                # Determine if there's alignment across timeframes
                alignment_threshold = 0.15  # Maximum deviation for "aligned" signals
                deviations = [abs(s - consensus_score) for s in signal_scores]
                max_deviation = max(deviations) if deviations else 0
                
                result['aligned'] = max_deviation <= alignment_threshold
                result['max_deviation'] = max_deviation
                
                # Multi-timeframe confidence adjustment
                if 'confidence' in result:
                    # Increase confidence if signals are aligned across timeframes
                    if result['aligned']:
                        result['confidence'] = np.minimum(result['confidence'] * 1.2, 1.0)
                    else:
                        # Decrease confidence if there's significant disagreement
                        result['confidence'] = result['confidence'] * (1.0 - min(0.5, max_deviation))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating multi-timeframe score: {str(e)}")
            return result

# Create a utility function for easier usage
def calculate_advanced_signal_score(data: pd.DataFrame) -> pd.Series:
    """
    Simplified interface for calculating advanced signal scores
    
    Args:
        data: DataFrame with price and indicator data
        
    Returns:
        Series with signal scores from 0 to 1
    """
    processor = AdvancedSignalProcessor()
    result = processor.calculate_signal_score(data)
    return result.get('signal_score', pd.Series(0, index=data.index))

# Multi-timeframe version of the utility function
def calculate_multi_timeframe_signal_score(data_dict: Dict[str, pd.DataFrame], 
                                          primary_tf: str = None) -> Dict[str, Any]:
    """
    Simplified interface for calculating multi-timeframe signal scores
    
    Args:
        data_dict: Dictionary mapping timeframe names to DataFrames
        primary_tf: Primary timeframe to use
        
    Returns:
        Dictionary with multi-timeframe signal scores
    """
    processor = AdvancedSignalProcessor()
    return processor.calculate_multi_timeframe_score(data_dict, primary_tf) 