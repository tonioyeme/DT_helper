import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import traceback

# Import from scoring.py to ensure we're using the same enums
from .scoring import MarketRegime, SignalStrength

# Import position manager for sequential exit-entry
from .position_manager import get_position_manager, SequentialPositionManager

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
        Detect current market regime based on ADX and ATR values
        
        Args:
            data: DataFrame with OHLCV and indicator data
            
        Returns:
            MarketRegime: Detected market regime
        """
        if data is None or len(data) < 20:
            return MarketRegime.SIDEWAYS  # Default to sideways if insufficient data
        
        try:
            # Calculate ADX if not already in the dataframe
            if 'adx' not in data.columns:
                # Calculate true range
                high_low = data['high'] - data['low']
                high_close = abs(data['high'] - data['close'].shift(1))
                low_close = abs(data['low'] - data['close'].shift(1))
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                
                # Calculate directional movement
                plus_dm = data['high'].diff()
                minus_dm = data['low'].diff()
                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm > 0] = 0
                minus_dm = abs(minus_dm)
                
                # Smooth DM and TR using Wilder's smoothing method
                period = 14
                plus_di = 100 * plus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum()
                minus_di = 100 * minus_dm.rolling(window=period).sum() / tr.rolling(window=period).sum()
                
                # Calculate DX and ADX
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                data['adx'] = dx.rolling(window=period).mean()
                
                # Also calculate +DI and -DI for directional information
                data['+di'] = plus_di
                data['-di'] = minus_di
            
            # Calculate ATR if not already in the dataframe
            if 'atr' not in data.columns:
                high_low = data['high'] - data['low']
                high_close = abs(data['high'] - data['close'].shift(1))
                low_close = abs(data['low'] - data['close'].shift(1))
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                data['atr'] = tr.rolling(window=14).mean()
            
            # Calculate normalized ATR (as percentage of price)
            norm_atr = data['atr'] / data['close']
            current_norm_atr = norm_atr.iloc[-1]
            
            # Get historical volatility average (30-day)
            historical_norm_atr = norm_atr.iloc[-30:].mean()
            volatility_ratio = current_norm_atr / historical_norm_atr if historical_norm_atr > 0 else 1.0
            
            # Get current ADX, +DI, -DI values
            current_adx = data['adx'].iloc[-1]
            
            # Get +DI and -DI if available, otherwise default to 0
            current_plus_di = data.get('+di', pd.Series(0)).iloc[-1]
            current_minus_di = data.get('-di', pd.Series(0)).iloc[-1]
            
            # Determine trend direction
            if 'ema20' in data.columns and 'ema50' in data.columns:
                trend_direction = 1 if data['ema20'].iloc[-1] > data['ema50'].iloc[-1] else -1
            else:
                sma20 = data['close'].rolling(20).mean().iloc[-1]
                sma50 = data['close'].rolling(50).mean().iloc[-1]
                trend_direction = 1 if sma20 > sma50 else -1
            
            # Detect breakout conditions
            price_range_20d = data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
            avg_range = price_range_20d.iloc[-5:].mean()
            latest_move = abs(data['close'].iloc[-1] - data['close'].iloc[-5])
            is_breakout = latest_move > (0.7 * avg_range)
            
            # Enhanced regime classification based on ADX and volatility
            # 1. Strong Trend (ADX > 40)
            if current_adx > 40:
                if current_plus_di > current_minus_di:
                    regime = MarketRegime.BULL_TREND
                else:
                    regime = MarketRegime.BEAR_TREND
                
            # 2. Moderate Trend (ADX 25-40)
            elif current_adx > 25:
                if is_breakout:
                    regime = MarketRegime.BREAKOUT
                elif current_plus_di > current_minus_di:
                    regime = MarketRegime.BULL_TREND
                else:
                    regime = MarketRegime.BEAR_TREND
                
            # 3. Weak Trend (ADX 20-25)
            elif current_adx > 20:
                if is_breakout:
                    regime = MarketRegime.BREAKOUT
                elif volatility_ratio > 1.3:
                    regime = MarketRegime.HIGH_VOLATILITY
                elif volatility_ratio < 0.7:
                    regime = MarketRegime.LOW_VOLATILITY
                else:
                    regime = MarketRegime.RANGE_BOUND
                
            # 4. No Trend (ADX < 20)
            else:
                if volatility_ratio > 1.5:
                    regime = MarketRegime.HIGH_VOLATILITY
                elif volatility_ratio < 0.6:
                    regime = MarketRegime.LOW_VOLATILITY
                elif is_breakout:
                    regime = MarketRegime.BREAKOUT
                elif abs(data['close'].pct_change(5).iloc[-1]) > 0.02:  # 2% change in last 5 bars
                    regime = MarketRegime.REVERSAL
                else:
                    regime = MarketRegime.SIDEWAYS
            
            # Log the detected regime and key metrics
            self.logger.info(f"Detected market regime: {regime.value} (ADX: {current_adx:.2f}, +DI: {current_plus_di:.2f}, -DI: {current_minus_di:.2f}, Volatility ratio: {volatility_ratio:.2f})")
            return regime
            
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
            signal_agreement = pd.Series(False, index=trend_scores.index)
            
            # Safely check each component and create a mask for agreement
            valid_indices = ~trend_scores.isna() & ~momentum_scores.isna() & ~volume_scores.isna()
            agreement_indices = valid_indices & (trend_scores > 0.5) & (momentum_scores > 0.5) & (volume_scores > 0.5)
            signal_agreement.loc[agreement_indices] = True
            
            # Convert to float for arithmetic operations
            signal_agreement = signal_agreement.astype(float)
            
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
        
        if not isinstance(data_dict, dict) or len(data_dict) == 0:
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
                primary_tf = timeframes[0] if len(timeframes) > 0 else None
            
            if primary_tf is None:
                return result
                
            # Calculate signal scores for each timeframe
            tf_scores = {}
            for tf, data in data_dict.items():
                tf_scores[tf] = self.calculate_signal_score(data)
            
            # Get primary timeframe scores
            primary_scores = tf_scores.get(primary_tf, {})
            if len(primary_scores) == 0:
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
            if len(signal_scores) > 0 and len(weights) > 0:
                weighted_sum = sum(s * w for s, w in zip(signal_scores, weights))
                weight_sum = sum(weights)
                consensus_score = weighted_sum / weight_sum if weight_sum > 0 else 0
                
                result['consensus_score'] = consensus_score
                
                # Determine if there's alignment across timeframes
                alignment_threshold = 0.15  # Maximum deviation for "aligned" signals
                deviations = [abs(s - consensus_score) for s in signal_scores]
                max_deviation = max(deviations) if len(deviations) > 0 else 0
                
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

    def get_adaptive_indicator_parameters(self, data: pd.DataFrame, regime: MarketRegime = None) -> Dict[str, Dict[str, Any]]:
        """
        Get indicator parameters adapted to the current market regime
        
        Args:
            data: DataFrame with price data
            regime: Current market regime (if None, will be detected)
            
        Returns:
            Dictionary of indicator parameters adjusted for the market regime
        """
        # Detect market regime if not provided
        if regime is None:
            regime = self.detect_market_regime(data)
        
        # Base parameters for different indicators
        base_params = {
            'ema': {
                'fast_period': 9,
                'slow_period': 21,
                'signal_period': 9
            },
            'macd': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'rsi': {
                'period': 14,
                'overbought': 70,
                'oversold': 30
            },
            'stochastic': {
                'k_period': 14,
                'd_period': 3,
                'overbought': 80,
                'oversold': 20
            },
            'bollinger': {
                'period': 20,
                'std_dev': 2.0
            },
            'atr': {
                'period': 14,
                'multiplier': 2.0
            },
            'adx': {
                'period': 14,
                'threshold': 25
            },
            'hull_ma': {
                'period': 16
            },
            'ttm_squeeze': {
                'bb_period': 20,
                'bb_std_dev': 2.0,
                'kc_period': 20,
                'kc_multiplier': 1.5
            }
        }
        
        # Adjust parameters based on market regime
        adjusted_params = base_params.copy()
        
        if regime == MarketRegime.BULL_TREND:
            # In bull trend, use faster EMAs and less sensitive RSI
            adjusted_params['ema']['fast_period'] = 8
            adjusted_params['ema']['slow_period'] = 17
            adjusted_params['macd']['fast_period'] = 8
            adjusted_params['macd']['slow_period'] = 17
            adjusted_params['rsi']['overbought'] = 75
            adjusted_params['rsi']['oversold'] = 40
            adjusted_params['bollinger']['std_dev'] = 2.2
            adjusted_params['hull_ma']['period'] = 12
            
        elif regime == MarketRegime.BEAR_TREND:
            # In bear trend, use faster EMAs and less sensitive RSI
            adjusted_params['ema']['fast_period'] = 8
            adjusted_params['ema']['slow_period'] = 17
            adjusted_params['macd']['fast_period'] = 8
            adjusted_params['macd']['slow_period'] = 17
            adjusted_params['rsi']['overbought'] = 60
            adjusted_params['rsi']['oversold'] = 25
            adjusted_params['bollinger']['std_dev'] = 2.2
            adjusted_params['hull_ma']['period'] = 12
            
        elif regime == MarketRegime.BREAKOUT:
            # For breakouts, use faster indicators and wider Bollinger Bands
            adjusted_params['ema']['fast_period'] = 5
            adjusted_params['ema']['slow_period'] = 15
            adjusted_params['macd']['fast_period'] = 8
            adjusted_params['macd']['slow_period'] = 17
            adjusted_params['rsi']['period'] = 10
            adjusted_params['bollinger']['std_dev'] = 2.5
            adjusted_params['atr']['multiplier'] = 2.5
            adjusted_params['hull_ma']['period'] = 9
            
        elif regime == MarketRegime.HIGH_VOLATILITY:
            # For high volatility, use slower indicators and wider bands
            adjusted_params['ema']['fast_period'] = 12
            adjusted_params['ema']['slow_period'] = 30
            adjusted_params['macd']['fast_period'] = 16
            adjusted_params['macd']['slow_period'] = 32
            adjusted_params['rsi']['period'] = 21
            adjusted_params['rsi']['overbought'] = 75
            adjusted_params['rsi']['oversold'] = 25
            adjusted_params['bollinger']['std_dev'] = 3.0
            adjusted_params['atr']['multiplier'] = 3.0
            adjusted_params['ttm_squeeze']['bb_std_dev'] = 2.5
            adjusted_params['ttm_squeeze']['kc_multiplier'] = 2.0
            
        elif regime == MarketRegime.LOW_VOLATILITY:
            # For low volatility, use faster indicators and tighter bands
            adjusted_params['ema']['fast_period'] = 7
            adjusted_params['ema']['slow_period'] = 14
            adjusted_params['macd']['fast_period'] = 8
            adjusted_params['macd']['slow_period'] = 17
            adjusted_params['rsi']['period'] = 10
            adjusted_params['rsi']['overbought'] = 65
            adjusted_params['rsi']['oversold'] = 35
            adjusted_params['bollinger']['std_dev'] = 1.5
            adjusted_params['atr']['multiplier'] = 1.5
            adjusted_params['ttm_squeeze']['bb_std_dev'] = 1.5
            adjusted_params['ttm_squeeze']['kc_multiplier'] = 1.0
            
        elif regime == MarketRegime.SIDEWAYS or regime == MarketRegime.RANGE_BOUND:
            # For range-bound markets, use standard params but more sensitive overbought/oversold
            adjusted_params['rsi']['overbought'] = 65
            adjusted_params['rsi']['oversold'] = 35
            adjusted_params['stochastic']['overbought'] = 75
            adjusted_params['stochastic']['oversold'] = 25
            adjusted_params['bollinger']['std_dev'] = 1.8
            
        elif regime == MarketRegime.REVERSAL:
            # For potential reversals, use medium-speed indicators
            adjusted_params['ema']['fast_period'] = 7
            adjusted_params['ema']['slow_period'] = 21
            adjusted_params['macd']['fast_period'] = 10
            adjusted_params['macd']['slow_period'] = 20
            adjusted_params['rsi']['period'] = 12
            adjusted_params['bollinger']['std_dev'] = 2.0
            adjusted_params['hull_ma']['period'] = 12
        
        return adjusted_params

def process_signals_with_position_manager(signals: pd.DataFrame, 
                                          data: pd.DataFrame) -> pd.DataFrame:
    """
    Process signals using the position manager to ensure proper exit-entry sequencing
    
    Args:
        signals: DataFrame with signals
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with processed signals incorporating exit-entry sequencing
    """
    if signals.empty or data.empty:
        return signals
    
    # Get the position manager instance (singleton)
    position_manager = get_position_manager()
    
    # Reset the position manager for fresh signal processing
    position_manager.__init__()
    
    # Track processed state in the DataFrame
    signals['position_active'] = False
    signals['exit_pending'] = False
    signals['entry_queued'] = False
    signals['position_direction'] = None
    signals['execution_price'] = None
    
    # Process each signal time step
    for i, idx in enumerate(signals.index):
        # Determine signal type at this timestep
        if signals.loc[idx, 'buy_signal']:
            signal_type = 'buy'
        elif signals.loc[idx, 'sell_signal']:
            signal_type = 'sell'
        else:
            signal_type = 'neutral'
        
        # Get current price
        current_price = signals.loc[idx, 'signal_price'] if 'signal_price' in signals.columns else data.loc[idx, 'close']
        
        # Extract data up to this point for liquidity/volatility calculation
        bar_data = data.loc[:idx] if idx in data.index else None
        
        # Process through position manager
        position_manager.process_signal(signal_type, bar_data, current_price, idx)
        
        # Update signal DataFrame with position manager state
        current_position = position_manager.get_current_position()
        if current_position:
            signals.loc[idx, 'position_active'] = True
            signals.loc[idx, 'position_direction'] = current_position.direction
            signals.loc[idx, 'execution_price'] = current_position.entry_price
        
        signals.loc[idx, 'exit_pending'] = position_manager.is_pending_exit()
        signals.loc[idx, 'entry_queued'] = position_manager.has_queued_signals()
        
        # Handle signal suppression during exit sequences
        if position_manager.is_pending_exit():
            # Disable conflicting entry signals during exit sequence
            if current_position and current_position.direction == 'buy':
                signals.loc[idx, 'sell_signal'] = False
            elif current_position and current_position.direction == 'sell':
                signals.loc[idx, 'buy_signal'] = False
        
        # Get trading state for monitoring
        if i % 10 == 0:  # Log every 10 bars to reduce verbosity
            trading_state = position_manager.get_trading_state()
            print(f"Position state at {idx}: {trading_state}")
    
    return signals

def calculate_advanced_signal_score(data: pd.DataFrame) -> pd.Series:
    """Enhanced signal score calculation with advanced signal processor"""
    processor = AdvancedSignalProcessor()
    scores = processor.calculate_signal_score(data)
    
    # Create signals DataFrame
    signals = pd.DataFrame(index=data.index)
    signals['buy_score'] = scores['buy']
    signals['sell_score'] = scores['sell']
    signals['buy_signal'] = scores['buy'] > 0.6
    signals['sell_signal'] = scores['sell'] > 0.6
    
    # Process signals with position manager for exit-entry sequencing
    signals = process_signals_with_position_manager(signals, data)
    
    return signals

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

class TrendAdaptiveStrategyMatrix:
    """
    A matrix of trend-adaptive trading strategies that adjust based on detected market conditions.
    This provides optimal strategies, confirmation indicators, and position sizing for each market regime.
    """
    
    def __init__(self, logger=None):
        """Initialize the strategy matrix with default settings"""
        self.logger = logger or logging.getLogger(__name__)
        
        # Define the base strategy matrix
        self._strategy_matrix = {
            # Strong Bull Trend strategies
            MarketRegime.BULL_TREND: {
                'primary_strategy': 'VWAP_PULLBACK',
                'description': 'Buy on pullbacks to VWAP in strong uptrend',
                'confirmation_indicators': [
                    {'indicator': 'MACD_HISTOGRAM', 'condition': 'GREATER_THAN', 'value': 0},
                    {'indicator': 'EMA_SLOPE', 'condition': 'POSITIVE', 'timeframe': '1h'}
                ],
                'position_sizing': 1.5,  # 150% of normal position size
                'stop_loss_multiplier': 1.5,
                'target_multiplier': 2.5,
                'timeframes': ['1h', '15m', '5m']
            },
            
            # Weak Bull Trend strategies
            MarketRegime.RANGE_BOUND: {
                'primary_strategy': 'EMA_CLOUD_BREAKOUT',
                'description': 'Buy on breakouts above EMA cloud resistance',
                'confirmation_indicators': [
                    {'indicator': 'VOLUME', 'condition': 'GREATER_THAN', 'value': 'VOLUME_20_AVG'},
                    {'indicator': 'RSI', 'condition': 'BETWEEN', 'value': [40, 60]}
                ],
                'position_sizing': 1.0,  # 100% of normal position size
                'stop_loss_multiplier': 1.0,
                'target_multiplier': 2.0,
                'timeframes': ['15m', '5m']
            },
            
            # Strong Bear Trend strategies
            MarketRegime.BEAR_TREND: {
                'primary_strategy': 'RALLY_SHORT',
                'description': 'Short rallies in downtrends when price approaches resistance',
                'confirmation_indicators': [
                    {'indicator': 'RSI', 'condition': 'LESS_THAN', 'value': 60},
                    {'indicator': 'VOLUME_DECLINING', 'condition': 'TRUE'}
                ],
                'position_sizing': 1.25,  # 125% of normal position size
                'stop_loss_multiplier': 1.2,
                'target_multiplier': 2.0,
                'timeframes': ['1h', '15m']
            },
            
            # Range-Bound Market strategies
            MarketRegime.SIDEWAYS: {
                'primary_strategy': 'OPTIONS_STRADDLE',
                'description': 'Options straddle to profit from breakout in either direction',
                'confirmation_indicators': [
                    {'indicator': 'IV_PERCENTILE', 'condition': 'GREATER_THAN', 'value': 30},
                    {'indicator': 'BB_WIDTH', 'condition': 'LESS_THAN', 'value': 'BB_WIDTH_20_AVG'}
                ],
                'position_sizing': 0.75,  # 75% of normal position size
                'stop_loss_multiplier': 1.0,
                'target_multiplier': 1.5,
                'timeframes': ['1h', '15m']
            },
            
            # High Volatility Market strategies
            MarketRegime.HIGH_VOLATILITY: {
                'primary_strategy': 'ATR_CHANNEL_BREAKOUT',
                'description': 'Trade breakouts of ATR-based channels in volatile conditions',
                'confirmation_indicators': [
                    {'indicator': 'BB_WIDTH', 'condition': 'GREATER_THAN', 'value': 0.02},
                    {'indicator': 'ADX', 'condition': 'GREATER_THAN', 'value': 25}
                ],
                'position_sizing': 0.5,  # 50% of normal position size
                'stop_loss_multiplier': 2.0,
                'target_multiplier': 3.0,
                'timeframes': ['1h', '15m', '5m']
            },
            
            # Breakout condition strategies
            MarketRegime.BREAKOUT: {
                'primary_strategy': 'MOMENTUM_BREAKOUT',
                'description': 'Trade in the direction of strong breakouts from consolidation',
                'confirmation_indicators': [
                    {'indicator': 'VOLUME_SURGE', 'condition': 'GREATER_THAN', 'value': 2.0},
                    {'indicator': 'ATR_EXPANSION', 'condition': 'GREATER_THAN', 'value': 1.5}
                ],
                'position_sizing': 1.0,  # 100% of normal position size
                'stop_loss_multiplier': 1.5,
                'target_multiplier': 2.5,
                'timeframes': ['1h', '15m', '5m']
            },
            
            # Low Volatility Market strategies
            MarketRegime.LOW_VOLATILITY: {
                'primary_strategy': 'MEAN_REVERSION',
                'description': 'Mean reversion trades in low volatility environments',
                'confirmation_indicators': [
                    {'indicator': 'RSI_EXTREME', 'condition': 'TRUE'},
                    {'indicator': 'BOLLINGER_BAND_BOUNCE', 'condition': 'TRUE'}
                ],
                'position_sizing': 0.75,  # 75% of normal position size
                'stop_loss_multiplier': 0.8,
                'target_multiplier': 1.5,
                'timeframes': ['15m', '5m']
            },
            
            # Reversal Market strategies
            MarketRegime.REVERSAL: {
                'primary_strategy': 'DIVERGENCE_REVERSAL',
                'description': 'Trade reversals based on price-indicator divergences',
                'confirmation_indicators': [
                    {'indicator': 'RSI_DIVERGENCE', 'condition': 'TRUE'},
                    {'indicator': 'VOLUME_CLIMAX', 'condition': 'TRUE'}
                ],
                'position_sizing': 0.6,  # 60% of normal position size
                'stop_loss_multiplier': 1.2,
                'target_multiplier': 2.0,
                'timeframes': ['1h', '15m', '5m']
            }
        }
        
        # Trend confirmation protocol steps
        self._confirmation_protocol = [
            {'step': 'CHECK_1H_SMA_SLOPE', 'params': {'period': 200, 'lookback': 5}},
            {'step': 'VERIFY_15M_MACD_CROSSOVER', 'params': {'direction': 'SAME_AS_TREND'}},
            {'step': 'CONFIRM_5M_VOLUME_PRICE_ACTION', 'params': {'min_volume_ratio': 1.2}}
        ]
    
    def get_strategy_for_regime(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get the optimal strategy configuration for a specific market regime
        
        Args:
            regime: Detected market regime
            
        Returns:
            Dictionary with strategy configuration
        """
        # Use the detected regime or fall back to SIDEWAYS if not found
        strategy = self._strategy_matrix.get(regime, self._strategy_matrix[MarketRegime.SIDEWAYS])
        
        self.logger.info(f"Selected strategy for {regime.value}: {strategy['primary_strategy']} - {strategy['description']}")
        
        return strategy
    
    def get_confirmation_protocol(self) -> List[Dict[str, Any]]:
        """
        Get the trend confirmation protocol steps
        
        Returns:
            List of confirmation protocol steps
        """
        return self._confirmation_protocol.copy()
    
    def adjust_strategy_for_conditions(self, 
                                     strategy: Dict[str, Any], 
                                     conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust strategy parameters based on specific market conditions
        
        Args:
            strategy: Base strategy dictionary
            conditions: Dictionary with current market conditions
            
        Returns:
            Adjusted strategy dictionary
        """
        adjusted_strategy = strategy.copy()
        
        # Adjust position sizing based on specific conditions
        if 'volatility_percentile' in conditions:
            vol_percentile = conditions['volatility_percentile']
            
            # Reduce position size in extremely high volatility
            if vol_percentile > 90:
                adjusted_strategy['position_sizing'] *= 0.6
            # Increase position size in very low volatility for range-bound strategies
            elif vol_percentile < 10 and strategy['primary_strategy'] in ['MEAN_REVERSION', 'OPTIONS_STRADDLE']:
                adjusted_strategy['position_sizing'] *= 1.2
        
        # Adjust stop loss based on recent price action
        if 'atr_percentile' in conditions:
            atr_percentile = conditions['atr_percentile']
            
            # Widen stops in high ATR environments
            if atr_percentile > 80:
                adjusted_strategy['stop_loss_multiplier'] *= 1.3
            # Tighten stops in low ATR environments
            elif atr_percentile < 20:
                adjusted_strategy['stop_loss_multiplier'] *= 0.8
        
        # Adjust for time of day if provided
        if 'market_session' in conditions:
            session = conditions['market_session']
            
            # Reduce position sizing in less favorable sessions
            if session == 'midday':
                adjusted_strategy['position_sizing'] *= 0.8
            # Increase position sizing in more volatile sessions
            elif session in ['open', 'close']:
                adjusted_strategy['position_sizing'] *= 1.2
        
        return adjusted_strategy
    
    def execute_confirmation_protocol(self, 
                                    data_dict: Dict[str, pd.DataFrame],
                                    strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the trend confirmation protocol across multiple timeframes
        
        Args:
            data_dict: Dictionary mapping timeframe names to DataFrames
            strategy: Strategy configuration dictionary
            
        Returns:
            Dictionary with confirmation results
        """
        results = {
            'confirmed': False,
            'confirmation_score': 0.0,
            'steps_passed': 0,
            'details': {}
        }
        
        try:
            # Check for required timeframes
            required_timeframes = strategy['timeframes']
            available_timeframes = list(data_dict.keys())
            
            missing_timeframes = [tf for tf in required_timeframes if tf not in available_timeframes]
            if missing_timeframes:
                self.logger.warning(f"Missing required timeframes for confirmation: {missing_timeframes}")
                results['details']['missing_timeframes'] = missing_timeframes
                return results
            
            # Extract the 1-hour, 15-min, and 5-min data if available
            data_1h = data_dict.get('1h', None)
            data_15m = data_dict.get('15m', None)
            data_5m = data_dict.get('5m', None)
            
            steps_passed = 0
            confirmation_score = 0.0
            
            # Step 1: Check 1-hour SMA slope (200-period)
            if data_1h is not None and len(data_1h) >= 200:
                # Calculate 200-period SMA
                sma_200 = data_1h['close'].rolling(window=200).mean()
                
                # Calculate slope over last 5 periods
                lookback = 5
                slope = (sma_200.iloc[-1] - sma_200.iloc[-lookback]) / lookback
                
                # Normalize slope as percentage of price
                norm_slope = slope / data_1h['close'].iloc[-1]
                
                # Check if slope aligns with strategy direction
                if strategy['primary_strategy'] in ['VWAP_PULLBACK', 'EMA_CLOUD_BREAKOUT', 'MOMENTUM_BREAKOUT']:
                    # Bullish strategies need positive slope
                    if norm_slope > 0:
                        steps_passed += 1
                        confirmation_score += 0.4
                        results['details']['1h_sma_slope'] = 'POSITIVE'
                    else:
                        results['details']['1h_sma_slope'] = 'NEGATIVE'
                elif strategy['primary_strategy'] in ['RALLY_SHORT']:
                    # Bearish strategies need negative slope
                    if norm_slope < 0:
                        steps_passed += 1
                        confirmation_score += 0.4
                        results['details']['1h_sma_slope'] = 'NEGATIVE'
                    else:
                        results['details']['1h_sma_slope'] = 'POSITIVE'
                else:
                    # Other strategies just need consistency with overall trend
                    steps_passed += 1
                    confirmation_score += 0.2
                    results['details']['1h_sma_slope'] = 'SLOPE_CHECK_SKIPPED'
            
            # Step 2: Verify 15-min MACD crossover direction
            if data_15m is not None and len(data_15m) >= 35:
                # Calculate MACD components if not already in the data
                if 'macd' not in data_15m.columns or 'macd_signal' not in data_15m.columns:
                    ema12 = data_15m['close'].ewm(span=12, adjust=False).mean()
                    ema26 = data_15m['close'].ewm(span=26, adjust=False).mean()
                    macd = ema12 - ema26
                    macd_signal = macd.ewm(span=9, adjust=False).mean()
                    macd_hist = macd - macd_signal
                else:
                    macd = data_15m['macd']
                    macd_signal = data_15m['macd_signal']
                    macd_hist = data_15m.get('macd_hist', macd - macd_signal)
                
                # Check for recent crossover (within last 3 bars)
                lookback = 3
                
                # Bullish crossover: MACD crosses above signal line
                bullish_crossover = False
                for i in range(1, min(lookback + 1, len(macd))):
                    if macd.iloc[-i] > macd_signal.iloc[-i] and macd.iloc[-i-1] <= macd_signal.iloc[-i-1]:
                        bullish_crossover = True
                        break
                
                # Bearish crossover: MACD crosses below signal line
                bearish_crossover = False
                for i in range(1, min(lookback + 1, len(macd))):
                    if macd.iloc[-i] < macd_signal.iloc[-i] and macd.iloc[-i-1] >= macd_signal.iloc[-i-1]:
                        bearish_crossover = True
                        break
                
                # Check if crossover aligns with strategy direction
                if strategy['primary_strategy'] in ['VWAP_PULLBACK', 'EMA_CLOUD_BREAKOUT', 'MOMENTUM_BREAKOUT']:
                    # Bullish strategies need bullish crossover
                    if bullish_crossover:
                        steps_passed += 1
                        confirmation_score += 0.3
                        results['details']['15m_macd_crossover'] = 'BULLISH'
                    else:
                        results['details']['15m_macd_crossover'] = 'NO_BULLISH_CROSSOVER'
                elif strategy['primary_strategy'] in ['RALLY_SHORT']:
                    # Bearish strategies need bearish crossover
                    if bearish_crossover:
                        steps_passed += 1
                        confirmation_score += 0.3
                        results['details']['15m_macd_crossover'] = 'BEARISH'
                    else:
                        results['details']['15m_macd_crossover'] = 'NO_BEARISH_CROSSOVER'
                else:
                    # For other strategies, any recent crossover is good
                    if bullish_crossover or bearish_crossover:
                        steps_passed += 1
                        confirmation_score += 0.2
                        results['details']['15m_macd_crossover'] = 'CROSSOVER_DETECTED'
                    else:
                        results['details']['15m_macd_crossover'] = 'NO_CROSSOVER'
            
            # Step 3: Confirm with 5-min volume-weighted price action
            if data_5m is not None and len(data_5m) >= 20:
                # Calculate volume metrics
                avg_volume = data_5m['volume'].rolling(window=20).mean()
                recent_vol_ratio = data_5m['volume'].iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
                
                # Calculate price action
                is_bullish_bar = data_5m['close'].iloc[-1] > data_5m['open'].iloc[-1]
                is_bearish_bar = data_5m['close'].iloc[-1] < data_5m['open'].iloc[-1]
                
                # Check if volume-price action aligns with strategy direction
                volume_confirmed = recent_vol_ratio >= 1.2
                
                if strategy['primary_strategy'] in ['VWAP_PULLBACK', 'EMA_CLOUD_BREAKOUT', 'MOMENTUM_BREAKOUT']:
                    # Bullish strategies need bullish price action with volume
                    if is_bullish_bar and volume_confirmed:
                        steps_passed += 1
                        confirmation_score += 0.3
                        results['details']['5m_volume_price'] = 'BULLISH_VOLUME_CONFIRMED'
                    elif is_bullish_bar:
                        steps_passed += 1
                        confirmation_score += 0.1
                        results['details']['5m_volume_price'] = 'BULLISH_LOW_VOLUME'
                    else:
                        results['details']['5m_volume_price'] = 'NOT_BULLISH'
                        
                elif strategy['primary_strategy'] in ['RALLY_SHORT']:
                    # Bearish strategies need bearish price action with volume
                    if is_bearish_bar and volume_confirmed:
                        steps_passed += 1
                        confirmation_score += 0.3
                        results['details']['5m_volume_price'] = 'BEARISH_VOLUME_CONFIRMED'
                    elif is_bearish_bar:
                        steps_passed += 1
                        confirmation_score += 0.1
                        results['details']['5m_volume_price'] = 'BEARISH_LOW_VOLUME'
                    else:
                        results['details']['5m_volume_price'] = 'NOT_BEARISH'
                else:
                    # For other strategies, just check for above-average volume
                    if volume_confirmed:
                        steps_passed += 1
                        confirmation_score += 0.2
                        results['details']['5m_volume_price'] = 'VOLUME_CONFIRMED'
                    else:
                        results['details']['5m_volume_price'] = 'LOW_VOLUME'
            
            # Final confirmation results
            results['steps_passed'] = steps_passed
            results['confirmation_score'] = confirmation_score
            
            # Confirm if we passed at least 2 out of 3 steps and score is high enough
            results['confirmed'] = steps_passed >= 2 and confirmation_score >= 0.5
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error in trend confirmation protocol: {str(e)}")
            results['details']['error'] = str(e)
            return results

    def get_strategy_recommendations(self, 
                                   data_dict: Dict[str, pd.DataFrame], 
                                   regime: MarketRegime = None,
                                   conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get comprehensive strategy recommendations based on market conditions
        
        Args:
            data_dict: Dictionary mapping timeframe names to DataFrames
            regime: Detected market regime (if None, will be detected)
            conditions: Additional market conditions
            
        Returns:
            Dictionary with strategy recommendations
        """
        try:
            # Use provided conditions or initialize empty dict
            market_conditions = conditions or {}
            
            # Detect market regime if not provided
            if regime is None:
                processor = AdvancedSignalProcessor()
                # Use primary timeframe data for regime detection
                primary_tf = min(data_dict.keys()) if data_dict is not None and isinstance(data_dict, dict) and len(data_dict) > 0 else None
                if primary_tf and primary_tf in data_dict:
                    regime = processor.detect_market_regime(data_dict[primary_tf])
                else:
                    # Default to sideways if we can't detect
                    regime = MarketRegime.SIDEWAYS
            
            # Get base strategy for the detected regime
            base_strategy = self.get_strategy_for_regime(regime)
            
            # Adjust strategy based on current market conditions
            adjusted_strategy = self.adjust_strategy_for_conditions(base_strategy, market_conditions)
            
            # Execute trend confirmation protocol
            confirmation_results = self.execute_confirmation_protocol(data_dict, adjusted_strategy)
            
            # Compile final recommendations
            recommendations = {
                'market_regime': regime,
                'primary_strategy': adjusted_strategy['primary_strategy'],
                'description': adjusted_strategy['description'],
                'position_sizing': adjusted_strategy['position_sizing'],
                'stop_loss_multiplier': adjusted_strategy['stop_loss_multiplier'],
                'target_multiplier': adjusted_strategy['target_multiplier'],
                'confirmation': confirmation_results,
                'ready_to_trade': confirmation_results['confirmed'],
                'timeframes': adjusted_strategy['timeframes'],
                'key_indicators': adjusted_strategy['confirmation_indicators']
            }
            
            # Add strategy-specific trading rules
            if adjusted_strategy['primary_strategy'] == 'VWAP_PULLBACK':
                recommendations['entry_rules'] = [
                    "Wait for price to pull back to VWAP",
                    "Confirm bullish reversal candle at VWAP",
                    "Enter on close of confirmation candle",
                    "Place stop loss below the low of reversal candle"
                ]
            elif adjusted_strategy['primary_strategy'] == 'EMA_CLOUD_BREAKOUT':
                recommendations['entry_rules'] = [
                    "Wait for price to break above EMA cloud resistance",
                    "Confirm breakout with volume increase",
                    "Enter on close above resistance",
                    "Place stop loss below the EMA cloud"
                ]
            elif adjusted_strategy['primary_strategy'] == 'RALLY_SHORT':
                recommendations['entry_rules'] = [
                    "Wait for price to rally to resistance area",
                    "Confirm bearish reversal pattern",
                    "Enter on break of low of reversal candle",
                    "Place stop loss above the high of reversal pattern"
                ]
            else:
                recommendations['entry_rules'] = [
                    "Follow confirmation indicators for entry signals",
                    "Size position according to recommended position sizing",
                    "Use the appropriate stop loss multiplier for risk management"
                ]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting strategy recommendations: {str(e)}")
            return {
                'market_regime': regime or MarketRegime.SIDEWAYS,
                'primary_strategy': 'ERROR',
                'error': str(e),
                'ready_to_trade': False
            }

# Utility function for easier usage
def get_trend_adaptive_strategy(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Simplified interface for getting trend-adaptive strategy recommendations
    
    Args:
        data_dict: Dictionary mapping timeframe names to DataFrames
        
    Returns:
        Dictionary with strategy recommendations
    """
    matrix = TrendAdaptiveStrategyMatrix()
    return matrix.get_strategy_recommendations(data_dict) 