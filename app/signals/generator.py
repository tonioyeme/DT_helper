import pandas as pd
import numpy as np
from enum import Enum
import pytz
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from app.indicators import (
    calculate_ema,
    calculate_ema_cloud,
    calculate_macd,
    calculate_vwap,
    calculate_obv,
    calculate_ad_line,
    calculate_rsi,
    calculate_stochastic,
    calculate_fibonacci_sma,
    calculate_pain,
    calculate_ema_vwap_strategy,
    calculate_measured_move_volume_strategy,
    multi_indicator_confirmation
)
from app.patterns import (
    identify_doji,
    identify_hammer,
    identify_engulfing,
    identify_morning_star,
    identify_evening_star,
    # New price action patterns
    identify_head_and_shoulders,
    identify_inverse_head_and_shoulders,
    identify_double_top,
    identify_double_bottom,
    identify_triple_top,
    identify_triple_bottom,
    identify_rectangle,
    identify_channel,
    identify_triangle,
    identify_flag
)

from app.signals.layers import (
    IndicatorLayers, IndicatorLayer, Indicator, 
    IndicatorCategory, SignalType,
    create_moving_average_crossover,
    create_rsi_indicator,
    create_macd_indicator,
    create_bollinger_bands_indicator,
    create_standard_layers
)

from app.signals.timeframes import (
    TimeFramePriority, TimeFrame, TimeFrameManager,
    create_standard_timeframes
)

from app.signals.scoring import (
    DynamicSignalScorer, SignalStrength, MarketRegime
)

from app.signals.processing import (
    AdvancedSignalProcessor, calculate_advanced_signal_score,
    calculate_multi_timeframe_signal_score, MarketRegime, SignalStrength
)

class SignalGenerator:
    """
    Signal generator that integrates layered indicators, multi-timeframe analysis,
    and dynamic signal scoring
    """
    
    def __init__(self):
        """Initialize the signal generator with various components"""
        self.indicator_layers = create_standard_layers()
        self.timeframe_manager = create_standard_timeframes()
        self.signal_scorer = DynamicSignalScorer()
        self.logger = logging.getLogger(__name__)
        
        # Initialize scorer with standard indicators
        self._setup_signal_scorer()
        
    def _setup_signal_scorer(self):
        """Set up the signal scorer with default configurations"""
        # Register trend indicators
        self.signal_scorer.register_indicator(
            name="SMA 50/200",
            category=IndicatorCategory.TREND,
            base_weight=1.2,
            condition_weights={
                MarketRegime.BULL_TREND: 1.5,
                MarketRegime.BEAR_TREND: 1.5,
                MarketRegime.SIDEWAYS: 0.7,
                MarketRegime.HIGH_VOLATILITY: 0.8
            }
        )
        
        self.signal_scorer.register_indicator(
            name="EMA 9/21",
            category=IndicatorCategory.TREND,
            base_weight=1.0,
            condition_weights={
                MarketRegime.BULL_TREND: 1.1,
                MarketRegime.BEAR_TREND: 1.1,
                MarketRegime.SIDEWAYS: 0.9,
                MarketRegime.HIGH_VOLATILITY: 1.0
            }
        )
        
        # Register momentum indicators
        self.signal_scorer.register_indicator(
            name="RSI 14",
            category=IndicatorCategory.OSCILLATOR,
            base_weight=0.9,
            condition_weights={
                MarketRegime.BULL_TREND: 0.8,
                MarketRegime.BEAR_TREND: 0.8,
                MarketRegime.SIDEWAYS: 1.2,
                MarketRegime.HIGH_VOLATILITY: 0.7
            }
        )
        
        self.signal_scorer.register_indicator(
            name="MACD 12/26/9",
            category=IndicatorCategory.MOMENTUM,
            base_weight=1.0,
            condition_weights={
                MarketRegime.BULL_TREND: 1.1,
                MarketRegime.BEAR_TREND: 1.1,
                MarketRegime.SIDEWAYS: 0.9,
                MarketRegime.HIGH_VOLATILITY: 0.8
            }
        )
        
        # Register volatility indicators
        self.signal_scorer.register_indicator(
            name="BB 20/2",
            category=IndicatorCategory.VOLATILITY,
            base_weight=0.8,
            condition_weights={
                MarketRegime.BULL_TREND: 0.7,
                MarketRegime.BEAR_TREND: 0.7,
                MarketRegime.SIDEWAYS: 1.1,
                MarketRegime.HIGH_VOLATILITY: 1.2
            }
        )
        
    def add_custom_indicator(self, indicator: Indicator, layer_name: str = None):
        """
        Add a custom indicator to a specific layer
        
        Args:
            indicator: Indicator object to add
            layer_name: Name of the layer to add to (creates a new layer if not found)
        """
        if layer_name is None:
            layer_name = str(indicator.category.value).capitalize()
            
        # Check if layer exists, create if needed
        if layer_name not in self.indicator_layers.layers:
            new_layer = IndicatorLayer(name=layer_name)
            self.indicator_layers.add_layer(new_layer)
            
        # Add indicator to layer
        self.indicator_layers.layers[layer_name].add_indicator(indicator)
        
        # Register with scorer
        self.signal_scorer.register_indicator(
            name=indicator.name,
            category=indicator.category,
            base_weight=1.0  # Default weight
        )
        
        return self
        
    def set_layer_weight(self, layer_name: str, weight: float):
        """
        Set the weight for a specific indicator layer
        
        Args:
            layer_name: Name of the layer
            weight: New weight value
        """
        self.indicator_layers.set_layer_weight(layer_name, weight)
        return self
        
    def add_timeframe(self, timeframe: TimeFrame):
        """
        Add a timeframe for multi-timeframe analysis
        
        Args:
            timeframe: TimeFrame object to add
        """
        self.timeframe_manager.add_timeframe(timeframe)
        return self
        
    def set_primary_timeframe(self, timeframe_name: str):
        """
        Set the primary timeframe for analysis
        
        Args:
            timeframe_name: Name of the timeframe to set as primary
        """
        self.timeframe_manager.set_primary_timeframe(timeframe_name)
        return self
        
    def process_data(self, data: pd.DataFrame, timeframe_name: str):
        """
        Process data for a specific timeframe
        
        Args:
            data: DataFrame with OHLCV data
            timeframe_name: Name of the timeframe this data belongs to
            
        Returns:
            Dictionary with processed signals for this timeframe
        """
        if timeframe_name not in self.timeframe_manager.timeframes:
            self.logger.warning(f"Timeframe {timeframe_name} not registered, creating with default settings")
            self.timeframe_manager.add_timeframe(TimeFrame(
                name=timeframe_name,
                interval=timeframe_name,
                priority=TimeFramePriority.SECONDARY
            ))
            
        # Store data in the timeframe
        self.timeframe_manager.timeframes[timeframe_name].set_data(data)
        
        # Calculate indicators using the layered system
        indicator_results = self.indicator_layers.calculate_all(data)
        
        # Aggregate signals from all layers
        aggregated_signals = self.indicator_layers.aggregate_signals(indicator_results)
        
        # Detect market regime for the signal scorer
        market_regime = self.signal_scorer.detect_market_regime(data)
        
        # Score the signals using dynamic scoring
        scored_signals = self.signal_scorer.score_signal(
            aggregated_signals, 
            market_regime=market_regime
        )
        
        # Store signals in the timeframe
        self.timeframe_manager.timeframes[timeframe_name].set_signals(scored_signals)
        
        return scored_signals
        
    def generate_signals(self, data_dict: Dict[str, pd.DataFrame] = None):
        """
        Generate signals across all configured timeframes
        
        Args:
            data_dict: Optional dictionary mapping timeframe names to DataFrames
                      If None, uses data already stored in timeframes
                      
        Returns:
            Dictionary with consolidated multi-timeframe signals
        """
        # Process any new data
        if data_dict:
            for timeframe_name, data in data_dict.items():
                self.process_data(data, timeframe_name)
                
        # Check if we have signals for the primary timeframe
        primary_tf = self.timeframe_manager.primary_timeframe
        if primary_tf is None:
            self.logger.error("No primary timeframe set")
            return {"error": "No primary timeframe set"}
            
        # Process multi-timeframe validation
        multi_tf_signals = self.timeframe_manager.analyze_multi_timeframe_signals()
        
        # Add context
        if "error" not in multi_tf_signals:
            # Calculate signal data for building context
            multi_tf_signals['indicator_signals'] = {}
            
            # Add signal information from each indicator in the primary timeframe
            if primary_tf in self.timeframe_manager.timeframes:
                primary_data = self.timeframe_manager.timeframes[primary_tf].data
                
                if primary_data is not None:
                    for layer_name, layer in self.indicator_layers.layers.items():
                        layer_results = {}
                        
                        for indicator_name, indicator in layer.indicators.items():
                            # Calculate each indicator
                            result = indicator.calculate(primary_data)
                            layer_results[indicator_name] = {
                                'value': result.get('values', {}),
                                'buy_signal': result.get('buy_signal', False),
                                'sell_signal': result.get('sell_signal', False),
                                'strength': max(
                                    result.get('buy_strength', 0) if result.get('buy_signal', False) else 0,
                                    result.get('sell_strength', 0) if result.get('sell_signal', False) else 0
                                )
                            }
                            
                        multi_tf_signals['indicator_signals'][layer_name] = layer_results
        
        return multi_tf_signals
        
    def get_signal_explanation(self, signals: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation of the signals
        
        Args:
            signals: Dictionary with signal data from generate_signals()
            
        Returns:
            String with explanation of the signals
        """
        if "error" in signals:
            return f"Error: {signals['error']}"
            
        # Base explanation on the signal type and confirmation level
        explanation = []
        
        if signals.get("buy_signal", False):
            signal_type = "Buy"
        elif signals.get("sell_signal", False):
            signal_type = "Sell"
        else:
            signal_type = "Neutral"
            
        # Add signal type and confidence
        confidence = signals.get("confidence", 0) * 100
        explanation.append(f"{signal_type} signal with {confidence:.1f}% confidence.")
        
        # Add primary timeframe info
        primary_tf = signals.get("primary_timeframe", "unknown")
        explanation.append(f"Primary timeframe: {primary_tf}")
        
        # Add confirmations
        confirmations = signals.get("confirmed_by", [])
        if confirmations:
            tf_names = [conf["timeframe"] for conf in confirmations]
            explanation.append(f"Confirmed by: {', '.join(tf_names)}")
            
        # Add conflicts
        conflicts = signals.get("conflicts_with", [])
        if conflicts:
            tf_names = [conf["timeframe"] for conf in conflicts]
            explanation.append(f"Conflicts with: {', '.join(tf_names)}")
            
        # Add supporting indicators if available
        supporting = []
        supporting_indicators = []
        
        if signal_type == "Buy" and "buy_signals" in signals:
            for signal in signals["buy_signals"]:
                supporting_indicators.append(f"{signal['indicator']} ({signal['strength']:.1f})")
                
        elif signal_type == "Sell" and "sell_signals" in signals:
            for signal in signals["sell_signals"]:
                supporting_indicators.append(f"{signal['indicator']} ({signal['strength']:.1f})")
                
        if supporting_indicators:
            supporting.append(f"Supporting indicators: {', '.join(supporting_indicators)}")
            
        explanation.extend(supporting)
        
        return "\n".join(explanation)
        
def create_default_signal_generator() -> SignalGenerator:
    """
    Create a signal generator with default settings
    
    Returns:
        Configured SignalGenerator
    """
    generator = SignalGenerator()
    
    # Additional customizations could be added here
    
    return generator

def is_market_hours(timestamp):
    """
    Check if the given timestamp is during market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
    
    Args:
        timestamp: Datetime object or index
        
    Returns:
        bool: True if timestamp is during market hours, False otherwise
    """
    # If timestamp has no tzinfo, assume it's UTC and convert to Eastern
    if hasattr(timestamp, 'tzinfo'):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=pytz.UTC)
        # Convert to Eastern Time
        eastern = pytz.timezone('US/Eastern')
        timestamp = timestamp.astimezone(eastern)
    else:
        # If not a datetime, just return True (can't determine)
        return True
    
    # Check if it's a weekday (0 = Monday, 4 = Friday)
    if timestamp.weekday() > 4:  # Saturday or Sunday
        return False
    
    # Check if during market hours (9:30 AM - 4:00 PM ET)
    market_open = time(9, 30)
    market_close = time(16, 0)
    current_time = timestamp.time()
    
    return market_open <= current_time <= market_close

def generate_signals(data):
    """
    Generate trading signals based on various indicators and patterns
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC and volume data
        
    Returns:
        pd.DataFrame: DataFrame with buy/sell signals and strength
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain these columns: {required_cols}")
    
    try:
        # Initialize signals DataFrame - first create a complete frame with default values
        signals = pd.DataFrame(index=data.index)
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['signal_strength'] = 0
        signals['target_price'] = np.nan
        signals['stop_loss'] = np.nan
        signals['signal_price'] = np.nan  # Add price at which signal was generated
        
        # Convert timestamps to Eastern Time
        eastern = pytz.timezone('US/Eastern')
        signals['signal_time_et'] = [idx.astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'astimezone') else idx 
                                     for idx in signals.index]
        
        # Get market hours mask - only calculate signals during market hours
        market_hours_mask = pd.Series([is_market_hours(idx) for idx in data.index], index=data.index)
        data_market_hours = data[market_hours_mask]
        
        # If no market hours data, return the empty signals DataFrame
        if len(data_market_hours) == 0:
            print("No data points during market hours, skipping signal calculation")
            return signals
            
        # Calculate volume ratio - vectorized approach
        signals['volume_ratio'] = 1.0  # Initialize with default value
        try:
            # Ensure volume is positive and non-zero
            volume_sma = data['volume'].replace(0, np.nan).rolling(window=20, min_periods=1).mean()
            # Calculate ratio only for non-zero SMA using vectorized operations
            mask = volume_sma > 0
            if mask.any():
                signals.loc[mask, 'volume_ratio'] = data.loc[mask, 'volume'] / volume_sma[mask]
        except Exception as e:
            print(f"Error calculating volume ratio: {str(e)}")
        
        # Initialize strategy signal columns
        signals['ema_vwap_bullish'] = False
        signals['ema_vwap_bearish'] = False
        signals['mm_vol_bullish'] = False
        signals['mm_vol_bearish'] = False
        
        # Calculate indicators for all data points
        ema20 = calculate_ema(data, 20)
        ema50 = calculate_ema(data, 50)
        ema200 = calculate_ema(data, 200)
        fast_ema, slow_ema = calculate_ema_cloud(data, 20, 50)
        macd_line, signal_line, macd_hist = calculate_macd(data)
        vwap = calculate_vwap(data)
        obv = calculate_obv(data)
        ad_line = calculate_ad_line(data)
        rsi = calculate_rsi(data)
        stoch_k, stoch_d = calculate_stochastic(data)
        sma5, sma8, sma13 = calculate_fibonacci_sma(data)
        intraday_momentum, late_selling, late_buying = calculate_pain(data)
        
        # Calculate new strategies
        ema_vwap_bullish, ema_vwap_bearish, ema_vwap_targets, ema_vwap_stops = calculate_ema_vwap_strategy(data)
        mm_vol_bullish, mm_vol_bearish, mm_vol_targets, mm_vol_stops = calculate_measured_move_volume_strategy(data)
        
        # Identify patterns
        doji = identify_doji(data)
        hammer = identify_hammer(data)
        bullish_engulfing, bearish_engulfing = identify_engulfing(data)
        morning_star = identify_morning_star(data)
        evening_star = identify_evening_star(data)
        
        # Identify price action patterns
        head_and_shoulders = identify_head_and_shoulders(data)
        inverse_head_and_shoulders = identify_inverse_head_and_shoulders(data)
        double_top = identify_double_top(data)
        double_bottom = identify_double_bottom(data)
        triple_top = identify_triple_top(data)
        triple_bottom = identify_triple_bottom(data)
        bullish_rectangle, bearish_rectangle = identify_rectangle(data)
        ascending_channel, descending_channel = identify_channel(data)
        ascending_triangle, descending_triangle, symmetric_triangle = identify_triangle(data)
        bull_flag, bear_flag = identify_flag(data)
        
        # Compute signals using vectorized operations
        # Only apply to market hours data
        market_idx = data.index[market_hours_mask]
        
        # Set signal prices for all market hours points
        signals.loc[market_idx, 'signal_price'] = data.loc[market_idx, 'close']
        
        # Initialize new signal scoring columns
        signals['buy_score'] = 0.0
        signals['sell_score'] = 0.0
        
        # ---------- WEIGHTED SCORING SYSTEM IMPLEMENTATION ----------
        
        # Define weights for different indicators based on their predictive power
        # These weights should be calibrated based on backtesting results
        indicator_weights = {
            # Trend indicators
            'price_above_ema20': 0.15,
            'price_above_ema50': 0.20,
            'price_above_ema200': 0.25,
            'price_above_vwap': 0.15,
            'ema_cloud_bullish': 0.20,
            
            # Momentum indicators
            'macd_cross_above_signal': 0.25,
            'macd_cross_below_signal': 0.25,
            'macd_above_zero': 0.15,
            'macd_below_zero': 0.15,
            'rsi_oversold': 0.20,
            'rsi_overbought': 0.20,
            
            # Volume indicators
            'obv_increasing': 0.20,
            'obv_decreasing': 0.20,
            'ad_line_increasing': 0.15,
            'ad_line_decreasing': 0.15,
            'volume_ratio_high': 0.25,
            
            # Pattern indicators
            'bullish_engulfing': 0.30,
            'bearish_engulfing': 0.30,
            'hammer': 0.25,
            'doji': 0.15,
            'morning_star': 0.35,
            'evening_star': 0.35,
            
            # Strategy indicators
            'ema_vwap_bullish': 0.40,
            'ema_vwap_bearish': 0.40,
            'mm_vol_bullish': 0.35,
            'mm_vol_bearish': 0.35,
            
            # Price action patterns
            'inverse_head_shoulders': 0.30,
            'head_shoulders': 0.30,
            'double_bottom': 0.25,
            'double_top': 0.25,
            'bullish_rectangle': 0.20,
            'bearish_rectangle': 0.20,
            'bull_flag': 0.25,
            'bear_flag': 0.25
        }
        
        # Detect market conditions
        # Calculate volatility using ATR or recent price range
        if len(data) >= 14:
            # Calculate ATR
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            # Calculate volatility as percentage of price
            volatility = atr / data['close']
            high_volatility = volatility > volatility.rolling(window=30).mean() * 1.5
            
            # Determine trend strength using ADX or moving average slopes
            ma_short = data['close'].rolling(window=20).mean()
            ma_long = data['close'].rolling(window=50).mean()
            
            uptrend = ma_short > ma_long
            downtrend = ma_short < ma_long
            
            # Adjust weights based on market conditions
            for idx in market_idx:
                # Adjust for high volatility
                volatility_factor = 1.0
                if len(volatility) > 0 and idx in volatility.index and not pd.isna(volatility[idx]):
                    if high_volatility[idx]:
                        # In high volatility, increase weight of trend indicators, decrease oscillators
                        volatility_factor = 1.5
                    else:
                        # In low volatility, do the opposite
                        volatility_factor = 0.8
                
                # Calculate RSI values
                rsi_value = rsi[idx] if idx in rsi.index else 50
                
                # Price above/below EMAs (vectorized)
                signals.loc[idx, 'price_above_ema20'] = data.loc[idx, 'close'] > ema20.loc[idx]
                signals.loc[idx, 'price_above_ema50'] = data.loc[idx, 'close'] > ema50.loc[idx]
                signals.loc[idx, 'price_above_ema200'] = data.loc[idx, 'close'] > ema200.loc[idx]
                signals.loc[idx, 'price_above_vwap'] = data.loc[idx, 'close'] > vwap.loc[idx]
                
                # EMA cloud signals
                signals.loc[idx, 'ema_cloud_bullish'] = fast_ema.loc[idx] > slow_ema.loc[idx]
                
                # RSI signals
                signals.loc[idx, 'rsi_oversold'] = rsi_value < 30
                signals.loc[idx, 'rsi_overbought'] = rsi_value > 70
                
                # MACD signals
                signals.loc[idx, 'macd_above_zero'] = macd_line.loc[idx] > 0
                signals.loc[idx, 'macd_below_zero'] = macd_line.loc[idx] < 0
                
                # Volume signals
                signals.loc[idx, 'volume_ratio_high'] = signals.loc[idx, 'volume_ratio'] > 1.2
        
        # Handle crossover signals (requires accessing previous values)
        for i, idx in enumerate(market_idx):
            if i > 0:
                prev_idx = market_idx[i-1]
                
                # Price crossovers
                signals.loc[idx, 'price_cross_above_ema20'] = (data.loc[idx, 'close'] > ema20.loc[idx]) & (data.loc[prev_idx, 'close'] <= ema20.loc[prev_idx])
                signals.loc[idx, 'price_cross_below_ema20'] = (data.loc[idx, 'close'] < ema20.loc[idx]) & (data.loc[prev_idx, 'close'] >= ema20.loc[prev_idx])
                
                # EMA cloud crossovers
                signals.loc[idx, 'ema_cloud_cross_bullish'] = (fast_ema.loc[idx] > slow_ema.loc[idx]) & (fast_ema.loc[prev_idx] <= slow_ema.loc[prev_idx])
                signals.loc[idx, 'ema_cloud_cross_bearish'] = (fast_ema.loc[idx] < slow_ema.loc[idx]) & (fast_ema.loc[prev_idx] >= slow_ema.loc[prev_idx])
                
                # MACD crossovers
                signals.loc[idx, 'macd_cross_above_signal'] = (macd_line.loc[idx] > signal_line.loc[idx]) & (macd_line.loc[prev_idx] <= signal_line.loc[prev_idx])
                signals.loc[idx, 'macd_cross_below_signal'] = (macd_line.loc[idx] < signal_line.loc[idx]) & (macd_line.loc[prev_idx] >= signal_line.loc[prev_idx])
                
                # VWAP crossovers
                signals.loc[idx, 'price_cross_above_vwap'] = (data.loc[idx, 'close'] > vwap.loc[idx]) & (data.loc[prev_idx, 'close'] <= vwap.loc[prev_idx])
                signals.loc[idx, 'price_cross_below_vwap'] = (data.loc[idx, 'close'] < vwap.loc[idx]) & (data.loc[prev_idx, 'close'] >= vwap.loc[prev_idx])
                
                # OBV and A/D line changes
                signals.loc[idx, 'obv_increasing'] = obv.loc[idx] > obv.loc[prev_idx]
                signals.loc[idx, 'obv_decreasing'] = obv.loc[idx] < obv.loc[prev_idx]
                signals.loc[idx, 'ad_line_increasing'] = ad_line.loc[idx] > ad_line.loc[prev_idx]
                signals.loc[idx, 'ad_line_decreasing'] = ad_line.loc[idx] < ad_line.loc[prev_idx]
                
                # Divergence patterns (3-day lookback)
                if i >= 3:
                    idx_minus_3 = market_idx[i-3]
                    signals.loc[idx, 'bullish_price_obv_divergence'] = (data.loc[idx, 'close'] < data.loc[idx_minus_3, 'close']) & (obv.loc[idx] > obv.loc[idx_minus_3])
                    signals.loc[idx, 'bearish_price_obv_divergence'] = (data.loc[idx, 'close'] > data.loc[idx_minus_3, 'close']) & (obv.loc[idx] < obv.loc[idx_minus_3])
        
        # Copy pattern signals
        for idx in market_idx:
            if idx in bullish_engulfing.index:
                signals.loc[idx, 'bullish_engulfing'] = bullish_engulfing.loc[idx]
            if idx in bearish_engulfing.index:
                signals.loc[idx, 'bearish_engulfing'] = bearish_engulfing.loc[idx]
            if idx in hammer.index:
                signals.loc[idx, 'hammer'] = hammer.loc[idx]
            if idx in doji.index:
                signals.loc[idx, 'doji'] = doji.loc[idx]
            if idx in morning_star.index:
                signals.loc[idx, 'morning_star'] = morning_star.loc[idx]
            if idx in evening_star.index:
                signals.loc[idx, 'evening_star'] = evening_star.loc[idx]
            
            # Price action patterns
            if idx in inverse_head_and_shoulders.index:
                signals.loc[idx, 'inverse_head_shoulders'] = inverse_head_and_shoulders.loc[idx]
            if idx in head_and_shoulders.index:
                signals.loc[idx, 'head_shoulders'] = head_and_shoulders.loc[idx]
            if idx in double_bottom.index:
                signals.loc[idx, 'double_bottom'] = double_bottom.loc[idx]
            if idx in double_top.index:
                signals.loc[idx, 'double_top'] = double_top.loc[idx]
            if idx in bullish_rectangle.index:
                signals.loc[idx, 'bullish_rectangle'] = bullish_rectangle.loc[idx]
            if idx in bearish_rectangle.index:
                signals.loc[idx, 'bearish_rectangle'] = bearish_rectangle.loc[idx]
            if idx in bull_flag.index:
                signals.loc[idx, 'bull_flag'] = bull_flag.loc[idx]
            if idx in bear_flag.index:
                signals.loc[idx, 'bear_flag'] = bear_flag.loc[idx]
        
        # Copy strategy signals (vectorized)
        common_idx = market_idx.intersection(ema_vwap_bullish.index)
        if not common_idx.empty:
            signals.loc[common_idx, 'ema_vwap_bullish'] = ema_vwap_bullish.loc[common_idx]
            
        common_idx = market_idx.intersection(ema_vwap_bearish.index)
        if not common_idx.empty:
            signals.loc[common_idx, 'ema_vwap_bearish'] = ema_vwap_bearish.loc[common_idx]
            
        common_idx = market_idx.intersection(mm_vol_bullish.index)
        if not common_idx.empty:
            signals.loc[common_idx, 'mm_vol_bullish'] = mm_vol_bullish.loc[common_idx]
            
        common_idx = market_idx.intersection(mm_vol_bearish.index)
        if not common_idx.empty:
            signals.loc[common_idx, 'mm_vol_bearish'] = mm_vol_bearish.loc[common_idx]
        
        # Calculate scores for buy signals
        for idx in market_idx:
            buy_score = 0.0
            sell_score = 0.0
            
            # Dynamically adjust weights based on market conditions
            trend_multiplier = 1.0
            reversal_multiplier = 1.0
            volume_multiplier = 1.0
            
            # Check if we have market condition data
            if 'uptrend' in locals() and idx in uptrend.index:
                # In strong uptrend, value trend continuations more
                if uptrend[idx]:
                    trend_multiplier = 1.2
                    reversal_multiplier = 0.8
                # In strong downtrend, value trend continuations more
                elif downtrend[idx]:
                    trend_multiplier = 1.2
                    reversal_multiplier = 0.8
                    
            # In high volume periods, value volume indicators more
            if signals.loc[idx, 'volume_ratio'] > 1.5:
                volume_multiplier = 1.3
                
            # Calculate buy score components
            for indicator, weight in indicator_weights.items():
                if indicator in signals.columns and signals.loc[idx, indicator] == True:
                    # Apply appropriate multiplier based on indicator type
                    adjusted_weight = weight
                    
                    # Trend indicators
                    if indicator in ['price_above_ema20', 'price_above_ema50', 'price_above_ema200', 
                                     'price_above_vwap', 'ema_cloud_bullish']:
                        adjusted_weight *= trend_multiplier
                        
                    # Reversal indicators
                    elif indicator in ['hammer', 'bullish_engulfing', 'morning_star', 'inverse_head_shoulders', 'double_bottom']:
                        adjusted_weight *= reversal_multiplier
                        
                    # Volume indicators
                    elif indicator in ['obv_increasing', 'ad_line_increasing', 'volume_ratio_high']:
                        adjusted_weight *= volume_multiplier
                        
                    # Add to buy score if it's a bullish indicator
                    if indicator in ['price_above_ema20', 'price_above_ema50', 'price_above_ema200', 
                                    'price_above_vwap', 'ema_cloud_bullish', 'macd_cross_above_signal',
                                    'macd_above_zero', 'rsi_oversold', 'obv_increasing', 'ad_line_increasing',
                                    'bullish_engulfing', 'hammer', 'morning_star', 'ema_vwap_bullish',
                                    'mm_vol_bullish', 'inverse_head_shoulders', 'double_bottom',
                                    'bullish_rectangle', 'bull_flag', 'bullish_price_obv_divergence',
                                    'price_cross_above_ema20', 'ema_cloud_cross_bullish', 'price_cross_above_vwap']:
                        buy_score += adjusted_weight
                    
                    # Add to sell score if it's a bearish indicator
                    elif indicator in ['macd_cross_below_signal', 'macd_below_zero', 'rsi_overbought',
                                      'obv_decreasing', 'ad_line_decreasing', 'bearish_engulfing',
                                      'evening_star', 'ema_vwap_bearish', 'mm_vol_bearish',
                                      'head_shoulders', 'double_top', 'bearish_rectangle', 'bear_flag',
                                      'bearish_price_obv_divergence', 'price_cross_below_ema20',
                                      'ema_cloud_cross_bearish', 'price_cross_below_vwap']:
                        sell_score += adjusted_weight
            
            # Store calculated scores
            signals.loc[idx, 'buy_score'] = min(1.0, buy_score)  # Cap at 1.0
            signals.loc[idx, 'sell_score'] = min(1.0, sell_score)  # Cap at 1.0
        
        # Apply thresholds to determine signals
        # Use dynamic thresholds based on market conditions
        base_buy_threshold = 0.6
        base_sell_threshold = 0.6
        
        # Adjust thresholds if we have volatility data
        if 'volatility' in locals():
            # Use last volatility value or median if not available
            recent_volatility = volatility.iloc[-1] if len(volatility) > 0 else np.nan
            median_volatility = volatility.median() if len(volatility) > 0 else np.nan
            
            # If recent volatility is high, require stronger signals
            if not pd.isna(recent_volatility) and not pd.isna(median_volatility):
                volatility_ratio = recent_volatility / median_volatility
                
                # Adjust thresholds - higher volatility requires stronger signals
                if volatility_ratio > 1.5:
                    base_buy_threshold = 0.75
                    base_sell_threshold = 0.75
                elif volatility_ratio < 0.7:
                    base_buy_threshold = 0.5
                    base_sell_threshold = 0.5
        
        # Generate buy/sell signals using calculated scores
        signals['buy_signal'] = signals['buy_score'] >= base_buy_threshold
        signals['sell_signal'] = signals['sell_score'] >= base_sell_threshold
        
        # Set buy_count and sell_count for backward compatibility
        signals['buy_count'] = (signals['buy_score'] * 10).astype(int)
        signals['sell_count'] = (signals['sell_score'] * 10).astype(int)
        
        # Set targets and stop losses - vectorized where possible
        buy_idx = signals.index[signals['buy_signal']].intersection(market_idx)
        sell_idx = signals.index[signals['sell_signal']].intersection(market_idx)
        
        # Default targets and stops
        if not buy_idx.empty:
            signals.loc[buy_idx, 'target_price'] = data.loc[buy_idx, 'close'] * 1.05
            signals.loc[buy_idx, 'stop_loss'] = data.loc[buy_idx, 'close'] * 0.97
        
        if not sell_idx.empty:
            signals.loc[sell_idx, 'target_price'] = data.loc[sell_idx, 'close'] * 0.95
            signals.loc[sell_idx, 'stop_loss'] = data.loc[sell_idx, 'close'] * 1.03
        
        # Override with strategy-specific targets if available
        # EMA/VWAP strategy buy signals
        if not buy_idx.empty:
            ema_vwap_buy_idx = buy_idx[signals.loc[buy_idx, 'ema_vwap_bullish']]
            if not ema_vwap_buy_idx.empty:
                common_idx = ema_vwap_buy_idx.intersection(ema_vwap_targets.index)
                if not common_idx.empty:
                    valid_idx = common_idx[~pd.isna(ema_vwap_targets.loc[common_idx])]
                    if not valid_idx.empty:
                        signals.loc[valid_idx, 'target_price'] = ema_vwap_targets.loc[valid_idx]
                        signals.loc[valid_idx, 'stop_loss'] = ema_vwap_stops.loc[valid_idx]
            
            # MM/Vol strategy buy signals
            mm_vol_buy_idx = buy_idx[signals.loc[buy_idx, 'mm_vol_bullish']]
            if not mm_vol_buy_idx.empty:
                common_idx = mm_vol_buy_idx.intersection(mm_vol_targets.index)
                if not common_idx.empty:
                    valid_idx = common_idx[~pd.isna(mm_vol_targets.loc[common_idx])]
                    if not valid_idx.empty:
                        signals.loc[valid_idx, 'target_price'] = mm_vol_targets.loc[valid_idx]
                        signals.loc[valid_idx, 'stop_loss'] = mm_vol_stops.loc[valid_idx]
        
        if not sell_idx.empty:
            # EMA/VWAP strategy sell signals
            ema_vwap_sell_idx = sell_idx[signals.loc[sell_idx, 'ema_vwap_bearish']]
            if not ema_vwap_sell_idx.empty:
                common_idx = ema_vwap_sell_idx.intersection(ema_vwap_targets.index)
                if not common_idx.empty:
                    valid_idx = common_idx[~pd.isna(ema_vwap_targets.loc[common_idx])]
                    if not valid_idx.empty:
                        signals.loc[valid_idx, 'target_price'] = ema_vwap_targets.loc[valid_idx]
                        signals.loc[valid_idx, 'stop_loss'] = ema_vwap_stops.loc[valid_idx]
            
            # MM/Vol strategy sell signals
            mm_vol_sell_idx = sell_idx[signals.loc[sell_idx, 'mm_vol_bearish']]
            if not mm_vol_sell_idx.empty:
                common_idx = mm_vol_sell_idx.intersection(mm_vol_targets.index)
                if not common_idx.empty:
                    valid_idx = common_idx[~pd.isna(mm_vol_targets.loc[common_idx])]
                    if not valid_idx.empty:
                        signals.loc[valid_idx, 'target_price'] = mm_vol_targets.loc[valid_idx]
                        signals.loc[valid_idx, 'stop_loss'] = mm_vol_stops.loc[valid_idx]
        
        # Map signal strength using buy_score and sell_score
        signals['signal_strength'] = 0
        
        # For buy signals
        buy_strength_conditions = [
            signals['buy_score'] >= 0.85,
            signals['buy_score'] >= 0.70,
            signals['buy_score'] >= 0.60
        ]
        
        buy_strength_choices = [
            SignalStrength.VERY_STRONG.value,
            SignalStrength.STRONG.value,
            SignalStrength.MODERATE.value
        ]
        
        # Default to WEAK if below threshold
        buy_default = SignalStrength.WEAK.value
        
        # Apply strength mapping for buy signals
        mask = signals['buy_signal']
        if mask.any():
            signals.loc[mask, 'signal_strength'] = np.select(
                buy_strength_conditions,
                buy_strength_choices,
                default=buy_default
            )[mask]
        
        # For sell signals
        sell_strength_conditions = [
            signals['sell_score'] >= 0.85,
            signals['sell_score'] >= 0.70,
            signals['sell_score'] >= 0.60
        ]
        
        sell_strength_choices = [
            SignalStrength.VERY_STRONG.value,
            SignalStrength.STRONG.value,
            SignalStrength.MODERATE.value
        ]
        
        # Default to WEAK if below threshold
        sell_default = SignalStrength.WEAK.value
        
        # Apply strength mapping for sell signals
        mask = signals['sell_signal']
        if mask.any():
            signals.loc[mask, 'signal_strength'] = np.select(
                sell_strength_conditions,
                sell_strength_choices,
                default=sell_default
            )[mask]
        
        # Set strong signals flag using vectorized operations
        signals['strong_buy'] = signals['buy_signal'] & (signals['signal_strength'] >= 3)
        signals['strong_sell'] = signals['sell_signal'] & (signals['signal_strength'] >= 3)
            
        return signals
    except Exception as e:
        # If the signal generation fails, return a minimal valid signals DataFrame
        print(f"Error in signal generation: {str(e)}")
        minimal_signals = pd.DataFrame(index=data.index)
        minimal_signals['buy_signal'] = False
        minimal_signals['sell_signal'] = False
        minimal_signals['signal_strength'] = 0
        minimal_signals['target_price'] = np.nan
        minimal_signals['stop_loss'] = np.nan
        minimal_signals['signal_price'] = np.nan
        minimal_signals['signal_time_et'] = ''
        return minimal_signals

def generate_signals_advanced(data_dict: Dict[str, pd.DataFrame], primary_tf: str = None) -> Dict[str, Any]:
    """
    Generate trading signals using the advanced signal processing system
    
    Args:
        data_dict: Dictionary mapping timeframe names to price DataFrames
        primary_tf: Primary timeframe to use (defaults to shortest available)
        
    Returns:
        Dictionary containing signals and analytical metrics
    """
    if not data_dict or not isinstance(data_dict, dict):
        return {}
        
    try:
        # Create advanced signal processor
        processor = AdvancedSignalProcessor()
        
        # Calculate multi-timeframe signal scores
        result = processor.calculate_multi_timeframe_score(data_dict, primary_tf)
        
        # Extract the primary timeframe data
        if primary_tf is None or primary_tf not in data_dict:
            primary_tf = list(data_dict.keys())[0] if data_dict else None
            
        if not primary_tf:
            return {}
            
        primary_data = data_dict[primary_tf]
        
        # Set up basic signal structure
        signals = pd.DataFrame(index=primary_data.index)
        
        # Add signal price (close price at signal time)
        signals['signal_price'] = primary_data['close']
        
        # Convert timestamps to Eastern Time
        eastern = pytz.timezone('US/Eastern')
        signals['signal_time_et'] = [idx.astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'astimezone') else str(idx) 
                                     for idx in signals.index]
        
        # Apply weighted scoring system
        # Calculate indicators for primary timeframe first
        try:
            # Calculate volatility using ATR
            high_low = primary_data['high'] - primary_data['low']
            high_close = abs(primary_data['high'] - primary_data['close'].shift(1))
            low_close = abs(primary_data['low'] - primary_data['close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            # Calculate volatility as percentage of price
            volatility = atr / primary_data['close']
            high_volatility = volatility > volatility.rolling(window=30).mean() * 1.5
            
            # Determine trend strength using moving average slopes
            ma_short = primary_data['close'].rolling(window=20).mean()
            ma_long = primary_data['close'].rolling(window=50).mean()
            
            uptrend = ma_short > ma_long
            downtrend = ma_short < ma_long
            
            # Adjust base thresholds based on market conditions
            base_buy_threshold = 0.6
            base_sell_threshold = 0.6
            
            # Use last volatility value or median if not available
            recent_volatility = volatility.iloc[-1] if len(volatility) > 0 else np.nan
            median_volatility = volatility.median() if len(volatility) > 0 else np.nan
            
            # If recent volatility is high, require stronger signals
            if not pd.isna(recent_volatility) and not pd.isna(median_volatility):
                volatility_ratio = recent_volatility / median_volatility
                
                # Adjust thresholds - higher volatility requires stronger signals
                if volatility_ratio > 1.5:
                    base_buy_threshold = 0.75
                    base_sell_threshold = 0.75
                elif volatility_ratio < 0.7:
                    base_buy_threshold = 0.5
                    base_sell_threshold = 0.5
                    
            # Store market regime information
            for idx in signals.index:
                if idx in uptrend.index and not pd.isna(uptrend[idx]):
                    if uptrend[idx]:
                        if idx in high_volatility.index and high_volatility[idx]:
                            result['market_regime'] = MarketRegime.HIGH_VOLATILITY
                        else:
                            result['market_regime'] = MarketRegime.BULL_TREND
                    elif downtrend[idx]:
                        if idx in high_volatility.index and high_volatility[idx]:
                            result['market_regime'] = MarketRegime.HIGH_VOLATILITY
                        else:
                            result['market_regime'] = MarketRegime.BEAR_TREND
                    else:
                        result['market_regime'] = MarketRegime.SIDEWAYS
        except Exception as e:
            # If volatility/trend calculation fails, use default regime
            result['market_regime'] = MarketRegime.SIDEWAYS
            print(f"Error calculating market conditions: {str(e)}")
        
        if 'signal_score' in result:
            signals['score'] = result['signal_score']
            
            # Generate buy/sell signals based on dynamic thresholds
            signals['buy_signal'] = signals['score'] >= base_buy_threshold
            signals['sell_signal'] = signals['score'] <= (1 - base_sell_threshold)
            
            # Set signal strength based on score using improved mapping
            signals['signal_strength'] = None
            
            # For buy signals
            buy_strength_conditions = [
                signals['score'] >= 0.85,
                signals['score'] >= 0.70,
                signals['score'] >= 0.60
            ]
            
            buy_strength_choices = [
                SignalStrength.VERY_STRONG.value,
                SignalStrength.STRONG.value,
                SignalStrength.MODERATE.value
            ]
            
            # Default to WEAK if below threshold
            buy_default = SignalStrength.WEAK.value
            
            # Apply strength mapping for buy signals
            mask = signals['buy_signal']
            if mask.any():
                signals.loc[mask, 'signal_strength'] = np.select(
                    buy_strength_conditions,
                    buy_strength_choices,
                    default=buy_default
                )[mask]
            
            # For sell signals - invert the score for sell signals (1.0 is strongest sell)
            sell_score = 1 - signals['score']
            sell_strength_conditions = [
                sell_score >= 0.85,
                sell_score >= 0.70,
                sell_score >= 0.60
            ]
            
            sell_strength_choices = [
                SignalStrength.VERY_STRONG.value,
                SignalStrength.STRONG.value,
                SignalStrength.MODERATE.value
            ]
            
            # Default to WEAK if below threshold
            sell_default = SignalStrength.WEAK.value
            
            # Apply strength mapping for sell signals
            mask = signals['sell_signal']
            if mask.any():
                signals.loc[mask, 'signal_strength'] = np.select(
                    sell_strength_conditions,
                    sell_strength_choices,
                    default=sell_default
                )[mask]
            
            # Add market regime
            if 'market_regime' in result:
                signals['market_regime'] = result['market_regime'].value
                
            # Add confidence
            if 'confidence' in result:
                signals['confidence'] = result['confidence']
                
            # Add component scores
            for component in ['trend_score', 'momentum_score', 'volume_score', 'volatility_score']:
                if component in result:
                    signals[component] = result[component]
                    
            # Add multi-timeframe metrics
            if 'consensus_score' in result:
                signals['consensus_score'] = result['consensus_score']
                
            if 'aligned' in result:
                signals['timeframes_aligned'] = result['aligned']
                
            # Generate target prices and stop losses based on ATR if available
            if 'atr' in primary_data.columns or 'atr' in locals():
                # Use ATR for position sizing with dynamic multipliers based on signal strength
                
                # Default multipliers
                target_multiplier_base = 2.0
                stop_multiplier_base = 1.0
                
                # Initialize target and stop loss columns
                signals['target_price'] = None
                signals['stop_loss'] = None
                
                # Use ATR if already in dataframe, otherwise use calculated version
                if 'atr' not in primary_data.columns and 'atr' in locals():
                    primary_data['atr'] = atr
                
                # Process buy signals
                buy_idx = signals.index[signals['buy_signal']]
                if not buy_idx.empty:
                    for idx in buy_idx:
                        if idx not in primary_data.index:
                            continue
                            
                        # Adjust multipliers based on signal strength
                        strength = signals.loc[idx, 'signal_strength'] if pd.notnull(signals.loc[idx, 'signal_strength']) else 1
                        
                        # Stronger signals get more aggressive targets
                        target_multiplier = target_multiplier_base * (1 + (strength - 1) * 0.2)
                        stop_multiplier = stop_multiplier_base * (1 - (strength - 1) * 0.1)
                        
                        if 'atr' in primary_data.columns and pd.notnull(primary_data.loc[idx, 'atr']):
                            signals.loc[idx, 'target_price'] = primary_data.loc[idx, 'close'] + (primary_data.loc[idx, 'atr'] * target_multiplier)
                            signals.loc[idx, 'stop_loss'] = primary_data.loc[idx, 'close'] - (primary_data.loc[idx, 'atr'] * stop_multiplier)
                        else:
                            # Fallback to percentage-based targets
                            signals.loc[idx, 'target_price'] = primary_data.loc[idx, 'close'] * (1 + 0.01 * target_multiplier)
                            signals.loc[idx, 'stop_loss'] = primary_data.loc[idx, 'close'] * (1 - 0.005 * stop_multiplier)
                
                # Process sell signals
                sell_idx = signals.index[signals['sell_signal']]
                if not sell_idx.empty:
                    for idx in sell_idx:
                        if idx not in primary_data.index:
                            continue
                            
                        # Adjust multipliers based on signal strength
                        strength = signals.loc[idx, 'signal_strength'] if pd.notnull(signals.loc[idx, 'signal_strength']) else 1
                        
                        # Stronger signals get more aggressive targets
                        target_multiplier = target_multiplier_base * (1 + (strength - 1) * 0.2)
                        stop_multiplier = stop_multiplier_base * (1 - (strength - 1) * 0.1)
                        
                        if 'atr' in primary_data.columns and pd.notnull(primary_data.loc[idx, 'atr']):
                            signals.loc[idx, 'target_price'] = primary_data.loc[idx, 'close'] - (primary_data.loc[idx, 'atr'] * target_multiplier)
                            signals.loc[idx, 'stop_loss'] = primary_data.loc[idx, 'close'] + (primary_data.loc[idx, 'atr'] * stop_multiplier)
                        else:
                            # Fallback to percentage-based targets
                            signals.loc[idx, 'target_price'] = primary_data.loc[idx, 'close'] * (1 - 0.01 * target_multiplier)
                            signals.loc[idx, 'stop_loss'] = primary_data.loc[idx, 'close'] * (1 + 0.005 * stop_multiplier)
                            
            # Set strong signals flag for convenience
            signals['strong_buy'] = signals['buy_signal'] & (signals['signal_strength'] >= 3)
            signals['strong_sell'] = signals['sell_signal'] & (signals['signal_strength'] >= 3)
            
        return {
            'signals': signals,
            'metrics': result,
            'primary_timeframe': primary_tf
        }
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error generating advanced signals: {str(e)}")
        # Return minimal valid DataFrame to avoid breaking visualizations
        minimal_signals = pd.DataFrame(index=list(data_dict.values())[0].index if data_dict else [])
        minimal_signals['signal_price'] = np.nan
        minimal_signals['signal_time_et'] = ''
        return {
            'signals': minimal_signals,
            'metrics': {},
            'primary_timeframe': primary_tf
        } 