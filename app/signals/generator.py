import pandas as pd
import numpy as np
from enum import Enum
import pytz
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import traceback

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
    multi_indicator_confirmation,
    calculate_roc, 
    calculate_hull_moving_average, 
    calculate_ttm_squeeze,
    detect_hidden_divergence,
    calculate_opening_range,
    detect_orb_breakout,
    analyze_session_data,
    calculate_adaptive_rsi,
    calculate_atr,
    calculate_adx
)
from app.patterns import (
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

# Import signals and market regime
from app.signals.scoring import (
    DynamicSignalScorer, MarketRegime, SignalStrength
)

from app.signals.processing import (
    AdvancedSignalProcessor, calculate_advanced_signal_score,
    calculate_multi_timeframe_signal_score
)

# Import SPY strategy module
try:
    from app.signals.spy_strategy import analyze_spy_day_trading
except ImportError:
    analyze_spy_day_trading = None
    print("SPY strategy module not available")

# Import the new multi-timeframe framework
from app.signals.multi_timeframe import (
    MultiTimeframeFramework, TimeframeTier, TrendDirection,
    multi_tf_confirmation
)

# Logging setup
logger = logging.getLogger(__name__)

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
        
        # Detect market regime for the signal scorer before aggregating signals
        market_regime = self.signal_scorer.detect_market_regime(data)
        
        # Define regime-based weights
        regime_weights = {
            MarketRegime.BULL_TREND: 1.3,
            MarketRegime.BEAR_TREND: 1.2,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.7
        }
        
        # Define a weight adjustment function based on current market regime
        def weight_adjustment(weight):
            return weight * regime_weights.get(market_regime, 1.0)
        
        # Define time decay function (0.95 per hour)
        def apply_time_decay(signals_data):
            if not isinstance(signals_data, dict) or not isinstance(data.index, pd.DatetimeIndex):
                return signals_data
                
            try:
                # Calculate time decay based on most recent timestamp
                latest_time = data.index[-1]
                time_delta = (latest_time - data.index).total_seconds() / 3600
                
                # Apply time decay to signals
                # This is a simplified approach - in a real implementation, you'd apply this
                # to actual signal scores within the aggregation logic
                if 'buy_score' in signals_data:
                    signals_data['buy_score'] = signals_data.get('buy_score', 0) * (0.95 ** max(0, time_delta[-1]))
                if 'sell_score' in signals_data:
                    signals_data['sell_score'] = signals_data.get('sell_score', 0) * (0.95 ** max(0, time_delta[-1]))
            except Exception:
                # If time decay calculation fails, just return original signals
                pass
                
            return signals_data
        
        # Aggregate signals with weight adjustment for market regime
        aggregated_signals = self.indicator_layers.aggregate_signals(
            indicator_results,
            weight_adjustment=weight_adjustment
        )
        
        # Apply time decay to aggregated signals
        aggregated_signals = apply_time_decay(aggregated_signals)
        
        # Add market regime information to signals
        aggregated_signals['market_regime'] = market_regime
        
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
            
        # Detect market regime using primary timeframe data
        market_regime = None
        if primary_tf in self.timeframe_manager.timeframes:
            primary_data = self.timeframe_manager.timeframes[primary_tf].data
            if primary_data is not None and len(primary_data) > 30:
                try:
                    # Use advanced signal processor to detect market regime
                    processor = AdvancedSignalProcessor()
                    market_regime = processor.detect_market_regime(primary_data)
                    self.logger.info(f"Detected market regime: {market_regime}")
                except Exception as e:
                    self.logger.error(f"Error detecting market regime: {str(e)}")
        
        # Process multi-timeframe validation using enhanced method if available
        try:
            # Use the enhanced multi-timeframe analysis with market regime information
            multi_tf_signals = self.timeframe_manager.enhanced_multi_timeframe_analysis(market_regime)
            self.logger.info(f"Generated enhanced multi-timeframe signals with regime adjustment")
        except (AttributeError, Exception) as e:
            # Fall back to basic multi-timeframe analysis if enhanced method isn't available
            self.logger.warning(f"Enhanced multi-timeframe analysis failed, using basic method: {str(e)}")
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
            
            # Add market regime information if available
            if market_regime:
                multi_tf_signals['market_regime'] = market_regime.value
                multi_tf_signals['market_regime_name'] = market_regime.name
        
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
    Generate trading signals based on various indicators
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: DataFrame with signal columns
    """
    # Check if current time is during market hours
    current_time = datetime.now(pytz.timezone('US/Eastern'))
    market_open = is_market_hours(current_time)
    
    if not market_open:
        print("Market is closed. Skipping signal calculation.")
        # Initialize signals DataFrame with necessary columns but no actual signals
        signals = pd.DataFrame(index=data.index)
        signals["buy_signal"] = False
        signals["sell_signal"] = False
        signals["buy_score"] = 0.0
        signals["sell_score"] = 0.0
        signals["signal_strength"] = 0
        signals["strong_buy_signal"] = False
        signals["strong_sell_signal"] = False
        signals["signal_price"] = data["close"]
        signals["target_price"] = None
        signals["stop_loss"] = None
        signals["market_status"] = "Closed"
        
        # Convert timestamps to Eastern Time for display
        eastern = pytz.timezone('US/Eastern')
        signals["signal_time_et"] = [idx.astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'astimezone') else str(idx) 
                                 for idx in signals.index]
        
        return signals
    
    print(f"Starting signal generation with {len(data)} data points")
    
    try:
        # Check if input data is a valid DataFrame
        if not isinstance(data, pd.DataFrame) or len(data) < 10:
            raise ValueError(f"Input data must be a DataFrame with at least 10 points, got {len(data) if isinstance(data, pd.DataFrame) else type(data)}")
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Input data is missing required columns: {', '.join(missing_columns)}")
        
        # Initialize signals DataFrame with necessary columns
        signals = pd.DataFrame(index=data.index)
        signals["buy_signal"] = False
        signals["sell_signal"] = False
        signals["buy_score"] = 0.0
        signals["sell_score"] = 0.0
        signals["signal_strength"] = 0
        signals["strong_buy_signal"] = False
        signals["strong_sell_signal"] = False
        signals["signal_price"] = data["close"]
        signals["target_price"] = None
        signals["stop_loss"] = None
        
        # Convert timestamps to Eastern Time for display
        eastern = pytz.timezone('US/Eastern')
        signals["signal_time_et"] = [idx.astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'astimezone') else str(idx) 
                                 for idx in signals.index]
        
        # === Calculate Opening Range Breakout (ORB) signals ===
        try:
            # Calculate opening range (first 5 minutes of the trading day)
            # Convert data index to Eastern Time for proper filtering
            market_data = data.copy()
            if market_data.index.tzinfo is not None:
                market_data.index = market_data.index.tz_convert(eastern)
            
            # Get the opening range if we have data from market open
            orb_high, orb_low = calculate_opening_range(market_data, minutes=5)
            
            if orb_high is not None and orb_low is not None:
                # Detect breakout signals
                orb_signals = detect_orb_breakout(market_data, orb_high, orb_low, buffer_percent=0.1)
                
                # Merge with signals DataFrame
                for idx in orb_signals.index:
                    if idx in signals.index:
                        if orb_signals.loc[idx, 'orb_long']:
                            signals.loc[idx, 'buy_signal'] = True
                            signals.loc[idx, 'buy_score'] += 0.8  # Strong signal
                            # Add ORB-specific columns
                            signals.loc[idx, 'orb_signal'] = True
                            signals.loc[idx, 'orb_level'] = orb_high
                            
                        if orb_signals.loc[idx, 'orb_short']:
                            signals.loc[idx, 'sell_signal'] = True
                            signals.loc[idx, 'sell_score'] += 0.8  # Strong signal
                            # Add ORB-specific columns
                            signals.loc[idx, 'orb_signal'] = True
                            signals.loc[idx, 'orb_level'] = orb_low
        except Exception as e:
            print(f"Error calculating ORB signals: {str(e)}")
        
        # === Calculate EMA Cloud crossover signals ===
        try:
            # Calculate EMA cloud (5, 13)
            fast_ema, slow_ema = calculate_ema_cloud(data, fast_period=5, slow_period=13)
            
            # Calculate crossover points
            signals['ema_cloud_cross_bullish'] = False
            signals['ema_cloud_cross_bearish'] = False
            
            # First point must have both EMAs available
            valid_indices = fast_ema.dropna().index.intersection(slow_ema.dropna().index)
            
            if len(valid_indices) >= 2:
                # Calculate for remaining points
                for i in range(1, len(valid_indices)):
                    current_idx = valid_indices[i]
                    prev_idx = valid_indices[i-1]
                    
                    # Bullish crossover: fast EMA crosses above slow EMA
                    if fast_ema[prev_idx] <= slow_ema[prev_idx] and fast_ema[current_idx] > slow_ema[current_idx]:
                        signals.loc[current_idx, 'ema_cloud_cross_bullish'] = True
                        signals.loc[current_idx, 'buy_signal'] = True
                        signals.loc[current_idx, 'buy_score'] += 0.7  # Strong signal
                    
                    # Bearish crossover: fast EMA crosses below slow EMA
                    if fast_ema[prev_idx] >= slow_ema[prev_idx] and fast_ema[current_idx] < slow_ema[current_idx]:
                        signals.loc[current_idx, 'ema_cloud_cross_bearish'] = True
                        signals.loc[current_idx, 'sell_signal'] = True
                        signals.loc[current_idx, 'sell_score'] += 0.7  # Strong signal
        except Exception as e:
            print(f"Error calculating EMA cloud signals: {str(e)}")
        
        # Calculate price trend signals
        try:
            # Use a short SMA to determine trend direction (8-period)
            sma8 = data['close'].rolling(window=8).mean()
            sma20 = data['close'].rolling(window=20).mean()
            
            # Uptrend: current close > SMAs
            uptrend = (data['close'] > sma8) & (sma8 > sma20)
            downtrend = (data['close'] < sma8) & (sma8 < sma20)
            
            # Add trend signals
            for idx in data.index:
                if idx in uptrend.index:
                    if uptrend[idx]:
                        signals.loc[idx, 'buy_score'] += 0.2
                    elif downtrend[idx]:
                        signals.loc[idx, 'sell_score'] += 0.2
        except Exception as e:
            print(f"Error calculating trend signals: {str(e)}")
        
        # Calculate advanced indicator signals
        try:
            # RSI signals
            rsi = calculate_rsi(data, 14)
            # Add adaptive RSI for more responsive signals in volatile markets
            adaptive_rsi = calculate_adaptive_rsi(data)
            
            for idx in rsi.index:
                if rsi[idx] < 30:  # Oversold
                    signals.loc[idx, 'buy_score'] += 0.3
                elif rsi[idx] > 70:  # Overbought
                    signals.loc[idx, 'sell_score'] += 0.3
                
                # If adaptive RSI data is available for this index point
                if idx in adaptive_rsi.index and not np.isnan(adaptive_rsi[idx]):
                    # Adaptive RSI provides stronger signals during volatile periods
                    if adaptive_rsi[idx] < 25:  # More aggressive oversold threshold
                        signals.loc[idx, 'buy_score'] += 0.4
                    elif adaptive_rsi[idx] > 75:  # More aggressive overbought threshold
                        signals.loc[idx, 'sell_score'] += 0.4
                    
                    # Check for divergence between regular and adaptive RSI
                    if idx > 0 and idx-1 in adaptive_rsi.index and idx-1 in rsi.index:
                        # Bullish divergence: adaptive RSI turns up while regular RSI still down
                        if adaptive_rsi[idx] > adaptive_rsi[idx-1] and rsi[idx] < rsi[idx-1]:
                            signals.loc[idx, 'buy_score'] += 0.2
                        # Bearish divergence: adaptive RSI turns down while regular RSI still up
                        elif adaptive_rsi[idx] < adaptive_rsi[idx-1] and rsi[idx] > rsi[idx-1]:
                            signals.loc[idx, 'sell_score'] += 0.2
                
            # Stochastic signals
            stoch_k, stoch_d = calculate_stochastic(data)
            for idx in stoch_k.index:
                if idx in stoch_d.index:
                    # Bullish stochastic crossover in oversold territory
                    if stoch_k[idx] < 20 and stoch_d[idx] < 20 and stoch_k[idx] > stoch_d[idx]:
                        signals.loc[idx, 'buy_score'] += 0.4
                    # Bearish stochastic crossover in overbought territory
                    elif stoch_k[idx] > 80 and stoch_d[idx] > 80 and stoch_k[idx] < stoch_d[idx]:
                        signals.loc[idx, 'sell_score'] += 0.4
                        
            # MACD signals
            macd, macd_signal, macd_hist = calculate_macd(data)
            for i in range(1, len(macd_hist)):
                idx = macd_hist.index[i]
                prev_idx = macd_hist.index[i-1]
                
                # Bullish MACD crossover
                if macd_hist[prev_idx] < 0 and macd_hist[idx] > 0:
                    signals.loc[idx, 'buy_score'] += 0.5
                    
                # Bearish MACD crossover
                elif macd_hist[prev_idx] > 0 and macd_hist[idx] < 0:
                    signals.loc[idx, 'sell_score'] += 0.5
        except Exception as e:
            print(f"Error calculating advanced indicator signals: {str(e)}")
        
        # === Finalize Signal Generation ===
        # Determine final signals based on scores
        signals['buy_signal'] = signals['buy_score'] >= 0.6
        signals['sell_signal'] = signals['sell_score'] >= 0.6
        
        # Track confirmation count for visualization
        signals["buy_count"] = 0
        signals["sell_count"] = 0
        signals["buy_confirm_count"] = 0
        signals["sell_confirm_count"] = 0
        
        # Count signals for each indicator type
        signal_counts = {
            'ema_cloud': 0,
            'trend': 0,
            'rsi': 0,
            'stochastic': 0,
            'macd': 0
        }
        
        # Set signal strength based on score
        for idx in signals.index:
            # Map signal strength based on confirmation count
            if signals.at[idx, 'buy_score'] >= 0.8:
                signals.at[idx, 'signal_strength'] = SignalStrength.VERY_STRONG.value
                signals.at[idx, 'buy_confirm_count'] = 3
            elif signals.at[idx, 'buy_score'] >= 0.6:
                signals.at[idx, 'signal_strength'] = SignalStrength.STRONG.value
                signals.at[idx, 'buy_confirm_count'] = 2
            elif signals.at[idx, 'buy_score'] >= 0.4:
                signals.at[idx, 'signal_strength'] = SignalStrength.MODERATE.value
                signals.at[idx, 'buy_confirm_count'] = 1
            
            # For sell signals
            if signals.at[idx, 'sell_score'] >= 0.8:
                signals.at[idx, 'signal_strength'] = SignalStrength.VERY_STRONG.value
                signals.at[idx, 'sell_confirm_count'] = 3
            elif signals.at[idx, 'sell_score'] >= 0.6:
                signals.at[idx, 'signal_strength'] = SignalStrength.STRONG.value
                signals.at[idx, 'sell_confirm_count'] = 2
            elif signals.at[idx, 'sell_score'] >= 0.4:
                signals.at[idx, 'signal_strength'] = SignalStrength.MODERATE.value
                signals.at[idx, 'sell_confirm_count'] = 1
            
            # Set strong signals flags
            signals.at[idx, 'strong_buy_signal'] = signals.at[idx, 'buy_signal'] and signals.at[idx, 'signal_strength'] >= SignalStrength.STRONG.value
            signals.at[idx, 'strong_sell_signal'] = signals.at[idx, 'sell_signal'] and signals.at[idx, 'signal_strength'] >= SignalStrength.STRONG.value
            
            # Set buy/sell counts for visualization
            signals.at[idx, 'buy_count'] = signals.at[idx, 'buy_confirm_count']
            signals.at[idx, 'sell_count'] = signals.at[idx, 'sell_confirm_count']
        
        # Calculate target prices and stop losses
        for idx in signals.index:
            if signals.loc[idx, 'buy_signal']:
                # Calculate ATR for dynamic target/stop
                atr_period = 14
                if idx in data.index and len(data.loc[:idx]) > atr_period:
                    high_low = data['high'] - data['low']
                    high_close = abs(data['high'] - data['close'].shift(1))
                    low_close = abs(data['low'] - data['close'].shift(1))
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = tr.rolling(window=atr_period).mean().loc[idx]
                    
                    # Set targets based on ATR and signal strength
                    if signals.at[idx, 'signal_strength'] == SignalStrength.VERY_STRONG.value:
                        signals.loc[idx, 'target_price'] = data.loc[idx, 'close'] + (atr * 3.0)
                        signals.loc[idx, 'stop_loss'] = data.loc[idx, 'close'] - (atr * 1.0)
                    elif signals.at[idx, 'signal_strength'] == SignalStrength.STRONG.value:
                        signals.loc[idx, 'target_price'] = data.loc[idx, 'close'] + (atr * 2.5)
                        signals.loc[idx, 'stop_loss'] = data.loc[idx, 'close'] - (atr * 1.0)
                    else:
                        signals.loc[idx, 'target_price'] = data.loc[idx, 'close'] + (atr * 2.0)
                        signals.loc[idx, 'stop_loss'] = data.loc[idx, 'close'] - (atr * 1.0)
                else:
                    # Fallback to percentage-based targets
                    signals.loc[idx, 'target_price'] = data.loc[idx, 'close'] * 1.02
                    signals.loc[idx, 'stop_loss'] = data.loc[idx, 'close'] * 0.98
                    
            elif signals.loc[idx, 'sell_signal']:
                # Calculate ATR for dynamic target/stop
                atr_period = 14
                if idx in data.index and len(data.loc[:idx]) > atr_period:
                    high_low = data['high'] - data['low']
                    high_close = abs(data['high'] - data['close'].shift(1))
                    low_close = abs(data['low'] - data['close'].shift(1))
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = tr.rolling(window=atr_period).mean().loc[idx]
                    
                    # Set targets based on ATR and signal strength
                    if signals.at[idx, 'signal_strength'] == SignalStrength.VERY_STRONG.value:
                        signals.loc[idx, 'target_price'] = data.loc[idx, 'close'] - (atr * 3.0)
                        signals.loc[idx, 'stop_loss'] = data.loc[idx, 'close'] + (atr * 1.0)
                    elif signals.at[idx, 'signal_strength'] == SignalStrength.STRONG.value:
                        signals.loc[idx, 'target_price'] = data.loc[idx, 'close'] - (atr * 2.5)
                        signals.loc[idx, 'stop_loss'] = data.loc[idx, 'close'] + (atr * 1.0)
                    else:
                        signals.loc[idx, 'target_price'] = data.loc[idx, 'close'] - (atr * 2.0)
                        signals.loc[idx, 'stop_loss'] = data.loc[idx, 'close'] + (atr * 1.0)
                else:
                    # Fallback to percentage-based targets
                    signals.loc[idx, 'target_price'] = data.loc[idx, 'close'] * 0.98
                    signals.loc[idx, 'stop_loss'] = data.loc[idx, 'close'] * 1.02
        
        print(f"Signal generation complete: {signals['buy_signal'].sum()} buy signals, {signals['sell_signal'].sum()} sell signals")
        return signals
        
    except Exception as e:
        print(f"Error in signal generation: {str(e)}")
        traceback.print_exc()
        
        # Return minimal valid signals DataFrame
        minimal_signals = pd.DataFrame(index=data.index)
        minimal_signals["signal_price"] = data["close"]
        minimal_signals["signal_time_et"] = ""
        minimal_signals["buy_signal"] = False
        minimal_signals["sell_signal"] = False
        minimal_signals["signal_strength"] = 0
        minimal_signals["buy_count"] = 0
        minimal_signals["sell_count"] = 0
        minimal_signals["buy_score"] = 0.0
        minimal_signals["sell_score"] = 0.0
        minimal_signals["buy_confirm_count"] = 0
        minimal_signals["sell_confirm_count"] = 0
        minimal_signals["target_price"] = None
        minimal_signals["stop_loss"] = None
        
        print(f"Signal generation complete: {minimal_signals['buy_signal'].sum()} buy signals, {minimal_signals['sell_signal'].sum()} sell signals")
        
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
        # Filter data to only include market hours for each timeframe
        market_hours_data = {}
        for tf, df in data_dict.items():
            # Check if data has enough points
            if df is None or len(df) < 20:
                continue
                
            # Filter to include only market hours
            market_idx = df.index[[is_market_hours(idx) for idx in df.index]]
            
            # Only include this timeframe if it has sufficient data points after filtering
            if len(market_idx) >= 20:
                market_hours_data[tf] = df.loc[market_idx]
            else:
                print(f"Insufficient market hours data for timeframe {tf} (only {len(market_idx)} points)")
        
        # Check if we have any timeframes with sufficient market hours data
        if not market_hours_data:
            print("Insufficient data for signal scoring - no timeframes have enough market hours data (min 20 points)")
            # Return empty results with message
            return {
                'signals': pd.DataFrame(),
                'metrics': {'error': 'Insufficient market hours data'},
                'primary_timeframe': primary_tf
            }
        
        # Update data_dict to use only filtered data
        data_dict = market_hours_data
        
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
            for component, score in result.get('component_scores', {}).items():
                signals[component] = score
        
        # Setup TimeFrameManager and process multi-timeframe data
        tf_manager = TimeFrameManager()
        
        # Add all timeframes to the manager
        for tf_name, tf_data in data_dict.items():
            # Determine priority based on timeframe name
            priority = TimeFramePriority.SECONDARY
            if tf_name == primary_tf:
                priority = TimeFramePriority.PRIMARY
            elif tf_name in ["1d", "1w"]:
                priority = TimeFramePriority.CONTEXT
            elif tf_name in ["1m", "5m"]:
                priority = TimeFramePriority.TERTIARY
                
            # Create TimeFrame object
            timeframe = TimeFrame(
                name=tf_name,
                interval=tf_name,
                priority=priority
            )
            
            # Set data and add to manager
            timeframe.set_data(tf_data)
            
            # Generate basic signals for this timeframe
            tf_signals = {}
            if 'score' in signals.columns:
                # Use already calculated score if available for primary timeframe
                if tf_name == primary_tf:
                    tf_signals = {
                        'buy_signal': signals['buy_signal'].iloc[-1],
                        'sell_signal': signals['sell_signal'].iloc[-1],
                        'buy_score': signals['score'].iloc[-1] if signals['buy_signal'].iloc[-1] else 0,
                        'sell_score': 1 - signals['score'].iloc[-1] if signals['sell_signal'].iloc[-1] else 0,
                        'signal_strength': signals['signal_strength'].iloc[-1] if 'signal_strength' in signals.columns else 1
                    }
                else:
                    # Calculate basic signals for other timeframes
                    tf_processor = AdvancedSignalProcessor()
                    tf_result = tf_processor.calculate_signal_score(tf_data)
                    tf_signals = {
                        'buy_signal': tf_result.get('signal_score', 0.5) >= base_buy_threshold,
                        'sell_signal': tf_result.get('signal_score', 0.5) <= (1 - base_sell_threshold),
                        'buy_score': tf_result.get('signal_score', 0.5) if tf_result.get('signal_score', 0.5) >= base_buy_threshold else 0,
                        'sell_score': 1 - tf_result.get('signal_score', 0.5) if tf_result.get('signal_score', 0.5) <= (1 - base_sell_threshold) else 0
                    }
            
            # Set signals on the timeframe
            timeframe.set_signals(tf_signals)
            
            # Add to manager
            tf_manager.add_timeframe(timeframe)
            
        # Set primary timeframe
        tf_manager.set_primary_timeframe(primary_tf)
        
        # Generate enhanced multi-timeframe analysis
        try:
            multi_tf_analysis = tf_manager.enhanced_multi_timeframe_analysis(result.get('market_regime'))
        except (AttributeError, Exception) as e:
            # Fallback to regular analysis if enhanced method isn't available
            print(f"Enhanced multi-timeframe analysis not available, using basic: {str(e)}")
            multi_tf_analysis = tf_manager.analyze_multi_timeframe_signals()
        
        # Calculate target prices and stop losses based on ATR
        if 'buy_signal' in signals.columns or 'sell_signal' in signals.columns:
            try:
                for idx in signals.index:
                    atr_value = atr.loc[idx] if idx in atr.index and not pd.isna(atr.loc[idx]) else signals['signal_price'].iloc[-1] * 0.01
                    
                    if signals.loc[idx, 'buy_signal']:
                        signals.loc[idx, 'target_price'] = signals.loc[idx, 'signal_price'] + (atr_value * 3)
                        signals.loc[idx, 'stop_loss'] = signals.loc[idx, 'signal_price'] - (atr_value * 1.5)
                    elif signals.loc[idx, 'sell_signal']:
                        signals.loc[idx, 'target_price'] = signals.loc[idx, 'signal_price'] - (atr_value * 3)
                        signals.loc[idx, 'stop_loss'] = signals.loc[idx, 'signal_price'] + (atr_value * 1.5)
            except Exception as e:
                print(f"Error calculating target prices: {str(e)}")
        
        return {
            'signals': signals,
            'metrics': result,
            'primary_timeframe': primary_tf,
            'multi_timeframe_analysis': multi_tf_analysis
        }
        
    except Exception as e:
        print(f"Error in advanced signal generation: {str(e)}")
        return {
            'signals': pd.DataFrame(),
            'metrics': {'error': f'Error generating signals: {str(e)}'},
            'primary_timeframe': primary_tf
        }

def analyze_single_day(data: pd.DataFrame, symbol: str = None) -> Dict[str, Any]:
    """
    Analyze a single day of trading data and generate intraday signals
    
    Args:
        data: DataFrame with OHLCV data for a single day
        symbol: Trading symbol (optional)
        
    Returns:
        Dictionary containing signals and analysis results
    """
    try:
        print(f"Starting single day analysis with {len(data)} data points")
        
        # Check for sufficient data
        if data is None or len(data) < 10:
            print("Insufficient data for single day analysis")
            return {
                "success": False,
                "error": "Insufficient data for analysis",
                "signals": pd.DataFrame()
            }
        
        # Special case: Use SPY-specific strategy for SPY symbol
        if symbol and symbol.upper() == "SPY" and analyze_spy_day_trading is not None:
            print("Using SPY-specific day trading strategy")
            try:
                spy_results = analyze_spy_day_trading(data)
                if spy_results.get("success", False):
                    return spy_results
                else:
                    print(f"SPY strategy failed: {spy_results.get('error', 'Unknown error')}. Falling back to standard analysis.")
            except Exception as e:
                print(f"Error in SPY day trading analysis: {str(e)}. Falling back to standard analysis.")
                # Fall through to standard analysis
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Ensure data is for a single day
        if isinstance(df.index[0], pd.Timestamp):
            # Get date of first candle (in Eastern time if possible)
            if df.index[0].tzinfo is not None:
                eastern_tz = pytz.timezone('US/Eastern')
                first_date = df.index[0].tz_convert(eastern_tz).date()
            else:
                first_date = df.index[0].date()
            
            # Filter data for just this one day
            if df.index[-1].date() != first_date:
                print(f"Filtering data to single day: {first_date}")
                df = df[df.index.date == first_date]
                
                if len(df) < 10:
                    print("Insufficient data after filtering to single day")
                    return {
                        "success": False,
                        "error": "Insufficient data after filtering to single day",
                        "signals": pd.DataFrame()
                    }
        
        # Extract market session data
        session_data = analyze_session_data(df)
        
        # Calculate opening range
        or_data = calculate_opening_range(df, minutes=15)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=df.index)
        
        # Generate basic signal columns
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['buy_score'] = 0.0  
        signals['sell_score'] = 0.0
        signals['signal_strength'] = 0.0
        signals['target_price'] = 0.0
        signals['stop_loss'] = 0.0
        signals['signal_price'] = 0.0  # Price at signal time
        
        # Convert timestamps to Eastern Time for display
        signals['signal_time'] = df.index
        if isinstance(df.index[0], pd.Timestamp) and df.index[0].tzinfo is not None:
            eastern_tz = pytz.timezone('US/Eastern')
            signals['signal_time'] = df.index.tz_convert(eastern_tz)
        
        # Calculate indicators
        # 1. EMA cloud
        fast_ema, slow_ema = calculate_ema_cloud(df, fast_period=9, slow_period=21)
        df['fast_ema'] = fast_ema
        df['slow_ema'] = slow_ema
        
        # 2. VWAP
        try:
            df['vwap'] = calculate_vwap(df)
        except Exception as e:
            print(f"Error calculating VWAP: {e}")
            df['vwap'] = df['close'].mean()
        
        # 3. RSI
        try:
            df['rsi'] = calculate_rsi(df, period=14)
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            df['rsi'] = 50  # Neutral value
        
        # 4. Stochastic
        try:
            df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, k_period=14, d_period=3)
        except Exception as e:
            print(f"Error calculating Stochastic: {e}")
            df['stoch_k'] = 50
            df['stoch_d'] = 50
        
        # 5. ATR for stop loss calculation
        try:
            df['atr'] = calculate_atr(df, period=14)
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            df['atr'] = (df['high'] - df['low']).mean()
        
        # Detect ORB signals
        orb_signals = detect_orb_breakout(df, or_data)
        for col in orb_signals.columns:
            signals[col] = orb_signals[col]
        
        # Get current time (in Eastern) for time-based strategies
        now = datetime.now(pytz.timezone('US/Eastern'))
        current_time = now.time()
        
        # Determine which session we're in
        morning_session = current_time >= pd.Timestamp('09:30:00').time() and current_time < pd.Timestamp('11:30:00').time()
        midday_session = current_time >= pd.Timestamp('11:30:00').time() and current_time < pd.Timestamp('14:30:00').time()
        closing_session = current_time >= pd.Timestamp('14:30:00').time() and current_time <= pd.Timestamp('16:00:00').time()
        
        # Apply different strategies based on session
        for i in range(len(df)):
            idx = df.index[i]
            
            # Skip the first few candles as they often have high volatility
            if i < 3:
                continue
            
            # Base variables for signal calculation
            buy_score = 0.0
            sell_score = 0.0
            
            # ---- Morning session strategies ----
            if morning_session:
                # 1. Trend following with EMA cloud
                if df.loc[idx, 'close'] > df.loc[idx, 'fast_ema'] > df.loc[idx, 'slow_ema']:
                    buy_score += 0.2
                elif df.loc[idx, 'close'] < df.loc[idx, 'fast_ema'] < df.loc[idx, 'slow_ema']:
                    sell_score += 0.2
                
                # 2. ORB strategy weight is higher in morning session
                if 'orb_breakout_up' in signals and signals.loc[idx, 'orb_breakout_up']:
                    buy_score += 0.3
                if 'orb_breakout_down' in signals and signals.loc[idx, 'orb_breakout_down']:
                    sell_score += 0.3
                
                # 3. VWAP bounces are important in morning
                if df.loc[idx, 'close'] > df.loc[idx, 'vwap'] and df.loc[idx-1, 'low'] < df.loc[idx-1, 'vwap']:
                    buy_score += 0.2  # Bounced up from VWAP
                if df.loc[idx, 'close'] < df.loc[idx, 'vwap'] and df.loc[idx-1, 'high'] > df.loc[idx-1, 'vwap']:
                    sell_score += 0.2  # Bounced down from VWAP
            
            # ---- Midday session strategies ----
            elif midday_session:
                # 1. Range trading often works better in midday
                middle_band = (df['high'].iloc[max(0, i-10):i+1].max() + df['low'].iloc[max(0, i-10):i+1].min()) / 2
                
                if df.loc[idx, 'close'] < middle_band and df.loc[idx, 'stoch_k'] < 20 and df.loc[idx, 'stoch_k'] > df.loc[idx, 'stoch_d']:
                    buy_score += 0.25  # Oversold in range
                if df.loc[idx, 'close'] > middle_band and df.loc[idx, 'stoch_k'] > 80 and df.loc[idx, 'stoch_k'] < df.loc[idx, 'stoch_d']:
                    sell_score += 0.25  # Overbought in range
                
                # 2. Volume spikes can indicate breakouts in slow sessions
                avg_volume = df['volume'].iloc[max(0, i-10):i].mean()
                if df.loc[idx, 'volume'] > 2 * avg_volume:
                    if df.loc[idx, 'close'] > df.loc[idx, 'open']:
                        buy_score += 0.15  # Volume spike on up candle
                    else:
                        sell_score += 0.15  # Volume spike on down candle
            
            # ---- Closing session strategies ----
            elif closing_session:
                # 1. Trend continuation is more likely in closing session
                if df.loc[idx, 'close'] > df.loc[idx, 'vwap'] and df.loc[idx, 'close'] > df.loc[idx, 'fast_ema']:
                    buy_score += 0.2
                if df.loc[idx, 'close'] < df.loc[idx, 'vwap'] and df.loc[idx, 'close'] < df.loc[idx, 'fast_ema']:
                    sell_score += 0.2
                
                # 2. Look for late day momentum
                if df.loc[idx, 'rsi'] > 60 and df.loc[idx, 'rsi'] > df.loc[idx-1, 'rsi']:
                    buy_score += 0.2
                if df.loc[idx, 'rsi'] < 40 and df.loc[idx, 'rsi'] < df.loc[idx-1, 'rsi']:
                    sell_score += 0.2
            
            # ---- Universal strategies (apply in all sessions) ----
            
            # 1. RSI divergence
            if i >= 5:
                # Bullish divergence
                if (df.loc[idx-5:idx, 'close'].min() < df.loc[idx-5:idx-1, 'close'].min() and 
                    df.loc[idx-5:idx, 'rsi'].min() > df.loc[idx-5:idx-1, 'rsi'].min() and
                    df.loc[idx, 'rsi'] < 40):
                    buy_score += 0.25
                
                # Bearish divergence
                if (df.loc[idx-5:idx, 'close'].max() > df.loc[idx-5:idx-1, 'close'].max() and 
                    df.loc[idx-5:idx, 'rsi'].max() < df.loc[idx-5:idx-1, 'rsi'].max() and
                    df.loc[idx, 'rsi'] > 60):
                    sell_score += 0.25
            
            # 2. Price action patterns
            # Bullish engulfing
            if (df.loc[idx, 'close'] > df.loc[idx, 'open'] and 
                df.loc[idx-1, 'close'] < df.loc[idx-1, 'open'] and
                df.loc[idx, 'close'] > df.loc[idx-1, 'open'] and
                df.loc[idx, 'open'] < df.loc[idx-1, 'close']):
                buy_score += 0.2
            
            # Bearish engulfing
            if (df.loc[idx, 'close'] < df.loc[idx, 'open'] and 
                df.loc[idx-1, 'close'] > df.loc[idx-1, 'open'] and
                df.loc[idx, 'close'] < df.loc[idx-1, 'open'] and
                df.loc[idx, 'open'] > df.loc[idx-1, 'close']):
                sell_score += 0.2
            
            # Assign scores and generate signals if they exceed thresholds
            signals.loc[idx, 'buy_score'] = buy_score
            signals.loc[idx, 'sell_score'] = sell_score
            
            # Only generate signals if scores exceed thresholds
            if buy_score > 0.4:
                signals.loc[idx, 'buy_signal'] = True
                signals.loc[idx, 'signal_strength'] = min(1.0, buy_score)
                signals.loc[idx, 'signal_price'] = df.loc[idx, 'close']
                
                # Set target and stop based on ATR
                atr_value = df.loc[idx, 'atr']
                signals.loc[idx, 'target_price'] = df.loc[idx, 'close'] + (atr_value * 2 * signals.loc[idx, 'signal_strength'])
                signals.loc[idx, 'stop_loss'] = df.loc[idx, 'close'] - (atr_value * 1 * signals.loc[idx, 'signal_strength'])
            
            elif sell_score > 0.4:
                signals.loc[idx, 'sell_signal'] = True
                signals.loc[idx, 'signal_strength'] = min(1.0, sell_score)
                signals.loc[idx, 'signal_price'] = df.loc[idx, 'close']
                
                # Set target and stop based on ATR
                atr_value = df.loc[idx, 'atr']
                signals.loc[idx, 'target_price'] = df.loc[idx, 'close'] - (atr_value * 2 * signals.loc[idx, 'signal_strength'])
                signals.loc[idx, 'stop_loss'] = df.loc[idx, 'close'] + (atr_value * 1 * signals.loc[idx, 'signal_strength'])
        
        # Get only rows with signals
        signal_rows = signals[(signals['buy_signal'] == True) | (signals['sell_signal'] == True)]
        
        # Return analysis results
        return {
            "success": True,
            "data": {
                "session_data": session_data,
                "opening_range": or_data,
                "signals": signals,
                "signal_rows": signal_rows,
                "last_close": df['close'].iloc[-1] if not df.empty else None,
                "last_data_point": df.index[-1] if not df.empty else None
            }
        }
    
    except Exception as e:
        print(f"Error in single day analysis: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "signals": pd.DataFrame()
        }

def analyze_session_data(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract market session data from the dataframe
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Dictionary with pre-market, regular hours, and post-market session data
    """
    eastern_tz = pytz.timezone('US/Eastern')
    
    # Convert index to eastern time if needed
    if isinstance(data.index[0], pd.Timestamp):
        df = data.copy()
        # If timezone info is missing, assume UTC and convert to Eastern
        if df.index[0].tzinfo is None:
            df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert(eastern_tz)
    else:
        # If not timestamp index, just use the original data
        df = data
    
    # Define market sessions
    pre_market_start = pd.Timestamp('04:00:00', tz=eastern_tz).time()
    market_open = pd.Timestamp('09:30:00', tz=eastern_tz).time()
    market_close = pd.Timestamp('16:00:00', tz=eastern_tz).time()
    post_market_end = pd.Timestamp('20:00:00', tz=eastern_tz).time()
    
    # Filter data for each session
    if isinstance(df.index[0], pd.Timestamp):
        pre_market = df[(df.index.time >= pre_market_start) & (df.index.time < market_open)]
        regular_hours = df[(df.index.time >= market_open) & (df.index.time <= market_close)]
        post_market = df[(df.index.time > market_close) & (df.index.time <= post_market_end)]
    else:
        # If not timestamp index, return empty DataFrames
        pre_market = pd.DataFrame()
        regular_hours = pd.DataFrame()
        post_market = pd.DataFrame()
    
    # Calculate session stats
    pm_stats = {}
    rh_stats = {}
    post_stats = {}
    
    if not pre_market.empty:
        pm_stats = {
            'open': pre_market['open'].iloc[0],
            'high': pre_market['high'].max(),
            'low': pre_market['low'].min(),
            'close': pre_market['close'].iloc[-1],
            'volume': pre_market['volume'].sum()
        }
    
    if not regular_hours.empty:
        rh_stats = {
            'open': regular_hours['open'].iloc[0],
            'high': regular_hours['high'].max(),
            'low': regular_hours['low'].min(),
            'close': regular_hours['close'].iloc[-1],
            'volume': regular_hours['volume'].sum()
        }
    
    if not post_market.empty:
        post_stats = {
            'open': post_market['open'].iloc[0],
            'high': post_market['high'].max(),
            'low': post_market['low'].min(),
            'close': post_market['close'].iloc[-1],
            'volume': post_market['volume'].sum()
        }
    
    return {
        'pre_market': pm_stats,
        'regular_hours': rh_stats,
        'post_market': post_stats
    }

def calculate_opening_range(data: pd.DataFrame, minutes: int = 15) -> Dict[str, float]:
    """
    Calculate the opening range for a given number of minutes
    
    Args:
        data: DataFrame with OHLCV data
        minutes: Number of minutes to consider for the opening range
        
    Returns:
        Dictionary with high and low of the opening range
    """
    eastern_tz = pytz.timezone('US/Eastern')
    
    # Convert index to eastern time if needed
    if isinstance(data.index[0], pd.Timestamp):
        df = data.copy()
        # If timezone info is missing, assume UTC and convert to Eastern
        if df.index[0].tzinfo is None:
            df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert(eastern_tz)
    else:
        # If not timestamp index, just use the original data
        df = data
    
    # Define market open
    market_open = pd.Timestamp('09:30:00', tz=eastern_tz).time()
    
    # Find market open in the data
    if isinstance(df.index[0], pd.Timestamp):
        # Look for the first candle that starts at or after market open
        market_open_candles = df[df.index.time >= market_open]
        
        if market_open_candles.empty:
            # If no candles found at or after market open, use the first candle in the dataset
            opening_range_end_time = df.index[0] + pd.Timedelta(minutes=minutes)
            opening_range = df[df.index <= opening_range_end_time]
        else:
            # Get the first market open candle
            first_candle_time = market_open_candles.index[0]
            # Calculate end time for the opening range
            opening_range_end_time = first_candle_time + pd.Timedelta(minutes=minutes)
            # Filter data for the opening range
            opening_range = df[(df.index >= first_candle_time) & (df.index <= opening_range_end_time)]
    else:
        # If not timestamp index, just use the first N candles
        opening_range = df.iloc[:max(1, min(minutes, len(df)))]
    
    # Calculate opening range high and low
    if opening_range.empty:
        # Return default values if no data is available
        return {
            'high': df['high'].iloc[0] if not df.empty else 0,
            'low': df['low'].iloc[0] if not df.empty else 0
        }
    else:
        return {
            'high': opening_range['high'].max(),
            'low': opening_range['low'].min()
        }

def detect_orb_breakout(data: pd.DataFrame, or_data: Dict[str, float]) -> pd.DataFrame:
    """
    Detect Opening Range Breakout (ORB) signals
    
    Args:
        data: DataFrame with OHLCV data
        or_data: Dictionary with opening range high and low values
        
    Returns:
        DataFrame with ORB breakout signals
    """
    if data is None or len(data) < 2 or 'high' not in data.columns or 'low' not in data.columns:
        return pd.DataFrame(index=data.index if data is not None else [])
    
    or_high = or_data.get('high', 0)
    or_low = or_data.get('low', 0)
    
    if or_high <= 0 or or_low <= 0:
        return pd.DataFrame(index=data.index)
    
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=data.index)
    signals['orb_breakout_up'] = False
    signals['orb_breakout_down'] = False
    
    # Used to track if we've already identified a breakout
    up_breakout_found = False
    down_breakout_found = False
    
    eastern_tz = pytz.timezone('US/Eastern')
    market_open_time = pd.Timestamp('09:30:00', tz=eastern_tz).time()
    
    # First, find the opening range end time
    if isinstance(data.index[0], pd.Timestamp):
        # Check if index has timezone info, if not assume UTC
        if data.index[0].tzinfo is None:
            df_idx = pd.DatetimeIndex(data.index).tz_localize('UTC').tz_convert(eastern_tz)
        else:
            df_idx = data.index.tz_convert(eastern_tz)
        
        # Find candles after market open
        after_open_mask = df_idx.time >= market_open_time
        if after_open_mask.any():
            first_candle_after_open = df_idx[after_open_mask][0]
            # Skip OR formation period
            opening_range_end_time = first_candle_after_open + pd.Timedelta(minutes=15)
            or_formation_mask = df_idx <= opening_range_end_time
            skip_indices = data.index[or_formation_mask]
        else:
            # If no candles after market open, use first 15 minutes of available data
            opening_range_end_time = df_idx[0] + pd.Timedelta(minutes=15)
            or_formation_mask = df_idx <= opening_range_end_time
            skip_indices = data.index[or_formation_mask]
    else:
        # If not timestamp index, assume first 15 candles are the opening range
        skip_indices = data.index[:min(15, len(data))]
    
    # Check each candle for breakouts, skipping the OR formation period
    for i in range(len(data)):
        idx = data.index[i]
        
        # Skip OR formation period
        if idx in skip_indices:
            continue
        
        # Check for upside breakout if not already found
        if not up_breakout_found and data.loc[idx, 'high'] > or_high:
            signals.loc[idx, 'orb_breakout_up'] = True
            up_breakout_found = True
        
        # Check for downside breakout if not already found
        if not down_breakout_found and data.loc[idx, 'low'] < or_low:
            signals.loc[idx, 'orb_breakout_down'] = True
            down_breakout_found = True
        
        # If both breakouts found, we can stop processing
        if up_breakout_found and down_breakout_found:
            break
    
    return signals 

def generate_signals_multi_timeframe(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Generate trading signals using the multi-timeframe integration framework.
    This implementation uses a three-tier structure to identify signals with higher confidence.
    
    Args:
        data_dict: Dictionary with timeframe names as keys and OHLCV DataFrames as values
            Must contain at least three timeframes: higher (1h/15m), middle (5m), and lower (1m)
            
    Returns:
        Dictionary with signal results including recommendations and analysis
    """
    print(f"Starting multi-timeframe signal generation with {len(data_dict)} timeframes")
    
    # Sanity check for data
    if not data_dict or len(data_dict) < 3:
        print("Insufficient timeframes for multi-timeframe analysis. Need at least 3 timeframes.")
        return {
            "success": False,
            "error": "Insufficient timeframes. Need at least 3 timeframes (1h/15m, 5m, 1m).",
            "signals": pd.DataFrame()
        }
    
    try:
        # Initialize multi-timeframe framework
        mtf = MultiTimeframeFramework()
        
        # Map timeframes to appropriate tiers
        for tf_name, tf_data in data_dict.items():
            if not isinstance(tf_data, pd.DataFrame) or tf_data.empty:
                continue
                
            # Determine tier based on timeframe name
            tier = None
            if tf_name in ['1h', '15m']:
                tier = TimeframeTier.TREND  # Higher TF for trend direction
            elif tf_name in ['5m']:
                tier = TimeframeTier.SIGNAL  # Middle TF for trading opportunities
            elif tf_name in ['1m']:
                tier = TimeframeTier.ENTRY  # Lower TF for precise entry
            else:
                tier = TimeframeTier.CONFIRMATION  # Additional confirmation timeframe
                
            # Add to framework
            mtf.add_timeframe(name=tf_name, data=tf_data, tier=tier, interval=tf_name)
            
        # Analyze all timeframes
        analysis_results = mtf.analyze_all_timeframes()
        
        # Get trading recommendation
        recommendation = mtf.get_trading_recommendation()
        
        # Create signals DataFrame based on the primary trading timeframe (signal timeframe)
        if mtf.signal_tf and mtf.signal_tf in data_dict:
            signal_data = data_dict[mtf.signal_tf]
            signals = pd.DataFrame(index=signal_data.index)
            
            # Initialize signal columns
            signals['buy_signal'] = False
            signals['sell_signal'] = False
            signals['buy_score'] = 0.0
            signals['sell_score'] = 0.0
            signals['signal_strength'] = 0.0
            signals['signal_price'] = signal_data['close']
            signals['target_price'] = 0.0
            signals['stop_loss'] = 0.0
            
            # Set signal for the latest candle
            if recommendation['action'] == 'BUY':
                signals.iloc[-1, signals.columns.get_indexer(['buy_signal'])[0]] = True
                signals.iloc[-1, signals.columns.get_indexer(['buy_score'])[0]] = recommendation['confidence'] * 10
                signals.iloc[-1, signals.columns.get_indexer(['signal_strength'])[0]] = recommendation['trend_strength']
                signals.iloc[-1, signals.columns.get_indexer(['target_price'])[0]] = recommendation['target']
                signals.iloc[-1, signals.columns.get_indexer(['stop_loss'])[0]] = recommendation['stop']
            
            elif recommendation['action'] == 'SELL':
                signals.iloc[-1, signals.columns.get_indexer(['sell_signal'])[0]] = True
                signals.iloc[-1, signals.columns.get_indexer(['sell_score'])[0]] = recommendation['confidence'] * 10
                signals.iloc[-1, signals.columns.get_indexer(['signal_strength'])[0]] = recommendation['trend_strength']
                signals.iloc[-1, signals.columns.get_indexer(['target_price'])[0]] = recommendation['target']
                signals.iloc[-1, signals.columns.get_indexer(['stop_loss'])[0]] = recommendation['stop']
            
            # Return both the signals and the detailed analysis
            return {
                "success": True,
                "signals": signals,
                "recommendation": recommendation,
                "analysis": analysis_results,
                "timeframes": list(data_dict.keys())
            }
        else:
            # Fallback if the signal timeframe is not available
            print("Signal timeframe not found in data_dict")
            return {
                "success": False,
                "error": "Signal timeframe not found or unavailable",
                "signals": pd.DataFrame()
            }
            
    except Exception as e:
        print(f"Error in multi-timeframe signal generation: {str(e)}")
        traceback.print_exc()
        
        # Return an empty but valid result
        return {
            "success": False,
            "error": str(e),
            "signals": pd.DataFrame()
        } 