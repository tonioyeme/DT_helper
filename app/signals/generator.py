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
    multi_indicator_confirmation,
    calculate_roc, 
    calculate_hull_moving_average, 
    calculate_ttm_squeeze,
    detect_hidden_divergence
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
    # Create a generator instance (this is a quick solution)
    generator = create_default_signal_generator()
    
    # Create a data dictionary with a default timeframe
    data_dict = {"1h": data}  # Using 1h as the default timeframe
    
    # Process the data using the generator
    generator.process_data(data, "1h")
    
    # Use the enhanced signal generation to get signals
    result = generator.generate_signals(data_dict)
    
    # If the result is a dictionary with signals, extract it
    if isinstance(result, dict) and "primary_signals" in result:
        # Get the signals from the primary timeframe
        signals = pd.DataFrame(index=data.index)
        
        # Copy over the signals data
        for key, value in result["primary_signals"].items():
            # Skip nested dictionaries or complex objects
            if not isinstance(value, (dict, list, tuple, pd.DataFrame, pd.Series)) and not callable(value):
                signals[key] = value
                
        # Ensure basic signal columns exist
        if "buy_signal" not in signals.columns:
            signals["buy_signal"] = False
        if "sell_signal" not in signals.columns:
            signals["sell_signal"] = False
        if "signal_strength" not in signals.columns:
            signals["signal_strength"] = 0
            
        # Set price and time information
        signals["signal_price"] = data["close"]
        eastern = pytz.timezone('US/Eastern')
        signals["signal_time_et"] = [idx.astimezone(eastern).strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'astimezone') else str(idx) 
                                  for idx in signals.index]
                                  
        # Set signal scores if available
        if "buy_strength" in result["primary_signals"]:
            signals["buy_score"] = result["primary_signals"]["buy_strength"]
        if "sell_strength" in result["primary_signals"]:
            signals["sell_score"] = result["primary_signals"]["sell_strength"]
            
        # Generate some dummy target and stop loss values
        for idx in signals.index:
            if signals.loc[idx, "buy_signal"]:
                signals.loc[idx, "target_price"] = data.loc[idx, "close"] * 1.02
                signals.loc[idx, "stop_loss"] = data.loc[idx, "close"] * 0.98
            elif signals.loc[idx, "sell_signal"]:
                signals.loc[idx, "target_price"] = data.loc[idx, "close"] * 0.98
                signals.loc[idx, "stop_loss"] = data.loc[idx, "close"] * 1.02
                
        # Try to generate buy/sell counts for visualization
        signals["buy_count"] = 0
        signals["sell_count"] = 0
        
        # Initialize confirmation count columns
        signals["buy_confirm_count"] = 0
        signals["sell_confirm_count"] = 0
        
        if "indicator_signals" in result:
            for layer_name, layer_data in result["indicator_signals"].items():
                for indicator_name, indicator_data in layer_data.items():
                    if indicator_data.get("buy_signal", False):
                        signals["buy_count"] += 1
                        signals["buy_confirm_count"] += 1
                    if indicator_data.get("sell_signal", False):
                        signals["sell_count"] += 1
                        signals["sell_confirm_count"] += 1
                        
        # Set signal strength based on confirmation count
        for idx in signals.index:
            # Map signal strength based on confirmation count
            if signals.at[idx, 'buy_confirm_count'] >= 3:
                signals.at[idx, 'signal_strength'] = SignalStrength.VERY_STRONG.value
            elif signals.at[idx, 'buy_confirm_count'] >= 2:
                signals.at[idx, 'signal_strength'] = SignalStrength.STRONG.value
            elif signals.at[idx, 'buy_confirm_count'] >= 1:
                signals.at[idx, 'signal_strength'] = SignalStrength.MODERATE.value
            else:
                signals.at[idx, 'signal_strength'] = SignalStrength.WEAK.value
                
            # Map sell signal strength
            if signals.at[idx, 'sell_confirm_count'] >= 3:
                signals.at[idx, 'signal_strength'] = SignalStrength.VERY_STRONG.value
            elif signals.at[idx, 'sell_confirm_count'] >= 2:
                signals.at[idx, 'signal_strength'] = SignalStrength.STRONG.value
            elif signals.at[idx, 'sell_confirm_count'] >= 1:
                signals.at[idx, 'signal_strength'] = SignalStrength.MODERATE.value
            else:
                signals.at[idx, 'signal_strength'] = SignalStrength.WEAK.value
                
            # Set strong signals flags
            signals.at[idx, 'strong_buy_signal'] = signals.at[idx, 'buy_signal'] and signals.at[idx, 'signal_strength'] >= SignalStrength.STRONG.value
            signals.at[idx, 'strong_sell_signal'] = signals.at[idx, 'sell_signal'] and signals.at[idx, 'signal_strength'] >= SignalStrength.STRONG.value
            
        return signals
        
    # If using the generator's signals failed, try the advanced approach
    try:
        advanced_results = generate_signals_advanced({"default": data}, "default")
        if "signals" in advanced_results and isinstance(advanced_results["signals"], pd.DataFrame):
            return advanced_results["signals"]
    except Exception as e:
        print(f"Error in advanced signal generation: {str(e)}")
    
    # Fallback to a minimal set of signals if all else fails
    logger = logging.getLogger(__name__)
    logger.warning("Falling back to minimal signal dataset")
    
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