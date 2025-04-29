import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Callable, Tuple, Optional, Any, Union

class SignalWeight(Enum):
    """Weight classification for different types of indicators"""
    PRICE_ACTION = 3.0     # Highest weight for price action
    TREND = 2.5           # Strong weight for trend indicators
    MOMENTUM = 2.0        # Moderate weight for momentum
    VOLUME = 1.5          # Lower weight for volume indicators
    OSCILLATOR = 1.0      # Base weight for oscillators
    
class ConfirmationLevel(Enum):
    """Levels of signal confirmation based on cumulative scores"""
    WEAK = 1              # Single indicator or low score
    MODERATE = 2          # Multiple indicators with moderate agreement
    STRONG = 3            # Strong agreement across indicator categories
    VERY_STRONG = 4       # Exceptional agreement across all categories

class IndicatorCategory(Enum):
    """Categories of technical indicators based on their function"""
    TREND = "trend"           # Indicators that identify trend direction and strength
    MOMENTUM = "momentum"     # Indicators measuring price momentum
    VOLATILITY = "volatility" # Indicators measuring market volatility
    VOLUME = "volume"         # Volume-based indicators
    SUPPORT_RESISTANCE = "support_resistance"  # Support/resistance level indicators
    OSCILLATOR = "oscillator" # Oscillating indicators typically bound within ranges
    PATTERN = "pattern"       # Pattern recognition indicators
    CUSTOM = "custom"         # User-defined custom indicators
    
class SignalType(Enum):
    """Types of signals that can be generated"""
    BUY = "buy"
    SELL = "sell"
    NEUTRAL = "neutral"

class Indicator:
    """
    Class representing a technical indicator with its calculation logic and
    signal generation rules.
    """
    
    def __init__(self, name: str, category: IndicatorCategory, 
                 calculate_func: Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, Any]],
                 params: Dict[str, Any] = None):
        """
        Initialize an indicator
        
        Args:
            name: Name of the indicator
            category: Category this indicator belongs to
            calculate_func: Function that calculates indicator values and signals
            params: Dictionary of parameters for the indicator calculation
        """
        self.name = name
        self.category = category
        self.calculate_func = calculate_func
        self.params = params or {}
        self.last_result = {}
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicator values and generate signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with calculated values and signals
        """
        result = self.calculate_func(data, self.params)
        self.last_result = result
        return result
        
class IndicatorLayer:
    """
    A layer containing a group of indicators that serve a similar purpose
    or have similar characteristics
    """
    
    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize an indicator layer
        
        Args:
            name: Name of the layer
            weight: Weight of this layer in the overall signal calculation
        """
        self.name = name
        self.weight = weight
        self.indicators: Dict[str, Indicator] = {}
        
    def add_indicator(self, indicator: Indicator) -> None:
        """
        Add an indicator to this layer
        
        Args:
            indicator: Indicator object to add
        """
        self.indicators[indicator.name] = indicator
        
    def remove_indicator(self, indicator_name: str) -> None:
        """
        Remove an indicator from this layer
        
        Args:
            indicator_name: Name of the indicator to remove
        """
        if indicator_name in self.indicators:
            del self.indicators[indicator_name]
            
    def calculate_all(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate all indicators in this layer
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping indicator names to their results
        """
        results = {}
        for name, indicator in self.indicators.items():
            results[name] = indicator.calculate(data)
        return results

class IndicatorLayers:
    """
    Manages multiple layers of indicators and aggregates their signals
    """
    
    def __init__(self):
        """Initialize the indicator layers manager"""
        self.layers: Dict[str, IndicatorLayer] = {}
        self.layer_order: List[str] = []
        
    def add_layer(self, layer: IndicatorLayer, position: Optional[int] = None) -> None:
        """
        Add a layer to the manager
        
        Args:
            layer: IndicatorLayer to add
            position: Optional position in the layer order (default: append)
        """
        self.layers[layer.name] = layer
        
        if position is not None:
            if position < 0 or (self.layer_order and position > len(self.layer_order)):
                raise ValueError(f"Position {position} is out of range")
                
            if layer.name in self.layer_order:
                self.layer_order.remove(layer.name)
                
            self.layer_order.insert(position, layer.name)
        else:
            if layer.name not in self.layer_order:
                self.layer_order.append(layer.name)
                
    def remove_layer(self, layer_name: str) -> None:
        """
        Remove a layer from the manager
        
        Args:
            layer_name: Name of the layer to remove
        """
        if layer_name in self.layers:
            del self.layers[layer_name]
            
        if layer_name in self.layer_order:
            self.layer_order.remove(layer_name)
            
    def calculate_layer(self, layer_name: str, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate indicators for a specific layer
        
        Args:
            layer_name: Name of the layer to calculate
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with indicator results for the layer
        """
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' not found")
            
        return self.layers[layer_name].calculate_all(data)
        
    def calculate_all(self, data: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Calculate all indicators in all layers
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Nested dictionary mapping layer names to indicator results
        """
        results = {}
        
        # Calculate in order of layer_order
        for layer_name in self.layer_order:
            if layer_name in self.layers:
                results[layer_name] = self.calculate_layer(layer_name, data)
                
        return results
        
    def get_layer_weights(self) -> Dict[str, float]:
        """
        Get weights for all layers
        
        Returns:
            Dictionary mapping layer names to their weights
        """
        return {name: layer.weight for name, layer in self.layers.items()}
        
    def set_layer_weight(self, layer_name: str, weight: float) -> None:
        """
        Set weight for a specific layer
        
        Args:
            layer_name: Name of the layer
            weight: New weight value
        """
        if layer_name in self.layers:
            self.layers[layer_name].weight = weight
            
    def aggregate_signals(self, results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Aggregate signals from all layers and indicators
        
        Args:
            results: Results from calculate_all method
            
        Returns:
            Dictionary with aggregated signals
        """
        buy_signals = []
        sell_signals = []
        neutral_signals = []
        
        # Process each layer
        for layer_name in self.layer_order:
            if layer_name not in results:
                continue
                
            layer_weight = self.layers[layer_name].weight
            layer_results = results[layer_name]
            
            # Process indicators in this layer
            for indicator_name, indicator_result in layer_results.items():
                # Check if indicator generated signals
                if 'buy_signal' in indicator_result and indicator_result['buy_signal']:
                    strength = indicator_result.get('buy_strength', 1.0)
                    buy_signals.append({
                        'layer': layer_name,
                        'indicator': indicator_name,
                        'layer_weight': layer_weight,
                        'strength': strength,
                        'effective_weight': layer_weight * strength
                    })
                    
                if 'sell_signal' in indicator_result and indicator_result['sell_signal']:
                    strength = indicator_result.get('sell_strength', 1.0)
                    sell_signals.append({
                        'layer': layer_name,
                        'indicator': indicator_name,
                        'layer_weight': layer_weight,
                        'strength': strength,
                        'effective_weight': layer_weight * strength
                    })
                    
                if ('neutral_signal' in indicator_result and indicator_result['neutral_signal']) or \
                   (not indicator_result.get('buy_signal', False) and not indicator_result.get('sell_signal', False)):
                    neutral_signals.append({
                        'layer': layer_name,
                        'indicator': indicator_name,
                        'layer_weight': layer_weight
                    })
                    
        # Calculate aggregated signal metrics
        buy_score = sum(s['effective_weight'] for s in buy_signals) if buy_signals else 0
        sell_score = sum(s['effective_weight'] for s in sell_signals) if sell_signals else 0
        
        # Normalize if we have any signals
        total_weight = sum(layer.weight for layer in self.layers.values())
        if total_weight > 0:
            buy_score = (buy_score / total_weight) * 10  # Scale to 0-10
            sell_score = (sell_score / total_weight) * 10  # Scale to 0-10
            
        # Determine final signal
        buy_signal = buy_score >= 5.0 and buy_score > sell_score
        sell_signal = sell_score >= 5.0 and sell_score > buy_score
        neutral_signal = not (buy_signal or sell_signal)
        
        # Prepare result
        return {
            'buy_score': buy_score,
            'sell_score': sell_score,
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'neutral_signal': neutral_signal,
            'buy_signals_count': len(buy_signals),
            'sell_signals_count': len(sell_signals),
            'neutral_signals_count': len(neutral_signals),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'neutral_signals': neutral_signals,
            'total_indicators': len(buy_signals) + len(sell_signals) + len(neutral_signals)
        }
        
# Factory functions to create common indicators

def create_moving_average_crossover(
    name: str,
    short_period: int = 9,
    long_period: int = 21,
    ma_type: str = 'sma'
) -> Indicator:
    """
    Factory function to create a moving average crossover indicator
    
    Args:
        name: Name for the indicator
        short_period: Period for short-term MA
        long_period: Period for long-term MA
        ma_type: Type of moving average ('sma', 'ema', 'wma')
        
    Returns:
        Indicator object
    """
    def calculate_ma_crossover(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        short_period = params['short_period']
        long_period = params['long_period']
        ma_type = params['ma_type'].lower()
        
        if ma_type == 'sma':
            short_ma = data['close'].rolling(window=short_period).mean()
            long_ma = data['close'].rolling(window=long_period).mean()
        elif ma_type == 'ema':
            short_ma = data['close'].ewm(span=short_period, adjust=False).mean()
            long_ma = data['close'].ewm(span=long_period, adjust=False).mean()
        elif ma_type == 'wma':
            # Weighted moving average
            weights_short = np.arange(1, short_period + 1)
            weights_long = np.arange(1, long_period + 1)
            
            short_ma = data['close'].rolling(window=short_period).apply(
                lambda x: np.sum(weights_short * x) / weights_short.sum(), raw=True)
            long_ma = data['close'].rolling(window=long_period).apply(
                lambda x: np.sum(weights_long * x) / weights_long.sum(), raw=True)
        else:
            raise ValueError(f"Unsupported MA type: {ma_type}")
            
        # Calculate crossover
        crossover = short_ma - long_ma
        
        # Generate signals
        buy_signal = crossover.iloc[-1] > 0 and crossover.iloc[-2] <= 0
        sell_signal = crossover.iloc[-1] < 0 and crossover.iloc[-2] >= 0
        
        # Calculate signal strength based on distance from zero
        buy_strength = 1.0
        sell_strength = 1.0
        
        if buy_signal:
            # Normalize strength: bigger gap = stronger signal
            buy_strength = min(2.0, max(0.5, abs(crossover.iloc[-1]) / data['close'].iloc[-1] * 200))
        
        if sell_signal:
            sell_strength = min(2.0, max(0.5, abs(crossover.iloc[-1]) / data['close'].iloc[-1] * 200))
        
        return {
            'short_ma': short_ma.iloc[-1],
            'long_ma': long_ma.iloc[-1],
            'crossover': crossover.iloc[-1],
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'values': {
                'short_ma': short_ma.to_dict(),
                'long_ma': long_ma.to_dict(),
                'crossover': crossover.to_dict()
            }
        }
    
    return Indicator(
        name=name,
        category=IndicatorCategory.TREND,
        calculate_func=calculate_ma_crossover,
        params={
            'short_period': short_period,
            'long_period': long_period,
            'ma_type': ma_type
        }
    )

def create_rsi_indicator(
    name: str,
    period: int = 14,
    overbought: int = 70,
    oversold: int = 30
) -> Indicator:
    """
    Factory function to create an RSI indicator
    
    Args:
        name: Name for the indicator
        period: RSI calculation period
        overbought: Overbought threshold
        oversold: Oversold threshold
        
    Returns:
        Indicator object
    """
    def calculate_rsi(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        period = params['period']
        overbought = params['overbought']
        oversold = params['oversold']
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # Buy when RSI crosses above oversold level
        buy_signal = current_rsi > oversold and prev_rsi <= oversold
        
        # Sell when RSI crosses below overbought level
        sell_signal = current_rsi < overbought and prev_rsi >= overbought
        
        # Calculate signal strength
        buy_strength = 1.0
        sell_strength = 1.0
        
        if buy_signal:
            # Deeper oversold condition = stronger buy signal
            buy_strength = min(2.0, max(0.5, 1.0 + (oversold - prev_rsi) / 15))
        
        if sell_signal:
            # Higher overbought condition = stronger sell signal
            sell_strength = min(2.0, max(0.5, 1.0 + (prev_rsi - overbought) / 15))
        
        return {
            'rsi': current_rsi,
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'values': {
                'rsi': rsi.to_dict()
            }
        }
    
    return Indicator(
        name=name,
        category=IndicatorCategory.OSCILLATOR,
        calculate_func=calculate_rsi,
        params={
            'period': period,
            'overbought': overbought,
            'oversold': oversold
        }
    )

def create_macd_indicator(
    name: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Indicator:
    """
    Factory function to create a MACD indicator
    
    Args:
        name: Name for the indicator
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Indicator object
    """
    def calculate_macd(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        fast_period = params['fast_period']
        slow_period = params['slow_period']
        signal_period = params['signal_period']
        
        # Calculate MACD
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        # Generate signals
        current_hist = macd_histogram.iloc[-1]
        prev_hist = macd_histogram.iloc[-2]
        
        # Buy signal: MACD histogram crosses above zero or MACD line crosses above signal line
        buy_signal = (current_hist > 0 and prev_hist <= 0) or \
                    (macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2])
        
        # Sell signal: MACD histogram crosses below zero or MACD line crosses below signal line
        sell_signal = (current_hist < 0 and prev_hist >= 0) or \
                      (macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2])
        
        # Calculate signal strength
        buy_strength = 1.0
        sell_strength = 1.0
        
        if buy_signal:
            # Stronger signal when both histogram is positive and MACD is above signal
            if current_hist > 0 and macd_line.iloc[-1] > signal_line.iloc[-1]:
                buy_strength = 1.5
            
            # Even stronger when MACD is also positive
            if macd_line.iloc[-1] > 0:
                buy_strength = 2.0
        
        if sell_signal:
            # Stronger signal when both histogram is negative and MACD is below signal
            if current_hist < 0 and macd_line.iloc[-1] < signal_line.iloc[-1]:
                sell_strength = 1.5
            
            # Even stronger when MACD is also negative
            if macd_line.iloc[-1] < 0:
                sell_strength = 2.0
        
        return {
            'macd_line': macd_line.iloc[-1],
            'signal_line': signal_line.iloc[-1],
            'histogram': current_hist,
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'values': {
                'macd_line': macd_line.to_dict(),
                'signal_line': signal_line.to_dict(),
                'histogram': macd_histogram.to_dict()
            }
        }
    
    return Indicator(
        name=name,
        category=IndicatorCategory.MOMENTUM,
        calculate_func=calculate_macd,
        params={
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        }
    )

def create_bollinger_bands_indicator(
    name: str,
    period: int = 20,
    std_dev: float = 2.0
) -> Indicator:
    """
    Factory function to create a Bollinger Bands indicator
    
    Args:
        name: Name for the indicator
        period: Period for moving average
        std_dev: Number of standard deviations
        
    Returns:
        Indicator object
    """
    def calculate_bollinger_bands(data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        period = params['period']
        std_dev = params['std_dev']
        
        # Calculate Bollinger Bands
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        middle_band = typical_price.rolling(window=period).mean()
        
        rolling_std = typical_price.rolling(window=period).std()
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        current_close = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2]
        
        # Generate signals
        # Buy when price crosses above lower band
        buy_signal = current_close > lower_band.iloc[-1] and prev_close <= lower_band.iloc[-2]
        
        # Sell when price crosses below upper band
        sell_signal = current_close < upper_band.iloc[-1] and prev_close >= upper_band.iloc[-2]
        
        # Calculate signal strength based on band width and position
        band_width = (upper_band.iloc[-1] - lower_band.iloc[-1]) / middle_band.iloc[-1]
        price_position = (current_close - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        
        buy_strength = 1.0
        sell_strength = 1.0
        
        if buy_signal:
            # Stronger signal when bands are narrow (lower volatility)
            if band_width < 0.05:  # Narrow bands (5% of price)
                buy_strength = 1.5
            
            # Even stronger when price was deeply oversold
            if price_position < 0.1:  # Very close to lower band
                buy_strength = 2.0
        
        if sell_signal:
            # Stronger signal when bands are narrow
            if band_width < 0.05:
                sell_strength = 1.5
            
            # Even stronger when price was deeply overbought
            if price_position > 0.9:  # Very close to upper band
                sell_strength = 2.0
        
        return {
            'middle_band': middle_band.iloc[-1],
            'upper_band': upper_band.iloc[-1],
            'lower_band': lower_band.iloc[-1],
            'band_width': band_width,
            'price_position': price_position,
            'buy_signal': buy_signal,
            'sell_signal': sell_signal,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'values': {
                'middle_band': middle_band.to_dict(),
                'upper_band': upper_band.to_dict(),
                'lower_band': lower_band.to_dict()
            }
        }
    
    return Indicator(
        name=name,
        category=IndicatorCategory.VOLATILITY,
        calculate_func=calculate_bollinger_bands,
        params={
            'period': period,
            'std_dev': std_dev
        }
    )

# Example: Create a standard setup with common indicator layers
def create_standard_layers() -> IndicatorLayers:
    """
    Create a standard setup with common indicator layers
    
    Returns:
        IndicatorLayers object with predefined layers and indicators
    """
    layers = IndicatorLayers()
    
    # Create trend layer
    trend_layer = IndicatorLayer(name="Trend", weight=1.2)
    trend_layer.add_indicator(create_moving_average_crossover("EMA 9/21", 9, 21, "ema"))
    trend_layer.add_indicator(create_moving_average_crossover("SMA 50/200", 50, 200, "sma"))
    
    # Create momentum layer
    momentum_layer = IndicatorLayer(name="Momentum", weight=1.0)
    momentum_layer.add_indicator(create_macd_indicator("MACD 12/26/9", 12, 26, 9))
    momentum_layer.add_indicator(create_rsi_indicator("RSI 14", 14, 70, 30))
    
    # Create volatility layer
    volatility_layer = IndicatorLayer(name="Volatility", weight=0.8)
    volatility_layer.add_indicator(create_bollinger_bands_indicator("BB 20/2", 20, 2.0))
    
    # Add layers to manager
    layers.add_layer(trend_layer, 0)
    layers.add_layer(momentum_layer, 1)
    layers.add_layer(volatility_layer, 2)
    
    return layers 