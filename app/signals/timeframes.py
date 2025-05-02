import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import pytz
import logging

class TimeFrameStrength(Enum):
    """Enum representing the importance weight of different timeframes"""
    MINUTE_1 = 0.5
    MINUTE_5 = 0.7
    MINUTE_15 = 1.0
    MINUTE_30 = 1.2
    HOUR_1 = 1.5
    HOUR_4 = 1.8
    DAY_1 = 2.0
    WEEK_1 = 2.5

class TimeFrameAlignment(Enum):
    """Represents alignment status between timeframes"""
    STRONGLY_ALIGNED = 3    # All timeframes agree
    ALIGNED = 2             # Most timeframes agree, including higher ones
    MIXED = 1               # Some agreement, some disagreement
    CONFLICTING = 0         # Significant disagreement across timeframes

class TimeFramePriority(Enum):
    """Priority levels for different timeframes in signal validation"""
    PRIMARY = 3    # Primary trading timeframe (highest priority)
    SECONDARY = 2  # Secondary confirmation timeframe
    TERTIARY = 1   # Additional context timeframe
    CONTEXT = 0    # Background context only (lowest priority)

class TimeFrame:
    """
    Class representing a timeframe for analysis with its data and priority
    """
    
    def __init__(self, 
                 name: str, 
                 interval: str,
                 priority: TimeFramePriority = TimeFramePriority.SECONDARY,
                 weight: float = 1.0):
        """
        Initialize a timeframe
        
        Args:
            name: Name of the timeframe (e.g., "1H", "Daily")
            interval: Interval string (e.g., "1h", "1d")
            priority: Priority level of this timeframe
            weight: Weight for signals from this timeframe (higher = more important)
        """
        self.name = name
        self.interval = interval
        self.priority = priority
        self.weight = weight
        self.data = None
        self.signals = None
        
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set price data for this timeframe
        
        Args:
            data: DataFrame with OHLCV data for this timeframe
        """
        self.data = data
        
    def set_signals(self, signals: Dict[str, Any]) -> None:
        """
        Set calculated signals for this timeframe
        
        Args:
            signals: Dictionary of signal data
        """
        self.signals = signals

class TimeFrameManager:
    """
    Manages multiple timeframes for multi-timeframe analysis
    """
    
    def __init__(self):
        """Initialize the timeframe manager"""
        self.timeframes: Dict[str, TimeFrame] = {}
        self.primary_timeframe: Optional[str] = None
        self.logger = logging.getLogger(__name__)
        
    def add_timeframe(self, timeframe: TimeFrame) -> None:
        """
        Add a timeframe to the manager
        
        Args:
            timeframe: TimeFrame object to add
        """
        self.timeframes[timeframe.name] = timeframe
        
        # If this is the first PRIMARY timeframe or there's no primary set yet, make it primary
        if (self.primary_timeframe is None) or \
           (timeframe.priority == TimeFramePriority.PRIMARY and 
            (self.primary_timeframe is None or 
             self.timeframes[self.primary_timeframe].priority != TimeFramePriority.PRIMARY)):
            self.primary_timeframe = timeframe.name
            
    def remove_timeframe(self, timeframe_name: str) -> None:
        """
        Remove a timeframe from the manager
        
        Args:
            timeframe_name: Name of the timeframe to remove
        """
        if timeframe_name in self.timeframes:
            # If removing the primary timeframe, need to find a new one
            if timeframe_name == self.primary_timeframe:
                self.primary_timeframe = None
                
                # Find the highest priority timeframe to become the new primary
                highest_priority = -1
                for name, tf in self.timeframes.items():
                    if name != timeframe_name and tf.priority.value > highest_priority:
                        highest_priority = tf.priority.value
                        self.primary_timeframe = name
                        
            del self.timeframes[timeframe_name]
            
    def set_primary_timeframe(self, timeframe_name: str) -> None:
        """
        Set a specific timeframe as primary
        
        Args:
            timeframe_name: Name of the timeframe to set as primary
        """
        if timeframe_name in self.timeframes:
            self.primary_timeframe = timeframe_name
        else:
            raise ValueError(f"Timeframe '{timeframe_name}' not found")
            
    def get_timeframe_data(self, timeframe_name: str) -> Optional[pd.DataFrame]:
        """
        Get price data for a specific timeframe
        
        Args:
            timeframe_name: Name of the timeframe
            
        Returns:
            DataFrame with price data or None if not available
        """
        if timeframe_name in self.timeframes and self.timeframes[timeframe_name].data is not None:
            return self.timeframes[timeframe_name].data
        return None
        
    def get_timeframe_signals(self, timeframe_name: str) -> Optional[Dict[str, Any]]:
        """
        Get signals for a specific timeframe
        
        Args:
            timeframe_name: Name of the timeframe
            
        Returns:
            Signal data or None if not available
        """
        if timeframe_name in self.timeframes and self.timeframes[timeframe_name].signals is not None:
            return self.timeframes[timeframe_name].signals
        return None
        
    def align_timestamps(self) -> Dict[str, pd.DataFrame]:
        """
        Align timestamps across all timeframes to ensure data consistency
        
        Returns:
            Dictionary mapping timeframe names to aligned DataFrames
        """
        aligned_data = {}
        
        # First, determine the common date range
        min_date = None
        max_date = None
        
        for name, tf in self.timeframes.items():
            if tf.data is not None and not tf.data.empty:
                tf_min = tf.data.index.min()
                tf_max = tf.data.index.max()
                
                if min_date is None or tf_min > min_date:
                    min_date = tf_min
                    
                if max_date is None or tf_max < max_date:
                    max_date = tf_max
                    
        if min_date is None or max_date is None:
            return aligned_data  # No data to align
            
        # Now filter each timeframe to the common range
        for name, tf in self.timeframes.items():
            if tf.data is not None and not tf.data.empty:
                # Filter to common date range
                mask = (tf.data.index >= min_date) & (tf.data.index <= max_date)
                aligned_data[name] = tf.data.loc[mask].copy()
                
        return aligned_data
        
    def resample_to_higher_timeframe(self, data: pd.DataFrame, 
                                     source_interval: str, 
                                     target_interval: str) -> pd.DataFrame:
        """
        Resample data from a lower timeframe to a higher timeframe
        
        Args:
            data: DataFrame with OHLCV data
            source_interval: Source interval (e.g., "1m", "5m")
            target_interval: Target interval (e.g., "1h", "4h")
            
        Returns:
            Resampled DataFrame
        """
        # Convert interval strings to pandas offset aliases
        interval_map = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '12h': '12H',
            '1d': 'D', '1w': 'W', '1M': 'M'
        }
        
        source_alias = interval_map.get(source_interval.lower())
        target_alias = interval_map.get(target_interval.lower())
        
        if source_alias is None or target_alias is None:
            raise ValueError(f"Unsupported interval: {source_interval} or {target_interval}")
            
        # Resample using OHLC method
        resampled = data.resample(target_alias).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled
        
    def downsample_to_lower_timeframe(self, data: pd.DataFrame, 
                                      source_interval: str,
                                      target_interval: str) -> pd.DataFrame:
        """
        Downsample data from a higher timeframe to a lower timeframe (approximate)
        
        Args:
            data: DataFrame with OHLCV data
            source_interval: Source interval (e.g., "1d", "4h")
            target_interval: Target interval (e.g., "1h", "15m")
            
        Returns:
            Downsampled DataFrame (uses forward fill for approximation)
        """
        # This is an approximation - true downsampling would require more data points
        # We'll create a new range of timestamps at the target frequency
        
        # Determine frequency in minutes
        interval_to_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720,
            '1d': 1440, '1w': 10080
        }
        
        source_mins = interval_to_minutes.get(source_interval.lower())
        target_mins = interval_to_minutes.get(target_interval.lower())
        
        if source_mins is None or target_mins is None:
            raise ValueError(f"Unsupported interval: {source_interval} or {target_interval}")
            
        if target_mins >= source_mins:
            raise ValueError("Target interval must be smaller than source interval for downsampling")
            
        # Create new index at target frequency
        start_date = data.index.min()
        end_date = data.index.max()
        
        # Create new date range with target frequency
        new_index = pd.date_range(start=start_date, end=end_date, freq=f"{target_mins}min")
        
        # Create new DataFrame with the expanded index
        downsampled = pd.DataFrame(index=new_index, columns=data.columns)
        
        # Forward fill from original data
        for idx in data.index:
            nearest_idx = new_index[new_index >= idx][0]  # Find the nearest new index
            downsampled.loc[nearest_idx] = data.loc[idx]
            
        # Forward fill the remaining NaN values
        downsampled = downsampled.ffill()
        
        return downsampled
        
    def analyze_multi_timeframe_signals(self) -> Dict[str, Any]:
        """
        Analyze signals across all timeframes to generate a consolidated view
        
        Returns:
            Dictionary with consolidated signal analysis
        """
        if not self.timeframes or self.primary_timeframe is None:
            return {"error": "No timeframes available or no primary timeframe set"}
            
        # Get primary timeframe signals
        primary_signals = self.get_timeframe_signals(self.primary_timeframe)
        if primary_signals is None:
            return {"error": "No signals available for primary timeframe"}
            
        # Start with primary timeframe signals
        consolidated = {
            "primary_timeframe": self.primary_timeframe,
            "primary_signals": primary_signals,
            "confirmed_by": [],
            "conflicts_with": [],
            "buy_signal": primary_signals.get("buy_signal", False),
            "sell_signal": primary_signals.get("sell_signal", False),
            "neutral_signal": primary_signals.get("neutral_signal", False),
            "buy_strength": primary_signals.get("buy_score", 0),
            "sell_strength": primary_signals.get("sell_score", 0),
            "confidence": 0.0
        }
        
        # Track confirmation and conflict counts for confidence calculation
        confirmation_weight = 0
        conflict_weight = 0
        total_weight = self.timeframes[self.primary_timeframe].weight
        
        # Check each timeframe against the primary
        for name, tf in self.timeframes.items():
            if name == self.primary_timeframe or tf.signals is None:
                continue
                
            tf_signals = tf.signals
            
            # Determine if this timeframe confirms or conflicts with primary
            confirms = False
            conflicts = False
            
            # Get boolean values safely
            primary_buy = bool(primary_signals.get("buy_signal", False))
            primary_sell = bool(primary_signals.get("sell_signal", False))
            primary_neutral = bool(primary_signals.get("neutral_signal", False))
            
            tf_buy = bool(tf_signals.get("buy_signal", False))
            tf_sell = bool(tf_signals.get("sell_signal", False))
            tf_neutral = bool(tf_signals.get("neutral_signal", False))
            
            if primary_buy and tf_buy:
                confirms = True
            elif primary_sell and tf_sell:
                confirms = True
            elif primary_neutral and tf_neutral:
                confirms = True
            elif (primary_buy and tf_sell) or (primary_sell and tf_buy):
                conflicts = True
                
            # Add to confirmation or conflict lists
            if confirms:
                consolidated["confirmed_by"].append({
                    "timeframe": name,
                    "priority": tf.priority.value,
                    "weight": tf.weight,
                    "buy_strength": tf_signals.get("buy_score", 0),
                    "sell_strength": tf_signals.get("sell_score", 0)
                })
                confirmation_weight += tf.weight
            elif conflicts:
                consolidated["conflicts_with"].append({
                    "timeframe": name,
                    "priority": tf.priority.value,
                    "weight": tf.weight,
                    "buy_strength": tf_signals.get("buy_score", 0),
                    "sell_strength": tf_signals.get("sell_score", 0)
                })
                conflict_weight += tf.weight
                
            total_weight += tf.weight
            
        # Calculate confidence based on confirmations vs conflicts
        # Weighted by timeframe priorities
        if total_weight > 0:
            # Base confidence on primary timeframe signal strength
            base_confidence = 0.5
            if primary_signals.get("buy_signal", False):
                base_confidence = min(0.8, 0.4 + (primary_signals.get("buy_score", 0) / 10) * 0.4)
            elif primary_signals.get("sell_signal", False):
                base_confidence = min(0.8, 0.4 + (primary_signals.get("sell_score", 0) / 10) * 0.4)
                
            # Adjust based on confirmations and conflicts
            confirmation_factor = confirmation_weight / total_weight
            conflict_factor = conflict_weight / total_weight
            
            confidence_adjustment = (confirmation_factor * 0.2) - (conflict_factor * 0.4)
            consolidated["confidence"] = max(0, min(1.0, base_confidence + confidence_adjustment))
            
        # Modify signal strength based on confirmations/conflicts
        if consolidated["buy_signal"]:
            # Boost buy strength if confirmed by other timeframes
            for confirmation in consolidated["confirmed_by"]:
                consolidated["buy_strength"] += confirmation["buy_strength"] * (confirmation["weight"] / total_weight) * 0.2
            
            # Reduce buy strength if conflicts exist
            if consolidated["conflicts_with"]:
                consolidated["buy_strength"] *= (1 - (conflict_weight / total_weight) * 0.5)
                
        elif consolidated["sell_signal"]:
            # Boost sell strength if confirmed by other timeframes
            for confirmation in consolidated["confirmed_by"]:
                consolidated["sell_strength"] += confirmation["sell_strength"] * (confirmation["weight"] / total_weight) * 0.2
            
            # Reduce sell strength if conflicts exist
            if consolidated["conflicts_with"]:
                consolidated["sell_strength"] *= (1 - (conflict_weight / total_weight) * 0.5)
                
        # Determine final signal based on adjusted strengths
        if not consolidated["conflicts_with"] or consolidated["confidence"] >= 0.6:
            # Keep the primary signal if confident enough
            pass
        else:
            # If low confidence due to conflicts, neutralize the signal
            consolidated["buy_signal"] = False
            consolidated["sell_signal"] = False
            consolidated["neutral_signal"] = True
            consolidated["buy_strength"] = 0
            consolidated["sell_strength"] = 0
            
        return consolidated
        
    def enhanced_multi_timeframe_analysis(self, market_regime=None) -> Dict[str, Any]:
        """
        Enhanced multi-timeframe signal analysis with dynamic weighting based on
        current market conditions and historical performance
        
        Args:
            market_regime: Optional market regime to adjust weights
            
        Returns:
            Dictionary with consolidated signal analysis and enhanced metrics
        """
        # Start with basic analysis from the regular method
        base_analysis = self.analyze_multi_timeframe_signals()
        
        if "error" in base_analysis:
            return base_analysis
            
        # Create enhanced analysis with additional metrics
        enhanced = {
            **base_analysis,
            "market_regime": market_regime,
            "weighted_signals": {},
            "timeframe_alignment": 0.0,
            "component_scores": {},
            "regime_adjusted_confidence": 0.0
        }
        
        # Define timeframe weights by importance/length
        timeframe_importance = {
            # General weights for timeframe importance in different market regimes
            "default": {
                "1m": 0.6,  # Less important, more noise
                "5m": 0.7, 
                "15m": 0.8,
                "30m": 0.9,
                "1h": 1.0,  # Base reference
                "4h": 1.2,
                "1d": 1.5,
                "1w": 2.0   # More important for overall trend
            },
            "trending": {  # When market is trending
                "1m": 0.5,  # Even less important in trends
                "5m": 0.6,
                "15m": 0.7,
                "30m": 0.8,
                "1h": 1.0,
                "4h": 1.3, 
                "1d": 1.7,  # Higher weights for longer timeframes in trends
                "1w": 2.2
            },
            "ranging": {   # When market is range-bound
                "1m": 0.8,  # More important in ranges
                "5m": 0.9,
                "15m": 1.0,
                "30m": 1.0,
                "1h": 1.0,
                "4h": 0.9,  # Less important
                "1d": 0.8,
                "1w": 0.7   # Less important in ranges
            },
            "volatile": {  # In volatile markets
                "1m": 0.5,  # Too noisy
                "5m": 0.6,
                "15m": 0.7,
                "30m": 0.8,
                "1h": 1.0,
                "4h": 1.2,
                "1d": 1.4,
                "1w": 1.6
            }
        }
        
        # Determine which set of weights to use based on market regime
        weight_key = "default"
        if market_regime:
            regime_name = str(market_regime).lower()
            if "bull" in regime_name or "bear" in regime_name:
                weight_key = "trending"
            elif "range" in regime_name or "sideways" in regime_name:
                weight_key = "ranging"
            elif "volatile" in regime_name or "breakout" in regime_name:
                weight_key = "volatile"
        
        # Get the weights for the current market condition
        regime_weights = timeframe_importance.get(weight_key, timeframe_importance["default"])
        
        # Track agreement and disagreement across timeframes
        agreement_score = 0.0
        total_comparisons = 0
        
        # Calculate weighted signals for each timeframe
        weighted_signals = {}
        
        for tf_name, tf in self.timeframes.items():
            if tf.signals is None:
                continue
                
            # Get dynamic weight based on timeframe
            tf_base_weight = regime_weights.get(tf_name, 1.0)
            
            # Apply priority multiplier
            priority_multiplier = 1.0
            if tf.priority == TimeFramePriority.PRIMARY:
                priority_multiplier = 1.4
            elif tf.priority == TimeFramePriority.SECONDARY:
                priority_multiplier = 1.2
            elif tf.priority == TimeFramePriority.TERTIARY:
                priority_multiplier = 1.0
            elif tf.priority == TimeFramePriority.CONTEXT:
                priority_multiplier = 0.8
                
            # Calculate final weight
            final_weight = tf_base_weight * priority_multiplier
            
            # Store signal with its weight
            weighted_signals[tf_name] = {
                "signals": tf.signals,
                "weight": final_weight
            }
            
        enhanced["weighted_signals"] = weighted_signals
        
        # Calculate alignment score across all timeframes
        # Higher value means better agreement between timeframes
        for i, (tf1_name, tf1_data) in enumerate(weighted_signals.items()):
            for j, (tf2_name, tf2_data) in enumerate(weighted_signals.items()):
                if i >= j:  # Skip self-comparisons and duplicates
                    continue
                    
                # Calculate similarity between two timeframes
                tf1_signals = tf1_data["signals"]
                tf2_signals = tf2_data["signals"]
                
                # Check signal agreement
                # Get the boolean values safely using bool() for any ambiguous values
                tf1_buy = bool(tf1_signals.get("buy_signal", False))
                tf1_sell = bool(tf1_signals.get("sell_signal", False))
                tf1_neutral = bool(tf1_signals.get("neutral_signal", False))
                
                tf2_buy = bool(tf2_signals.get("buy_signal", False))
                tf2_sell = bool(tf2_signals.get("sell_signal", False))
                tf2_neutral = bool(tf2_signals.get("neutral_signal", False))
                
                # Check signal agreement
                if (tf1_buy and tf2_buy) or (tf1_sell and tf2_sell) or (tf1_neutral and tf2_neutral):
                    agreement = 1.0
                # Check partial agreement (one neutral, one directional)
                elif (tf1_neutral or tf2_neutral):
                    agreement = 0.5
                # Complete disagreement
                else:
                    agreement = 0.0
                    
                # Weight the agreement by the importance of both timeframes
                weighted_agreement = agreement * (tf1_data["weight"] + tf2_data["weight"]) / 2
                
                agreement_score += weighted_agreement
                total_comparisons += 1
                
        # Calculate final alignment score (0-1 range)
        if total_comparisons > 0:
            enhanced["timeframe_alignment"] = agreement_score / total_comparisons
        
        # Adjust confidence based on timeframe alignment
        if "confidence" in enhanced:
            regime_adjustment = 1.0
            
            # Apply regime-specific confidence adjustments
            if market_regime:
                regime_name = str(market_regime).lower()
                if "bull" in regime_name:
                    if enhanced.get("buy_signal", False):
                        regime_adjustment = 1.2  # Increase confidence in bull market buy signals
                    else:
                        regime_adjustment = 0.8  # Decrease confidence in bull market sell signals
                elif "bear" in regime_name:
                    if enhanced.get("sell_signal", False):
                        regime_adjustment = 1.2  # Increase confidence in bear market sell signals
                    else:
                        regime_adjustment = 0.8  # Decrease confidence in bear market buy signals
                elif "volatile" in regime_name:
                    regime_adjustment = 0.85  # Decrease confidence in volatile markets
                
            # Calculate final confidence with alignment and regime adjustment
            alignment_factor = enhanced["timeframe_alignment"]
            base_confidence = enhanced["confidence"]
            
            # Higher alignment increases confidence, lower alignment decreases it
            alignment_adjustment = (alignment_factor - 0.5) * 0.3
            
            # Apply both adjustments
            enhanced["regime_adjusted_confidence"] = max(0.0, min(1.0, 
                (base_confidence + alignment_adjustment) * regime_adjustment
            ))
            
        # Calculate component scores for visualization
        component_scores = {}
        
        if self.primary_timeframe and self.primary_timeframe in self.timeframes:
            primary_tf = self.timeframes[self.primary_timeframe]
            if primary_tf.signals:
                # Extract component scores if available
                for key, value in primary_tf.signals.items():
                    if key.endswith("_score") and key not in ["buy_score", "sell_score"]:
                        component_scores[key] = value
                        
        enhanced["component_scores"] = component_scores
        
        return enhanced

# Utility functions for timeframe conversion and management

def create_standard_timeframes() -> TimeFrameManager:
    """
    Create a standard set of timeframes for analysis
    
    Returns:
        TimeFrameManager with common timeframes configured
    """
    manager = TimeFrameManager()
    
    # Add common timeframes with appropriate priorities
    manager.add_timeframe(TimeFrame(
        name="1m", 
        interval="1m",
        priority=TimeFramePriority.TERTIARY,
        weight=0.6
    ))
    
    manager.add_timeframe(TimeFrame(
        name="5m", 
        interval="5m",
        priority=TimeFramePriority.TERTIARY,
        weight=0.7
    ))
    
    manager.add_timeframe(TimeFrame(
        name="15m", 
        interval="15m",
        priority=TimeFramePriority.SECONDARY,
        weight=0.8
    ))
    
    manager.add_timeframe(TimeFrame(
        name="1h", 
        interval="1h",
        priority=TimeFramePriority.PRIMARY,
        weight=1.0
    ))
    
    manager.add_timeframe(TimeFrame(
        name="4h", 
        interval="4h",
        priority=TimeFramePriority.SECONDARY,
        weight=0.9
    ))
    
    manager.add_timeframe(TimeFrame(
        name="1d", 
        interval="1d",
        priority=TimeFramePriority.CONTEXT,
        weight=0.7
    ))
    
    return manager

def get_higher_timeframe(timeframe: str) -> str:
    """
    Get the next higher timeframe
    
    Args:
        timeframe: Current timeframe string (e.g., "5m", "1h")
        
    Returns:
        Next higher timeframe
    """
    timeframe_order = [
        "1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M"
    ]
    
    try:
        current_idx = timeframe_order.index(timeframe)
        if current_idx < len(timeframe_order) - 1:
            return timeframe_order[current_idx + 1]
        return timeframe  # Already at highest timeframe
    except ValueError:
        # If not found, return original
        return timeframe
        
def get_lower_timeframe(timeframe: str) -> str:
    """
    Get the next lower timeframe
    
    Args:
        timeframe: Current timeframe string (e.g., "1h", "15m")
        
    Returns:
        Next lower timeframe
    """
    timeframe_order = [
        "1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M"
    ]
    
    try:
        current_idx = timeframe_order.index(timeframe)
        if current_idx > 0:
            return timeframe_order[current_idx - 1]
        return timeframe  # Already at lowest timeframe
    except ValueError:
        # If not found, return original
        return timeframe

class MultiTimeFrameAnalyzer:
    """
    Analyzes trading signals across multiple timeframes to validate signals
    and assess their strength based on timeframe alignment.
    """
    
    def __init__(self):
        """Initialize the multi-timeframe analyzer"""
        self.timeframes = {}
        self.signal_evaluator = None
        
    def add_timeframe(self, name: str, data: pd.DataFrame, weight: Optional[float] = None):
        """
        Add a timeframe dataset for analysis
        
        Args:
            name: Identifier for the timeframe (e.g., "1m", "5m", "1h")
            data: Price data for this timeframe
            weight: Optional custom weight for this timeframe
        """
        # Get standard weight if possible
        std_weight = None
        try:
            tf_name = f"{'_'.join(name.upper().split())}"
            if hasattr(TimeFrameStrength, tf_name):
                std_weight = getattr(TimeFrameStrength, tf_name).value
        except (AttributeError, ValueError):
            pass
            
        # Use provided weight or standard weight or default to 1.0
        final_weight = weight if weight is not None else (std_weight if std_weight is not None else 1.0)
        
        self.timeframes[name] = {
            'data': data,
            'weight': final_weight,
            'signals': None
        }
    
    def set_evaluator(self, evaluator_func: Callable):
        """
        Set the function that will evaluate signals for each timeframe
        
        Args:
            evaluator_func: Function that takes DataFrame and returns signals DataFrame
        """
        self.signal_evaluator = evaluator_func
        
    def evaluate_all_timeframes(self):
        """
        Apply the signal evaluator to all timeframes
        
        Returns:
            Dictionary mapping timeframe names to signal DataFrames
        """
        if self.signal_evaluator is None:
            raise ValueError("Signal evaluator function not set. Use set_evaluator() first.")
            
        for name, tf_data in self.timeframes.items():
            tf_data['signals'] = self.signal_evaluator(tf_data['data'])
            
        return {name: tf_data['signals'] for name, tf_data in self.timeframes.items()}
    
    def align_timeframes(self, reference_time: datetime) -> Dict:
        """
        Find the nearest data points across all timeframes to align them
        
        Args:
            reference_time: The reference time to align around
            
        Returns:
            Dictionary with aligned signals across timeframes
        """
        aligned_signals = {}
        
        for name, tf_data in self.timeframes.items():
            if tf_data['signals'] is None:
                continue
                
            # Find nearest index to reference time
            nearest_idx = tf_data['signals'].index.get_indexer([reference_time], method='nearest')[0]
            if nearest_idx >= 0 and nearest_idx < len(tf_data['signals']):
                nearest_time = tf_data['signals'].index[nearest_idx]
                aligned_signals[name] = {
                    'time': nearest_time,
                    'signals': tf_data['signals'].iloc[nearest_idx].to_dict(),
                    'weight': tf_data['weight']
                }
                
        return aligned_signals
    
    def analyze_alignment(self, aligned_signals: Dict) -> Tuple[Dict, TimeFrameAlignment]:
        """
        Analyze the alignment of signals across timeframes
        
        Args:
            aligned_signals: Dictionary of aligned signals across timeframes
            
        Returns:
            Tuple of (composite_score, alignment_level)
        """
        # Initialize counters
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        
        # Count signal occurrences weighted by timeframe
        for tf_name, tf_data in aligned_signals.items():
            total_weight += tf_data['weight']
            
            # Safely get boolean values
            is_buy_signal = bool(tf_data['signals'].get('buy_signal', False))
            is_sell_signal = bool(tf_data['signals'].get('sell_signal', False))
            
            # Add to buy score if this timeframe has a buy signal
            if is_buy_signal:
                buy_factor = 1.0
                # Consider confirmation level if available
                if 'confirmation_level' in tf_data['signals']:
                    buy_factor = tf_data['signals']['confirmation_level'] / 2  # Scale by confirmation
                buy_score += tf_data['weight'] * buy_factor
                
            # Add to sell score if this timeframe has a sell signal
            if is_sell_signal:
                sell_factor = 1.0
                # Consider confirmation level if available
                if 'confirmation_level' in tf_data['signals']:
                    sell_factor = tf_data['signals']['confirmation_level'] / 2  # Scale by confirmation
                sell_score += tf_data['weight'] * sell_factor
        
        # Normalize scores
        if total_weight > 0:
            buy_score = buy_score / total_weight * 10  # Scale to 0-10
            sell_score = sell_score / total_weight * 10  # Scale to 0-10
        
        # Determine dominant signal
        dominant_signal = "neutral"
        if buy_score > sell_score and buy_score > 3.0:
            dominant_signal = "buy"
        elif sell_score > buy_score and sell_score > 3.0:
            dominant_signal = "sell"
            
        # Determine alignment level
        alignment_level = TimeFrameAlignment.MIXED
        score_diff = abs(buy_score - sell_score)
        max_score = max(buy_score, sell_score)
        
        if max_score > 7.0 and score_diff > 5.0:
            alignment_level = TimeFrameAlignment.STRONGLY_ALIGNED
        elif max_score > 5.0 and score_diff > 3.0:
            alignment_level = TimeFrameAlignment.ALIGNED
        elif max_score < 3.0 or score_diff < 1.0:
            alignment_level = TimeFrameAlignment.CONFLICTING
            
        # Prepare result
        result = {
            'buy_score': buy_score,
            'sell_score': sell_score,
            'dominant_signal': dominant_signal,
            'alignment_score': max_score,
            'alignment_level': alignment_level.name,
            'alignment_value': alignment_level.value,
            'timeframe_count': len(aligned_signals),
            'details': aligned_signals
        }
        
        return result, alignment_level
        
    def get_aligned_recommendation(self, reference_time: datetime) -> Dict:
        """
        Get a trading recommendation based on multi-timeframe alignment
        
        Args:
            reference_time: The reference time to align around
            
        Returns:
            Dictionary with recommendation details
        """
        # First align the timeframes
        aligned_signals = self.align_timeframes(reference_time)
        
        # Analyze alignment
        alignment_result, alignment_level = self.analyze_alignment(aligned_signals)
        
        # Prepare recommendation
        recommendation = {
            'time': reference_time,
            'signal': alignment_result['dominant_signal'],
            'confidence': alignment_result['alignment_score'],
            'alignment': alignment_result['alignment_level'],
            'entry_price': None,
            'target_price': None,
            'stop_loss': None,
            'risk_reward': None,
            'timeframes': alignment_result['details']
        }
        
        # Set entry, target and stop if we have a signal
        if alignment_result['dominant_signal'] != "neutral":
            # Find the most granular timeframe with the signal for precise entry
            best_tf = None
            best_tf_granularity = float('inf')
            
            for tf_name, tf_data in aligned_signals.items():
                # Simple heuristic: shorter timeframe names are usually more granular
                is_buy_signal = bool(tf_data['signals'].get('buy_signal', False))
                is_sell_signal = bool(tf_data['signals'].get('sell_signal', False))
                
                if ((alignment_result['dominant_signal'] == "buy" and is_buy_signal) or
                    (alignment_result['dominant_signal'] == "sell" and is_sell_signal)):
                    
                    # Convert common timeframe notation to minutes for comparison
                    granularity = float('inf')
                    if 'm' in tf_name.lower():
                        try:
                            granularity = int(tf_name.lower().replace('m', ''))
                        except ValueError:
                            pass
                    elif 'h' in tf_name.lower():
                        try:
                            granularity = int(tf_name.lower().replace('h', '')) * 60
                        except ValueError:
                            pass
                    elif 'd' in tf_name.lower():
                        try:
                            granularity = int(tf_name.lower().replace('d', '')) * 1440
                        except ValueError:
                            pass
                            
                    if granularity < best_tf_granularity:
                        best_tf_granularity = granularity
                        best_tf = tf_name
            
            # If we found a suitable timeframe, use its price targets
            if best_tf:
                tf_data = aligned_signals[best_tf]
                close_price = None
                
                # Get the close price from the original data
                orig_tf_data = self.timeframes[best_tf]['data']
                nearest_idx = orig_tf_data.index.get_indexer([tf_data['time']], method='nearest')[0]
                if nearest_idx >= 0 and nearest_idx < len(orig_tf_data):
                    close_price = orig_tf_data.iloc[nearest_idx]['close']
                
                if close_price:
                    recommendation['entry_price'] = close_price
                    
                    # Get ATR for dynamic calculations if available
                    atr_value = None
                    if 'atr' in orig_tf_data.columns:
                        atr_value = orig_tf_data.iloc[nearest_idx]['atr']
                    
                    # Get VIX level
                    vix_level = self._get_vix_level()
                    
                    # Use target and stop from signals if available, otherwise calculate
                    if 'target_price' in tf_data['signals'] and not pd.isna(tf_data['signals']['target_price']):
                        recommendation['target_price'] = tf_data['signals']['target_price']
                    elif alignment_result['dominant_signal'] == "buy":
                        # Use dynamic profit target
                        target_pct = self.dynamic_profit_target(atr_value, vix_level, close_price)
                        recommendation['target_price'] = close_price * (1 + target_pct)
                    elif alignment_result['dominant_signal'] == "sell":
                        # Use dynamic profit target for short positions
                        target_pct = self.dynamic_profit_target(atr_value, vix_level, close_price)
                        recommendation['target_price'] = close_price * (1 - target_pct)
                        
                    if 'stop_loss' in tf_data['signals'] and not pd.isna(tf_data['signals']['stop_loss']):
                        recommendation['stop_loss'] = tf_data['signals']['stop_loss']
                    elif alignment_result['dominant_signal'] == "buy":
                        # Use dynamic stop loss based on volatility
                        stop_pct = self.dynamic_stop_loss(atr_value, vix_level, close_price)
                        recommendation['stop_loss'] = close_price * (1 - stop_pct)
                    elif alignment_result['dominant_signal'] == "sell":
                        # Use dynamic stop loss for short positions
                        stop_pct = self.dynamic_stop_loss(atr_value, vix_level, close_price)
                        recommendation['stop_loss'] = close_price * (1 + stop_pct)
                        
                    # Calculate risk-reward ratio if we have both target and stop
                    if recommendation['target_price'] and recommendation['stop_loss'] and recommendation['entry_price']:
                        if alignment_result['dominant_signal'] == "buy":
                            reward = recommendation['target_price'] - recommendation['entry_price']
                            risk = recommendation['entry_price'] - recommendation['stop_loss']
                        else:  # sell signal
                            reward = recommendation['entry_price'] - recommendation['target_price']
                            risk = recommendation['stop_loss'] - recommendation['entry_price']
                            
                        if risk > 0:
                            recommendation['risk_reward'] = round(reward / risk, 2)
        
        return recommendation 
        
    def _get_vix_level(self):
        """
        Get current VIX level, either from data or default value
        
        Returns:
            float: VIX level (default 20.0 if not available)
        """
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'vix_level'):
                return st.session_state.vix_level
        except ImportError:
            pass
        
        return 20.0  # Default mid-range value
        
    def dynamic_profit_target(self, atr_value=None, vix_level=None, price=None):
        """
        Calculate dynamic profit target based on ATR and VIX
        
        Args:
            atr_value (float): Current ATR value (optional)
            vix_level (float): Current VIX level (optional)
            price (float): Current price (optional)
            
        Returns:
            float: Profit target as percentage of price
        """
        base_target = 0.006  # 0.6% base target
        
        # If we have ATR, adjust target based on volatility
        if atr_value is not None and price is not None and price > 0:
            atr_pct = atr_value / price
            base_target = max(base_target, atr_pct * 0.5)  # Target at least 50% of ATR
        
        # Adjust based on VIX level
        if vix_level is None:
            vix_level = self._get_vix_level()
            
        # Expand targets in high volatility
        if vix_level > 25:
            return base_target * 2.5  # 1.5% in high vol
        elif vix_level > 20:
            return base_target * 1.8  # 1.08% in elevated vol
        elif vix_level < 15:
            return base_target * 0.7  # 0.42% in low vol
        else:
            return base_target * 1.2  # 0.72% in normal vol
    
    def dynamic_stop_loss(self, atr_value=None, vix_level=None, price=None):
        """
        Calculate dynamic stop loss based on ATR and VIX
        
        Args:
            atr_value (float): Current ATR value (optional)
            vix_level (float): Current VIX level (optional)
            price (float): Current price (optional)
            
        Returns:
            float: Stop loss as percentage of price
        """
        base_stop = 0.004  # 0.4% base stop
        
        # If we have ATR, adjust stop based on volatility
        if atr_value is not None and price is not None and price > 0:
            atr_pct = atr_value / price
            base_stop = max(base_stop, atr_pct * 0.75)  # Stop at 75% of ATR
        
        # Adjust based on VIX level
        if vix_level is None:
            vix_level = self._get_vix_level()
            
        # Expand stops in high volatility
        if vix_level > 25:
            return base_stop * 1.8  # Wider stops in high vol
        elif vix_level > 20:
            return base_stop * 1.4  # Wider stops in elevated vol
        elif vix_level < 15:
            return base_stop * 0.9  # Tighter stops in low vol
        else:
            return base_stop * 1.0  # Normal stops otherwise 