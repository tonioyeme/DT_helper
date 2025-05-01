"""
Multi-Timeframe Integration Framework for Day Trading.
Implements a systematic approach to analyzing market data across multiple timeframes
for more reliable trading signals and precise entry/exit points.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import logging

from app.signals.timeframes import TimeFrame, TimeFrameManager, TimeFramePriority, TimeFrameAlignment

class TimeframeTier(Enum):
    """
    Represents the different tiers in the multi-timeframe analysis structure
    """
    TREND = 3       # Higher timeframe for trend direction (15-min/1-hour)
    SIGNAL = 2      # Middle timeframe for trading opportunities (5-min)
    ENTRY = 1       # Lower timeframe for precise entries (1-min)
    CONFIRMATION = 0  # Additional timeframe for confirmation

class TrendDirection(Enum):
    """
    Possible trend directions
    """
    BULLISH = 2     # Upward trend
    NEUTRAL = 1     # Sideways or unclear
    BEARISH = 0     # Downward trend

class MultiTimeframeFramework:
    """
    Implements the three-tier timeframe structure for day trading
    with primary trend direction, trading opportunities, and entry timing
    """
    
    def __init__(self):
        """Initialize the multi-timeframe framework"""
        self.timeframes = {}
        self.trend_tf = None  # Higher TF (15-min/1-hour)
        self.signal_tf = None  # Middle TF (5-min)
        self.entry_tf = None  # Lower TF (1-min)
        self.logger = logging.getLogger(__name__)
        
    def add_timeframe(self, 
                     name: str, 
                     data: pd.DataFrame, 
                     tier: TimeframeTier,
                     interval: str):
        """
        Add a timeframe to the framework
        
        Args:
            name: Identifier for the timeframe
            data: DataFrame with OHLCV data
            tier: The tier/role this timeframe plays in the framework
            interval: Timeframe interval (e.g., "1m", "5m", "1h")
        """
        self.timeframes[name] = {
            'data': data,
            'tier': tier,
            'interval': interval,
            'analysis': None  # Will store analysis results
        }
        
        # Set as primary timeframe for its tier
        if tier == TimeframeTier.TREND and self.trend_tf is None:
            self.trend_tf = name
        elif tier == TimeframeTier.SIGNAL and self.signal_tf is None:
            self.signal_tf = name
        elif tier == TimeframeTier.ENTRY and self.entry_tf is None:
            self.entry_tf = name
    
    def analyze_trend(self, tf_name: str) -> Dict[str, Any]:
        """
        Analyze trend direction and strength for a given timeframe
        
        Args:
            tf_name: Name of the timeframe to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if tf_name not in self.timeframes:
            return {'error': f"Timeframe {tf_name} not found"}
            
        data = self.timeframes[tf_name]['data']
        
        if data is None or len(data) < 20:
            return {'error': f"Insufficient data for trend analysis in {tf_name}"}
            
        # Calculate trend indicators
        try:
            # Calculate EMAs for trend direction
            data['ema20'] = data['close'].ewm(span=20, adjust=False).mean()
            data['ema50'] = data['close'].ewm(span=50, adjust=False).mean()
            
            # Calculate ADX for trend strength (simplified)
            # In a real implementation, use a proper ADX calculation
            data['tr1'] = abs(data['high'] - data['low'])
            data['tr2'] = abs(data['high'] - data['close'].shift(1))
            data['tr3'] = abs(data['low'] - data['close'].shift(1))
            data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
            data['atr14'] = data['tr'].rolling(14).mean()
            
            # Simple momentum
            data['momentum'] = data['close'] - data['close'].shift(10)
            
            # Last values
            last_ema20 = data['ema20'].iloc[-1]
            last_ema50 = data['ema50'].iloc[-1]
            last_close = data['close'].iloc[-1]
            last_momentum = data['momentum'].iloc[-1]
            
            # Determine trend direction
            if last_ema20 > last_ema50 and last_close > last_ema20:
                trend_direction = TrendDirection.BULLISH
                trend_strength = min(1.0, abs(last_ema20 - last_ema50) / last_ema50 * 100)
            elif last_ema20 < last_ema50 and last_close < last_ema20:
                trend_direction = TrendDirection.BEARISH
                trend_strength = min(1.0, abs(last_ema20 - last_ema50) / last_ema50 * 100)
            else:
                trend_direction = TrendDirection.NEUTRAL
                trend_strength = 0.5
                
            # Volume confirmation
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            last_volume = data['volume'].iloc[-1]
            volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Store analysis results
            analysis = {
                'trend': trend_direction,
                'strength': trend_strength,
                'momentum': last_momentum,
                'ema_delta': last_ema20 - last_ema50,
                'volume_ratio': volume_ratio
            }
            
            self.timeframes[tf_name]['analysis'] = analysis
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend for {tf_name}: {str(e)}")
            return {'error': str(e)}
    
    def analyze_all_timeframes(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all timeframes in the framework
        
        Returns:
            Dictionary with analysis results for each timeframe
        """
        results = {}
        
        for name in self.timeframes:
            results[name] = self.analyze_trend(name)
            
        return results
    
    def multi_tf_confirmation(self) -> Dict[str, Any]:
        """
        Check for trading confirmation across multiple timeframes
        
        Returns:
            Dictionary with confirmation results
        """
        # Ensure we have the main timeframes
        if not all([self.trend_tf, self.signal_tf, self.entry_tf]):
            missing = []
            if not self.trend_tf: missing.append("trend")
            if not self.signal_tf: missing.append("signal")
            if not self.entry_tf: missing.append("entry")
            return {'confirmed': False, 'error': f"Missing timeframes: {', '.join(missing)}"}
        
        # Get analysis for each timeframe
        trend_analysis = self.timeframes[self.trend_tf]['analysis']
        signal_analysis = self.timeframes[self.signal_tf]['analysis']
        entry_analysis = self.timeframes[self.entry_tf]['analysis']
        
        if not all([trend_analysis, signal_analysis, entry_analysis]):
            return {'confirmed': False, 'error': "Missing analysis for one or more timeframes"}
        
        # Check primary trend alignment
        trend_aligned = trend_analysis['trend'] == signal_analysis['trend']
        momentum_confirmation = signal_analysis.get('momentum', 0) > entry_analysis.get('momentum', 0)
        
        # Volume confirmation across timeframes
        trend_volume = self.timeframes[self.trend_tf]['data']['volume'].iloc[-1]
        trend_volume_mean = self.timeframes[self.trend_tf]['data']['volume'].rolling(20).mean().iloc[-1]
        signal_volume = self.timeframes[self.signal_tf]['data']['volume'].iloc[-1]
        signal_volume_mean = self.timeframes[self.signal_tf]['data']['volume'].rolling(20).mean().iloc[-1]
        
        volume_confirmed = (trend_volume > trend_volume_mean) and (signal_volume > signal_volume_mean)
        
        # Calculate overall confirmation
        confirmed = trend_aligned and momentum_confirmation
        
        # Return detailed confirmation results
        return {
            'confirmed': confirmed,
            'trend_aligned': trend_aligned,
            'momentum_confirmed': momentum_confirmation,
            'volume_confirmed': volume_confirmed,
            'primary_trend': trend_analysis['trend'],
            'confidence': 0.5 + (0.25 if trend_aligned else 0) + 
                         (0.15 if momentum_confirmation else 0) + 
                         (0.1 if volume_confirmed else 0)
        }
    
    def get_trading_recommendation(self) -> Dict[str, Any]:
        """
        Generate trading recommendation based on multi-timeframe analysis
        
        Returns:
            Dictionary with trading recommendation
        """
        # First analyze all timeframes
        self.analyze_all_timeframes()
        
        # Check for multi-timeframe confirmation
        confirmation = self.multi_tf_confirmation()
        
        if not confirmation['confirmed']:
            return {
                'action': 'WAIT',
                'reason': confirmation.get('error', 'No confirmation across timeframes'),
                'confidence': 0.0
            }
        
        # Get the trend direction from higher timeframe
        trend_direction = self.timeframes[self.trend_tf]['analysis']['trend']
        
        # Determine recommended action based on trend
        action = 'WAIT'
        if trend_direction == TrendDirection.BULLISH and confirmation['confirmed']:
            action = 'BUY'
        elif trend_direction == TrendDirection.BEARISH and confirmation['confirmed']:
            action = 'SELL'
            
        # Calculate target and stop based on ATR from signal timeframe
        signal_data = self.timeframes[self.signal_tf]['data']
        entry_price = signal_data['close'].iloc[-1]
        
        # Simple ATR calculation
        if 'atr14' in signal_data.columns:
            atr = signal_data['atr14'].iloc[-1]
        else:
            atr = signal_data['tr'].rolling(14).mean().iloc[-1]
            
        if action == 'BUY':
            target = entry_price + (atr * 2.0)
            stop = entry_price - (atr * 1.0)
        elif action == 'SELL':
            target = entry_price - (atr * 2.0)
            stop = entry_price + (atr * 1.0)
        else:
            target = entry_price
            stop = entry_price
            
        return {
            'action': action,
            'entry_price': entry_price,
            'target': target,
            'stop': stop,
            'risk_reward': abs(target - entry_price) / abs(stop - entry_price) if abs(stop - entry_price) > 0 else 0,
            'confidence': confirmation['confidence'],
            'trend_strength': self.timeframes[self.trend_tf]['analysis']['strength'],
            'volume_confirmed': confirmation['volume_confirmed']
        }

# Helper function for external calling
def multi_tf_confirmation(primary_tf, trading_tf, entry_tf):
    """
    Check if multiple timeframes confirm a trading opportunity
    
    Args:
        primary_tf: Higher timeframe data dict (trend direction)
        trading_tf: Middle timeframe data dict (trading opportunities) 
        entry_tf: Lower timeframe data dict (entry timing)
        
    Returns:
        Boolean indicating if the signal is confirmed across timeframes
    """
    # Check trend alignment
    if (primary_tf.get('trend') == trading_tf.get('trend') 
        and trading_tf.get('momentum', 0) > entry_tf.get('momentum', 0)):
        return True
    return False 