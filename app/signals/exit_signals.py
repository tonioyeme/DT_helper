import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pytz
from datetime import time, datetime, timedelta
import traceback

# Import indicators
from app.indicators import calculate_atr
from app.indicators.momentum import calculate_rsi, calculate_macd

# Try to import ADX calculation
try:
    from app.indicators.trend import calculate_adx
except ImportError:
    # Define a simple ADX calculation if not available
    def calculate_adx(data, period=14):
        """Simple Average Directional Index calculation"""
        # Calculate True Range
        data = data.copy()
        data['tr1'] = abs(data['high'] - data['low'])
        data['tr2'] = abs(data['high'] - data['close'].shift(1))
        data['tr3'] = abs(data['low'] - data['close'].shift(1))
        data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate +DM and -DM
        data['+dm'] = (data['high'] - data['high'].shift(1)).clip(lower=0)
        data['-dm'] = (data['low'].shift(1) - data['low']).clip(lower=0)
        
        # Calculate +DI and -DI
        data['+di'] = 100 * data['+dm'].rolling(period).mean() / data['tr'].rolling(period).mean()
        data['-di'] = 100 * data['-dm'].rolling(period).mean() / data['tr'].rolling(period).mean()
        
        # Calculate DX and ADX
        data['dx'] = 100 * abs(data['+di'] - data['-di']) / (data['+di'] + data['-di'])
        data['adx'] = data['dx'].rolling(period).mean()
        
        return data['adx']

class PositionAwareMomentum:
    """
    Position-aware momentum analysis that adapts to long vs short positions
    """
    def __init__(self):
        self.entry_price = None
        self.entry_time = None
        self.position_type = None
        self.momentum_weights = {
            'rsi': 0.3,
            'macd_divergence': 0.25,
            'price_velocity': 0.2,
            'trend_strength': 0.25
        }
        
    def set_position(self, position_type, entry_price, entry_time):
        """Set current position information"""
        self.position_type = position_type
        self.entry_price = entry_price
        self.entry_time = entry_time
        
    def calculate_momentum_strength(self, data):
        """
        Integrated momentum scoring system (0-100)
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            float: Momentum strength score (0-100)
        """
        if self.position_type is None:
            return 50  # Neutral if no position
            
        # Start with neutral baseline
        score = 40
        
        # Get latest data point
        latest = data.iloc[-1]
        
        # RSI momentum analysis (30% weight)
        rsi = latest['rsi'] if 'rsi' in latest else 50
        # For long positions, high RSI is bullish; for shorts, it's bearish
        rsi_strength = rsi if self.position_type == 'short' else (100 - rsi)
        score += self.momentum_weights['rsi'] * rsi_strength
        
        # MACD divergence (25% weight)
        if 'macd_hist' in latest:
            # Get last 3 histogram values to check trend
            hist_vals = data['macd_hist'].iloc[-3:].values
            macd_trend = np.mean(np.diff(hist_vals))
            macd_score = 0
            
            # Different interpretation for longs vs shorts
            if self.position_type == 'long':
                macd_score = 25 if macd_trend > 0 else 0
            else:  # short position
                macd_score = 25 if macd_trend < 0 else 0
                
            score += self.momentum_weights['macd_divergence'] * macd_score
        
        # Volume-adjusted price velocity (20% weight)
        if all(col in data.columns for col in ['close', 'volume', 'vol_ma']):
            # Calculate 3-period price change
            price_change = data['close'].pct_change(3).iloc[-1]
            # Adjust for position direction
            if self.position_type == 'short':
                price_change = -price_change
                
            # Volume ratio
            volume_ratio = latest['volume'] / latest['vol_ma'] if latest['vol_ma'] > 0 else 1.0
            velocity_score = (price_change * volume_ratio * 1000).clip(0, 20)
            score += self.momentum_weights['price_velocity'] * velocity_score
            
        # Trend strength (25% weight)
        adx_value = 0
        if 'adx' in latest:
            adx_value = latest['adx']
        elif len(data) > 20:
            # Calculate ADX if not present
            adx_value = calculate_adx(data).iloc[-1]
            
        # Scale ADX (0-100) to score (0-25)
        trend_score = min(25, adx_value / 4)
        score += self.momentum_weights['trend_strength'] * trend_score
        
        # Ensure score stays within 0-100 range
        return max(0, min(100, score))

    def dynamic_trailing_stop(self, data, lookback=5):
        """
        Calculate dynamic trailing stop based on ATR and recent price action
        
        Args:
            data: DataFrame with price data
            lookback: Number of periods to look back for price extremes
            
        Returns:
            float: Trailing stop price
        """
        if self.position_type is None or self.entry_price is None:
            return None
            
        # Ensure ATR is calculated
        if 'atr' not in data.columns:
            data['atr'] = calculate_atr(data)
            
        # Get current ATR and VIX-adjusted multiplier
        current_atr = data['atr'].iloc[-1]
        
        # Use VIX for volatility adjustment if available
        vix_value = data['vix'].iloc[-1] if 'vix' in data.columns else 20
        atr_multiplier = 2.0 + (vix_value / 30)  # Dynamic adjustment based on VIX
        
        if self.position_type == 'long':
            # For long positions, trailing stop is below recent highs
            recent_high = data['high'].rolling(lookback).max().iloc[-1]
            # Maximum of initial stop or trailing stop
            return max(
                self.entry_price - 2 * current_atr,  # Initial stop
                recent_high - atr_multiplier * current_atr  # Trailing stop
            )
        else:  # Short position
            # For short positions, trailing stop is above recent lows
            recent_low = data['low'].rolling(lookback).min().iloc[-1]
            # Minimum of initial stop or trailing stop
            return min(
                self.entry_price + 2 * current_atr,  # Initial stop
                recent_low + atr_multiplier * current_atr  # Trailing stop
            )

class PositionTracker:
    """
    Tracks position data including maximum favorable excursion
    for enhanced ATR filtering
    """
    def __init__(self):
        self.entry_price = None
        self.entry_time = None
        self.position_type = None
        self.high_since_entry = -np.inf
        self.low_since_entry = np.inf
        
    def set_position(self, position_type, entry_price, entry_time):
        """Set current position and reset tracking values"""
        self.position_type = position_type
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.high_since_entry = entry_price if position_type == 'long' else -np.inf
        self.low_since_entry = entry_price if position_type == 'short' else np.inf
        
    def update(self, price: float, high: float, low: float):
        """Update position tracking with new price data"""
        if self.position_type == 'long':
            self.high_since_entry = max(self.high_since_entry, high)
        elif self.position_type == 'short':
            self.low_since_entry = min(self.low_since_entry, low)
            
    def calculate_move(self, current_price: float) -> float:
        """Calculate effective move for current position"""
        if self.position_type is None or self.entry_price is None:
            return 0.0
            
        if self.position_type == 'long':
            return self.high_since_entry - self.entry_price
        else:  # short position
            return self.entry_price - self.low_since_entry
            
    def calculate_current_profit(self, current_price: float) -> float:
        """Calculate current profit/loss percentage"""
        if self.position_type is None or self.entry_price is None:
            return 0.0
            
        if self.position_type == 'long':
            return (current_price - self.entry_price) / self.entry_price
        else:  # short position
            return (self.entry_price - current_price) / self.entry_price

class SPYProfitStrategy:
    """
    Enhanced profit-taking framework specifically optimized for SPY day trading
    """
    def __init__(self):
        self.atr_period = 14  # Optimized for 5-minute bars
        self.vix_threshold = 20
        self.base_k = 2.5  # ATR multiplier
        self.last_scale_out = None  # Track last scale-out timestamp
        self.scale_out_levels = []  # Track scale-out levels
        self.initial_position_size = 1.0  # Full position
        self.current_position_size = 1.0  # Current position size after scaling
        
    def calculate_dynamic_targets(self, data, entry_price, position_type):
        """
        Volatility-adjusted profit targets with session awareness
        
        Args:
            data (pd.DataFrame): OHLCV price data with indicators
            entry_price (float): Position entry price
            position_type (str): Position type ('long' or 'short')
            
        Returns:
            dict: Dictionary with profit targets
        """
        # Ensure ATR is calculated
        if 'atr' not in data.columns:
            data['atr'] = calculate_atr(data, self.atr_period)
        
        # Get latest ATR value    
        atr = data['atr'].iloc[-1]
        
        # Get VIX for volatility adjustment
        current_vix = data['vix'].iloc[-1] if 'vix' in data.columns else get_vix_data(data)
        
        # Dynamic ATR multiplier
        k = self.base_k * (1 + (current_vix - 15)/20)  # Scales 1.25-3.75 when VIX=10-30
        
        # Session-specific adjustments
        session_info = session_adjustments(data)
        session_multiplier = session_info['k_multiplier']
        
        # Price-based targets for proper risk-reward ratio
        if position_type == 'long':
            return {
                'conservative': entry_price + (1.2 * k * atr * session_multiplier),
                'moderate': entry_price + (1.8 * k * atr * session_multiplier),
                'aggressive': entry_price + (2.5 * k * atr * session_multiplier)
            }
        else:  # short position
            return {
                'conservative': entry_price - (1.2 * k * atr * session_multiplier),
                'moderate': entry_price - (1.8 * k * atr * session_multiplier),
                'aggressive': entry_price - (2.5 * k * atr * session_multiplier)
            }
    
    def atr_trailing_stop(self, data, entry_price, position_type, elapsed_minutes):
        """
        Enhanced ATR trailing stop system with time-based adjustments
        
        Args:
            data (pd.DataFrame): OHLCV price data with indicators
            entry_price (float): Position entry price
            position_type (str): Position type ('long' or 'short')
            elapsed_minutes (float): Minutes elapsed since entry
            
        Returns:
            float: Trailing stop price
        """
        # Ensure ATR is calculated
        if 'atr' not in data.columns:
            data['atr'] = calculate_atr(data, self.atr_period)
            
        atr = data['atr'].iloc[-1]
        
        # Get 5-period rolling high/low
        highs = data['high'].rolling(5).max().iloc[-1]
        lows = data['low'].rolling(5).min().iloc[-1]
        
        # Time-based ATR multiplier adjustment (tightens over time)
        time_factor = max(0.5, 1.0 - (elapsed_minutes / 120))  # Linearly decreases over 2 hours
        
        # VIX-adjusted multiplier
        vix = data['vix'].iloc[-1] if 'vix' in data.columns else get_vix_data(data)
        vix_factor = 1.0 + ((vix - 15) / 50)  # Adjust up in high volatility
        
        # Combined multiplier
        k = 2.0 * time_factor * vix_factor
        
        if position_type == 'long':
            # Trailing stop must be above entry after reaching profit target
            if any(self.scale_out_levels) and entry_price < highs:
                return max(highs - (k * atr), entry_price)
            else:
                return max(highs - (k * atr), entry_price - (1.5 * atr))
        else:  # short position
            # Trailing stop must be below entry after reaching profit target
            if any(self.scale_out_levels) and entry_price > lows:
                return min(lows + (k * atr), entry_price)
            else:
                return min(lows + (k * atr), entry_price + (1.5 * atr))
    
    def volume_confirmation(self, data):
        """
        Volume confirmation for exit signals
        
        Args:
            data (pd.DataFrame): OHLCV price data
            
        Returns:
            bool: True if volume confirms exit
        """
        if 'volume' not in data.columns or 'vol_ma' not in data.columns:
            return True  # Default to true if volume data not available
            
        vol_ma = data['volume'].rolling(20).mean().iloc[-1]
        current_vol = data['volume'].iloc[-1]
        
        # Session-specific volume thresholds
        session_info = session_adjustments(data)
        vol_threshold = 1.3  # Default
        
        if session_info['session'] == 'power_hour':
            vol_threshold = 2.0  # Higher requirement during power hour
        elif session_info['session'] == 'midday':
            vol_threshold = 1.2  # Lower requirement during midday
            
        return current_vol > vol_threshold * vol_ma
    
    def time_based_exit(self, entry_time, current_time):
        """
        Time-based exit logic with session awareness
        
        Args:
            entry_time: Entry timestamp
            current_time: Current timestamp
            
        Returns:
            bool: True if time-based exit should be triggered
        """
        if entry_time is None or current_time is None:
            return False
            
        # Convert to Eastern Time if timezone info is available
        if hasattr(current_time, 'tzinfo') and current_time.tzinfo is not None:
            eastern = pytz.timezone('US/Eastern')
            current_time = current_time.astimezone(eastern)
            if hasattr(entry_time, 'tzinfo') and entry_time.tzinfo is not None:
                entry_time = entry_time.astimezone(eastern)
        
        # Calculate elapsed time
        elapsed = current_time - entry_time
        elapsed_minutes = elapsed.total_seconds() / 60
        
        # Force exit after 45 minutes
        if elapsed_minutes > 45:
            return True
            
        # Special handling for EOD
        if current_time.time() >= time(15, 45):
            return True  # Exit before market close
            
        return False
    
    def consecutive_close_confirm(self, data, buffer_pct=0.03):
        """
        Confirmation of exit based on consecutive closes
        
        Args:
            data (pd.DataFrame): OHLCV price data
            buffer_pct (float): Buffer percentage
            
        Returns:
            bool: True if consecutive close confirms exit
        """
        if len(data) < 3:
            return False
            
        last_close = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2]
        high_buffer = data['high'].iloc[-3] * (1 + buffer_pct)
        low_buffer = data['low'].iloc[-3] * (1 - buffer_pct)
        
        long_exit = (last_close < high_buffer) and (prev_close < high_buffer)
        short_exit = (last_close > low_buffer) and (prev_close > low_buffer)
        
        return long_exit or short_exit
    
    def momentum_exhaustion(self, data, position_type):
        """
        Detect momentum exhaustion based on technical indicators
        
        Args:
            data (pd.DataFrame): OHLCV price data with indicators
            position_type (str): Position type ('long' or 'short')
            
        Returns:
            bool: True if momentum is exhausted
        """
        # Calculate RSI if not present
        if 'rsi' not in data.columns:
            data['rsi'] = calculate_rsi(data)
            
        # Calculate MACD if not present
        if 'macd_hist' not in data.columns:
            macd, macd_signal, macd_hist = calculate_macd(data)
            data['macd'] = macd
            data['macd_signal'] = macd_signal
            data['macd_hist'] = macd_hist
            
        # Get latest values
        rsi = data['rsi'].iloc[-1]
        macd_hist = data['macd_hist'].iloc[-1]
        
        # Calculate VWAP for reference
        vwap = data['vwap'].iloc[-1] if 'vwap' in data.columns else data['close'].iloc[-1]
        close = data['close'].iloc[-1]
        
        if position_type == 'long':
            return (rsi > 68 and macd_hist < 0) or (close < vwap * 0.995)
        else:  # short position
            return (rsi < 32 and macd_hist > 0) or (close > vwap * 1.005)
    
    def execute_profit_taking(self, data, position, current_idx):
        """
        Scale-out profit taking strategy
        
        Args:
            data (pd.DataFrame): OHLCV price data
            position (dict): Position information
            current_idx: Current timestamp
            
        Returns:
            tuple: (scale_out_percent, new_stop)
        """
        if not position or 'type' not in position or 'entry_price' not in position:
            return (0, None)
            
        position_type = position['type']
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        
        # Calculate elapsed time
        elapsed_minutes = 0
        if entry_time is not None and current_idx is not None:
            elapsed = current_idx - entry_time
            elapsed_minutes = elapsed.total_seconds() / 60
        
        # Calculate profit targets
        targets = self.calculate_dynamic_targets(data, entry_price, position_type)
        current_price = data['close'].iloc[-1]
        
        # Calculate current gain percentage
        pct_gain = 0
        if position_type == 'long':
            pct_gain = (current_price - entry_price) / entry_price
        else:  # short position
            pct_gain = (entry_price - current_price) / entry_price
            
        # Default values
        scale_out_percent = 0
        new_stop = None
        
        # Scale out strategy
        if position_type == 'long':
            if current_price >= targets['conservative'] and self.current_position_size > 0.5:
                # Take 50% off at conservative target
                scale_out_percent = min(50, self.current_position_size * 100)
                new_stop = entry_price  # Move stop to break-even
                self.scale_out_levels.append('conservative')
                self.current_position_size -= 0.5
                
            elif current_price >= targets['moderate'] and self.current_position_size > 0.25:
                # Take another 25% off at moderate target
                scale_out_percent = min(25, self.current_position_size * 100)
                new_stop = max(entry_price, self.atr_trailing_stop(data, entry_price, position_type, elapsed_minutes))
                self.scale_out_levels.append('moderate')
                self.current_position_size -= 0.25
                
            elif current_price >= targets['aggressive'] and self.current_position_size > 0:
                # Final exit at aggressive target
                scale_out_percent = self.current_position_size * 100
                self.scale_out_levels.append('aggressive')
                self.current_position_size = 0
                
        else:  # short position
            if current_price <= targets['conservative'] and self.current_position_size > 0.5:
                # Take 50% off at conservative target
                scale_out_percent = min(50, self.current_position_size * 100)
                new_stop = entry_price  # Move stop to break-even
                self.scale_out_levels.append('conservative')
                self.current_position_size -= 0.5
                
            elif current_price <= targets['moderate'] and self.current_position_size > 0.25:
                # Take another 25% off at moderate target
                scale_out_percent = min(25, self.current_position_size * 100)
                new_stop = min(entry_price, self.atr_trailing_stop(data, entry_price, position_type, elapsed_minutes))
                self.scale_out_levels.append('moderate')
                self.current_position_size -= 0.25
                
            elif current_price <= targets['aggressive'] and self.current_position_size > 0:
                # Final exit at aggressive target
                scale_out_percent = self.current_position_size * 100
                self.scale_out_levels.append('aggressive')
                self.current_position_size = 0
        
        # If we're scaling out, record the time
        if scale_out_percent > 0:
            self.last_scale_out = current_idx
            
        return (scale_out_percent, new_stop)
    
    def reset(self):
        """Reset the strategy state for a new position"""
        self.last_scale_out = None
        self.scale_out_levels = []
        self.current_position_size = 1.0

class ImprovedExitSignalGenerator:
    def __init__(self):
        self.momentum_analyzer = PositionAwareMomentum()
        self.position_tracker = PositionTracker()
        self.profit_strategy = SPYProfitStrategy()
        self.current_position = None
        self.entry_price = None
        self.entry_time = None
        self.pending_exit = False
        
    def calculate_dynamic_thresholds(self, data):
        """
        Calculate volatility-adjusted dynamic thresholds
        
        Args:
            data: DataFrame with price and market data
            
        Returns:
            dict: Dictionary of dynamically adjusted threshold values
        """
        # Get VIX if available, or use an intelligent estimate
        vix_value = get_vix_data(data)
        
        # Get ADX for trend strength adjustment
        adx_value = 0
        if 'adx' in data.columns:
            adx_value = data['adx'].iloc[-1]
        else:
            # Try to calculate ADX
            try:
                adx_value = calculate_adx(data).iloc[-1]
            except:
                adx_value = 25  # Default to moderate trend
        
        # Get session adjustments for time-based parameter tuning
        session_params = session_adjustments(data)
        
        # Calculate VIX-adjusted K multiplier
        k_multiplier = dynamic_k_multiplier(vix_value) * session_params['k_multiplier']
        
        # Adjust thresholds based on volatility and trend strength
        return {
            'atr_multiplier': k_multiplier,
            'rsi_overbought': 75 if adx_value > 25 else 70,  # Strong trends can push RSI higher
            'rsi_oversold': 25 if adx_value > 25 else 30,    # Strong trends can push RSI lower
            'momentum_exit_threshold': 40 if vix_value > 25 else 45,  # More sensitive in high volatility
            'volume_threshold': 0.7 if vix_value > 30 else 0.8,  # Volume requirement adjusts with volatility
            'confirmation_count': 2 if vix_value > 25 else 3,   # Require fewer confirmations in high volatility
            'min_move': session_params['min_move'] * (1.0 + (vix_value - 20) / 100)  # Adjust min move by VIX
        }

    def generate_exit_signals(self, data, entry_signals):
        """
        Generate sophisticated exit signals with multiple confirmation filters
        
        Args:
            data (pd.DataFrame): OHLCV price data with indicators
            entry_signals (pd.DataFrame): DataFrame with buy/sell entry signals
            
        Returns:
            pd.DataFrame: DataFrame with exit signals and reasons
        """
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['exit_buy'] = False
        signals['exit_sell'] = False
        signals['exit_reason'] = None
        signals['atr_stop'] = None
        signals['effective_move'] = None  # Track effective moves
        signals['scale_out_pct'] = 0.0    # Track scaling out percentages
        signals['profit_target_hit'] = False  # Track profit targets
        
        # Calculate technical indicators if not present
        if 'rsi' not in data.columns:
            data['rsi'] = calculate_rsi(data)
        
        if 'macd' not in data.columns or 'macd_hist' not in data.columns:
            macd, signal, macd_hist = calculate_macd(data)
            data['macd'] = macd
            data['macd_signal'] = signal
            data['macd_hist'] = macd_hist
        
        if 'atr' not in data.columns:
            data['atr'] = calculate_atr(data)
        
        # Calculate ADX if not present
        if 'adx' not in data.columns:
            try:
                data['adx'] = calculate_adx(data)
            except:
                # Skip ADX if unavailable
                pass
        
        # Calculate volume MA if not present
        if 'volume' in data.columns and 'vol_ma' not in data.columns:
            data['vol_ma'] = data['volume'].rolling(20).mean()
        
        # Add VIX data if not present
        if 'vix' not in data.columns:
            # Create a VIX column with appropriate values
            vix_value = get_vix_data(data)
            data['vix'] = vix_value
        
        # Process each bar from 1st bar onward
        for i in range(1, len(data)):
            current = data.iloc[i]
            prev = data.iloc[i-1]
            current_idx = data.index[i]
            
            # Initialize effective_move to 0 at the start of each iteration
            effective_move = 0.0
            
            # Update position tracker with latest prices
            if self.position_tracker.position_type:
                self.position_tracker.update(current['close'], current['high'], current['low'])
            
            # Get adaptive thresholds with session adjustments
            thresholds = self.calculate_dynamic_thresholds(data.iloc[:i+1])
            
            # Get session-specific adjustments
            session_params = session_adjustments(data.iloc[:i+1])
            
            # Calculate VIX-adjusted ATR multiplier
            vix_k = dynamic_k_multiplier(data['vix'].iloc[-1]) * session_params['k_multiplier']
            
            # Setup position-aware momentum for both long and short scenarios
            # For long position
            self.momentum_analyzer.set_position('long', current['close'], current_idx)
            long_momentum = self.momentum_analyzer.calculate_momentum_strength(data.iloc[:i+1])
            
            # For short position
            self.momentum_analyzer.set_position('short', current['close'], current_idx)
            short_momentum = self.momentum_analyzer.calculate_momentum_strength(data.iloc[:i+1])
            
            # Calculate trailing stops with enhanced adjustment
            long_stop = current['low'] - vix_k * current['atr']
            short_stop = current['high'] + vix_k * current['atr']
            
            # Store ATR stop in signals
            signals.at[current_idx, 'atr_stop'] = long_stop if long_momentum < short_momentum else short_stop
            
            # -- NEW: Check for profit-taking opportunities --
            scale_out_pct = 0
            new_stop = None
            
            # Check if we have an active position to manage
            if self.position_tracker.position_type:
                position_info = {
                    'type': self.position_tracker.position_type,
                    'entry_price': self.position_tracker.entry_price,
                    'entry_time': self.position_tracker.entry_time
                }
                
                # Execute profit-taking strategy
                scale_out_pct, new_stop = self.profit_strategy.execute_profit_taking(
                    data.iloc[:i+1], position_info, current_idx
                )
                
                # Update signals with scale-out percentage
                signals.at[current_idx, 'scale_out_pct'] = scale_out_pct
                
                # If scaling out completely, set exit signal
                if scale_out_pct >= 100:
                    if self.position_tracker.position_type == 'long':
                        signals.at[current_idx, 'exit_buy'] = True
                        signals.at[current_idx, 'exit_reason'] = "Profit Target Hit"
                    else:  # short position
                        signals.at[current_idx, 'exit_sell'] = True
                        signals.at[current_idx, 'exit_reason'] = "Profit Target Hit"
                    signals.at[current_idx, 'profit_target_hit'] = True
                
                # Update trailing stop based on profit-taking logic if needed
                if new_stop is not None:
                    if self.position_tracker.position_type == 'long':
                        long_stop = max(long_stop, new_stop)
                    else:  # short position
                        short_stop = min(short_stop, new_stop)
            
            # Check for time-based exit
            if self.position_tracker.position_type and self.position_tracker.entry_time:
                # Check time-based exit
                if self.profit_strategy.time_based_exit(
                    self.position_tracker.entry_time, current_idx
                ):
                    if self.position_tracker.position_type == 'long':
                        signals.at[current_idx, 'exit_buy'] = True
                        signals.at[current_idx, 'exit_reason'] = "Time-Based Exit"
                    else:  # short position
                        signals.at[current_idx, 'exit_sell'] = True
                        signals.at[current_idx, 'exit_reason'] = "Time-Based Exit"
            
            # Count confirmation signals for long exit
            long_exit_confirmations = 0
            long_exit_reason = ""
            
            # RSI condition - Using adaptive overbought threshold
            rsi_condition = current['rsi'] < thresholds['rsi_overbought'] and prev['rsi'] >= thresholds['rsi_overbought']
            if rsi_condition:
                long_exit_confirmations += 1
                long_exit_reason = "RSI Weakening"
            
            # MACD histogram condition - Turning negative or decreasing
            macd_condition = (current['macd_hist'] < 0 and prev['macd_hist'] >= 0) or \
                             (current['macd_hist'] < prev['macd_hist'] and current['macd_hist'] < 0)
            if macd_condition:
                long_exit_confirmations += 1
                if not long_exit_reason:
                    long_exit_reason = "MACD Reversal"
            
            # Volume condition - Decreasing volume
            volume_condition = False
            if 'volume' in current and 'vol_ma' in current:
                volume_condition = current['volume'] < thresholds['volume_threshold'] * current['vol_ma']
                if volume_condition:
                    long_exit_confirmations += 1
                    if not long_exit_reason:
                        long_exit_reason = "Volume Decline"
            
            # Enhanced ATR-based stop condition with minimum move requirement
            atr_condition = False
            if self.position_tracker.position_type == 'long':
                # Calculate effective move using max favorable excursion
                effective_move = self.position_tracker.calculate_move(current['close'])
                # Minimum required move based on ATR and session
                min_required_move = max(
                    vix_k * current['atr'],
                    current['close'] * session_params['min_move']
                )
                # ATR stop hit or minimum move achieved
                atr_condition = (current['low'] <= long_stop) or (effective_move >= min_required_move)
            else:
                # Fallback to simple condition if position tracking not active
                atr_condition = current['low'] <= long_stop
                
            if atr_condition:
                long_exit_confirmations += 1
                if not long_exit_reason:
                    long_exit_reason = "Stop Level Hit"
            
            # Momentum score condition
            momentum_condition = long_momentum < thresholds['momentum_exit_threshold']
            if momentum_condition:
                long_exit_confirmations += 1
                if not long_exit_reason:
                    long_exit_reason = "Momentum Weakening"
            
            # -- NEW: Check for momentum exhaustion --
            if self.position_tracker.position_type == 'long' and self.profit_strategy.momentum_exhaustion(data.iloc[:i+1], 'long'):
                long_exit_confirmations += 1
                if not long_exit_reason:
                    long_exit_reason = "Momentum Exhaustion"
                    
            # -- NEW: Check for consecutive close confirmation --
            if self.position_tracker.position_type == 'long' and self.profit_strategy.consecutive_close_confirm(data.iloc[:i+1]):
                long_exit_confirmations += 1
                if not long_exit_reason:
                    long_exit_reason = "Consecutive Close Confirmation"
            
            # Generate long exit signal if enough confirmations and not already exiting due to profit target
            exit_long = long_exit_confirmations >= thresholds['confirmation_count'] and not signals.at[current_idx, 'profit_target_hit']
            signals.at[current_idx, 'exit_buy'] = signals.at[current_idx, 'exit_buy'] or exit_long
            if exit_long and not signals.at[current_idx, 'exit_reason']:
                signals.at[current_idx, 'exit_reason'] = long_exit_reason
            
            # Count confirmation signals for short exit
            short_exit_confirmations = 0
            short_exit_reason = ""
            
            # RSI condition - Using adaptive oversold threshold
            rsi_condition = current['rsi'] > thresholds['rsi_oversold'] and prev['rsi'] <= thresholds['rsi_oversold']
            if rsi_condition:
                short_exit_confirmations += 1
                short_exit_reason = "RSI Strengthening"
            
            # MACD histogram condition - Turning positive or increasing
            macd_condition = (current['macd_hist'] > 0 and prev['macd_hist'] <= 0) or \
                             (current['macd_hist'] > prev['macd_hist'] and current['macd_hist'] > 0)
            if macd_condition:
                short_exit_confirmations += 1
                if not short_exit_reason:
                    short_exit_reason = "MACD Reversal"
            
            # Volume condition - Decreasing volume
            if 'volume' in current and 'vol_ma' in current:
                volume_condition = current['volume'] < thresholds['volume_threshold'] * current['vol_ma']
                if volume_condition:
                    short_exit_confirmations += 1
                    if not short_exit_reason:
                        short_exit_reason = "Volume Decline"
            
            # Enhanced ATR-based stop condition with minimum move requirement
            atr_condition = False
            if self.position_tracker.position_type == 'short':
                # Calculate effective move using max favorable excursion
                effective_move = self.position_tracker.calculate_move(current['close'])
                # Minimum required move based on ATR and session
                min_required_move = max(
                    vix_k * current['atr'],
                    current['close'] * session_params['min_move']
                )
                # ATR stop hit or minimum move achieved
                atr_condition = (current['high'] >= short_stop) or (effective_move >= min_required_move)
            else:
                # Fallback to simple condition if position tracking not active
                atr_condition = current['high'] >= short_stop
                
            if atr_condition:
                short_exit_confirmations += 1
                if not short_exit_reason:
                    short_exit_reason = "Stop Level Hit"
            
            # Momentum score condition
            momentum_condition = short_momentum < thresholds['momentum_exit_threshold']
            if momentum_condition:
                short_exit_confirmations += 1
                if not short_exit_reason:
                    short_exit_reason = "Momentum Weakening"
                    
            # -- NEW: Check for momentum exhaustion --
            if self.position_tracker.position_type == 'short' and self.profit_strategy.momentum_exhaustion(data.iloc[:i+1], 'short'):
                short_exit_confirmations += 1
                if not short_exit_reason:
                    short_exit_reason = "Momentum Exhaustion"
                    
            # -- NEW: Check for consecutive close confirmation --
            if self.position_tracker.position_type == 'short' and self.profit_strategy.consecutive_close_confirm(data.iloc[:i+1]):
                short_exit_confirmations += 1
                if not short_exit_reason:
                    short_exit_reason = "Consecutive Close Confirmation"
            
            # Generate short exit signal if enough confirmations and not already exiting due to profit target
            exit_short = short_exit_confirmations >= thresholds['confirmation_count'] and not signals.at[current_idx, 'profit_target_hit']
            signals.at[current_idx, 'exit_sell'] = signals.at[current_idx, 'exit_sell'] or exit_short
            if exit_short and not signals.at[current_idx, 'exit_reason']:
                signals.at[current_idx, 'exit_reason'] = short_exit_reason
            
            signals.at[current_idx, 'effective_move'] = effective_move
        
        return signals

    def manage_positions(self, data, entry_signals, exit_signals):
        """
        Manage positions with proper entry and exit sequencing
        
        Args:
            data (pd.DataFrame): OHLCV price data with indicators
            entry_signals (pd.DataFrame): DataFrame with buy/sell entry signals
            exit_signals (pd.DataFrame): DataFrame with exit signals
            
        Returns:
            pd.Series: Series with position labels
        """
        positions = pd.Series(index=data.index, dtype='object')
        current_pos = None
        entry_price = None
        entry_time = None
        pending_exit = False
        scaled_out_amount = 0.0  # Track scaled-out percentage
        
        for i, idx in enumerate(data.index):
            current_price = data.loc[idx, 'close']
            
            # Check for exit signals first (exit before entry within the same bar)
            if current_pos:
                if (current_pos == 'long' and exit_signals.at[idx, 'exit_buy']) or \
                   (current_pos == 'short' and exit_signals.at[idx, 'exit_sell']):
                    # Record exit
                    positions[idx] = f"exit_{current_pos}"
                    
                    # Reset position tracking
                    current_pos = None
                    entry_price = None
                    entry_time = None
                    pending_exit = False
                    scaled_out_amount = 0.0
                    
                    # Reset profit strategy
                    self.profit_strategy.reset()
                    
                # Check for partial scaling out based on profit targets
                elif exit_signals.at[idx, 'scale_out_pct'] > 0 and exit_signals.at[idx, 'scale_out_pct'] < 100:
                    # Record partial exit
                    positions[idx] = f"scale_out_{current_pos}_{exit_signals.at[idx, 'scale_out_pct']:.0f}"
                    scaled_out_amount += exit_signals.at[idx, 'scale_out_pct']
                
                # Skip entry signals if we already have a position
                continue
            
            # Only process entry signals if we don't have an active position and not pending exit
            if not current_pos and not pending_exit:
                # Check for buy signal
                if entry_signals.at[idx, 'buy_signal']:
                    current_pos = 'long'
                    entry_price = current_price
                    entry_time = idx
                    positions[idx] = 'entry_long'
                    scaled_out_amount = 0.0
                    
                    # Reset and set position for tracking
                    self.position_tracker.set_position('long', entry_price, entry_time)
                    self.momentum_analyzer.set_position('long', entry_price, entry_time)
                    self.profit_strategy.reset()
                
                # Check for sell signal
                elif entry_signals.at[idx, 'sell_signal']:
                    current_pos = 'short'
                    entry_price = current_price
                    entry_time = idx
                    positions[idx] = 'entry_short'
                    scaled_out_amount = 0.0
                    
                    # Reset and set position for tracking
                    self.position_tracker.set_position('short', entry_price, entry_time)
                    self.momentum_analyzer.set_position('short', entry_price, entry_time)
                    self.profit_strategy.reset()
            
            # Special case: check for conflicting signals that would cause overlap
            # For example: we have a long position and see a sell signal without an exit signal
            elif current_pos and not pending_exit:
                conflict = False
                
                if current_pos == 'long' and entry_signals.at[idx, 'sell_signal']:
                    conflict = True
                elif current_pos == 'short' and entry_signals.at[idx, 'buy_signal']:
                    conflict = True
                
                if conflict:
                    # Mark as pending exit to enforce exit on next bar
                    pending_exit = True
                    
                    # Record in positions that we're waiting for exit
                    positions[idx] = f"pending_exit_{current_pos}"

        return positions

def get_spy_atr_filter_config() -> Dict[str, Any]:
    """
    SPY-optimized ATR filter configuration based on research
    
    Returns:
        Dict[str, Any]: Optimized configuration parameters for SPY
    """
    return {
        'atr_period': 14,  # More responsive to recent volatility
        'min_holding_period': pd.Timedelta('12min'),  # Matches SPY's 5m bar rhythm
        'base_k': 1.5,  # Base K multiplier from backtesting research
        'vix_sensitivity': 0.25,  # From research paper
        'session_adjustments': {
            'pre_open': {
                'k_multiplier': 0.8,
                'min_move': 0.0008,
                'min_holding': '10min'
            },
            'regular': {
                'k_multiplier': 1.5,
                'min_move': 0.0012,
                'min_holding': '12min'
            },
            'power_hour': {
                'k_multiplier': 2.2,
                'min_move': 0.0015,
                'min_holding': '5min'
            }
        },
        # Position-specific adjustments with new profit-taking parameters
        'position_specific': {
            'long': {
                'exit_score_threshold': 45,  # Higher threshold for long exits (more conservative)
                'profit_taking': True,  # Enable automatic profit taking
                'profit_targets': {
                    'conservative': 0.5,  # 0.5% profit target (conservative)
                    'moderate': 0.8,      # 0.8% profit target (moderate)
                    'aggressive': 1.2     # 1.2% profit target (aggressive)
                },
                'scale_out': {
                    'conservative': 50,   # Take 50% off at conservative target
                    'moderate': 25,       # Take 25% off at moderate target
                    'aggressive': 25      # Take final 25% off at aggressive target
                }
            },
            'short': {
                'exit_score_threshold': 40,  # Lower threshold for short exits (more sensitive)
                'profit_taking': True,
                'profit_targets': {
                    'conservative': 0.4,  # 0.4% profit target (conservative)
                    'moderate': 0.7,      # 0.7% profit target (moderate)
                    'aggressive': 1.0     # 1.0% profit target (aggressive)
                },
                'scale_out': {
                    'conservative': 50,   # Take 50% off at conservative target
                    'moderate': 25,       # Take 25% off at moderate target
                    'aggressive': 25      # Take final 25% off at aggressive target
                }
            }
        }
    }

class EnhancedExitStrategy:
    def __init__(self, config=None, symbol=None):
        """
        Initialize the enhanced exit strategy with configuration
        
        Args:
            config (dict, optional): Strategy configuration
            symbol (str, optional): Trading symbol for specific optimizations
        """
        self.position_manager = ImprovedExitSignalGenerator()
        
        # Use SPY-specific config if symbol is SPY
        if symbol == 'SPY' or symbol == 'spy':
            self.config = get_spy_atr_filter_config()
        else:
            self.config = config or IMPROVED_EXIT_CONFIG
            
        # Set min holding period based on config
        if isinstance(self.config.get('min_holding_period'), str):
            self.min_holding_period = pd.Timedelta(self.config.get('min_holding_period', '15min'))
        else:
            self.min_holding_period = pd.Timedelta(minutes=self.config.get('min_holding_minutes', 15))
        
    def execute_strategy(self, data, entry_signals, symbol=None):
        """
        Integrated strategy execution with sophisticated exit logic
        
        Args:
            data (pd.DataFrame): OHLCV price data
            entry_signals (pd.DataFrame): DataFrame with buy/sell entry signals
            symbol (str, optional): Trading symbol for specific optimizations
            
        Returns:
            dict: Dictionary with processed signals and positions
        """
        # Add VIX data if not present
        if 'vix' not in data.columns:
            vix_value = get_vix_data(data)
            data['vix'] = vix_value
            
        # Generate exit signals with improved logic
        exit_signals = self.position_manager.generate_exit_signals(data, entry_signals)
        
        # Apply volume confirmation filter to entries
        validated_entries = entry_signals.copy()
        
        # Only apply volume filter if volume column exists
        if 'volume' in data.columns and 'vol_ma' in data.columns:
            validated_entries['buy_signal'] = validated_entries['buy_signal'] & (data['volume'] > data['vol_ma'])
            validated_entries['sell_signal'] = validated_entries['sell_signal'] & (data['volume'] > data['vol_ma'])
        
        # Apply custom SPY filter for better signal quality if symbol is SPY
        if symbol == 'SPY' or symbol == 'spy':
            validated_entries = self._apply_spy_specific_filters(data, validated_entries)
        
        # Execute position management
        positions = self.position_manager.manage_positions(data, validated_entries, exit_signals)
        
        # Post-process positions with minimum holding period
        filtered_positions = self._apply_holding_period_filter(data, positions)
        
        # Convert positions to filtered signals
        filtered_signals = self._convert_positions_to_signals(entry_signals, exit_signals, filtered_positions)
        
        # Add effective move tracking to filtered signals
        if 'effective_move' in exit_signals.columns:
            filtered_signals['effective_move'] = exit_signals['effective_move']
        
        return {
            'exit_signals': exit_signals,
            'validated_entries': validated_entries,
            'positions': filtered_positions,
            'filtered_signals': filtered_signals
        }
        
    def _apply_spy_specific_filters(self, data, signals):
        """
        Apply SPY-specific signal quality filters
        
        Args:
            data (pd.DataFrame): OHLCV price data
            signals (pd.DataFrame): Trading signals
            
        Returns:
            pd.DataFrame: Filtered signals
        """
        filtered = signals.copy()
        
        # Get current market session
        session_type = 'regular'
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
            try:
                current_time = data.index[-1].time()
                
                # Convert to Eastern time if timezone info is available
                if hasattr(data.index[-1], 'tzinfo') and data.index[-1].tzinfo is not None:
                    eastern = pytz.timezone('US/Eastern')
                    current_time = data.index[-1].astimezone(eastern).time()
                    
                # Market open (9:30-10:30)
                if time(9, 30) <= current_time < time(10, 30):
                    session_type = 'pre_open'
                # Power hour (14:30-16:00)
                elif time(14, 30) <= current_time <= time(16, 0):
                    session_type = 'power_hour'
            except:
                pass
        
        # Get VIX-based adjustments
        vix_value = data['vix'].iloc[-1] if 'vix' in data.columns else get_vix_data(data)
        
        # Get session config
        session_config = self.config['session_adjustments'][session_type]
        
        # Filter out noise during high volatility periods
        if vix_value > 25:
            # More stringent filtering in high volatility
            # Require stronger signals (higher volume confirmation)
            if 'volume' in data.columns and 'vol_ma' in data.columns:
                vol_ratio = 1.2  # Require 20% above average volume in high volatility
                filtered['buy_signal'] = filtered['buy_signal'] & (data['volume'] > vol_ratio * data['vol_ma'])
                filtered['sell_signal'] = filtered['sell_signal'] & (data['volume'] > vol_ratio * data['vol_ma'])
        
        # Apply additional pre-open filters
        if session_type == 'pre_open':
            # More conservative during pre-open: require additional confirmation
            if 'adx' in data.columns:
                # Require stronger trend confirmation in pre-open session
                filtered['buy_signal'] = filtered['buy_signal'] & (data['adx'] > 20)
                filtered['sell_signal'] = filtered['sell_signal'] & (data['adx'] > 20)
                
        # Power hour specific adjustments
        elif session_type == 'power_hour':
            # More aggressive signals in power hour
            # No additional filtering needed, but may override certain filters
            pass
            
        return filtered

    def _apply_holding_period_filter(self, data, positions):
        """
        Enforce minimum holding period with exceptions for strong exit signals
        and proper handling of pending exits
        
        Args:
            data (pd.DataFrame): OHLCV price data
            positions (pd.Series): Position states
            
        Returns:
            pd.Series: Filtered positions
        """
        filtered_positions = positions.copy()
        last_entry_time = None
        entry_type = None
        in_pending_exit = False
        
        for idx in data.index:
            pos = positions[idx]
            
            if not isinstance(pos, str):
                continue
                
            # Track entry positions
            if 'entry' in pos:
                last_entry_time = idx
                entry_type = 'long' if 'long' in pos else 'short'
                in_pending_exit = False
            
            # Track pending exits
            elif 'pending_exit' in pos:
                in_pending_exit = True
            
            # Handle exit positions
            elif 'exit' in pos:
                # Check if we have a strong exit reason that overrides min holding period
                strong_exit = False
                
                # Pending exits, stop level or momentum exits override holding period
                if in_pending_exit or 'stop_level' in pos or 'momentum' in pos or 'threshold_reached' in pos:
                    strong_exit = True
                
                # Apply minimum holding period unless strong exit reason
                if last_entry_time and (idx - last_entry_time) < self.min_holding_period and not strong_exit:
                    filtered_positions[idx] = None
                else:
                    last_entry_time = None
                    entry_type = None
                    in_pending_exit = False
        
        return filtered_positions
    
    def _convert_positions_to_signals(self, entry_signals, exit_signals, positions):
        """
        Convert position states back to signals format with exit reasons
        and handling of pending exits
        
        Args:
            entry_signals (pd.DataFrame): Entry signals
            exit_signals (pd.DataFrame): Exit signals
            positions (pd.Series): Position states
            
        Returns:
            pd.DataFrame: Filtered signals
        """
        filtered = entry_signals.copy()
        
        # Initialize filtered signal columns
        if 'filtered_buy_signal' not in filtered.columns:
            filtered['filtered_buy_signal'] = False
        if 'filtered_sell_signal' not in filtered.columns:
            filtered['filtered_sell_signal'] = False
        if 'pending_exit' not in filtered.columns:
            filtered['pending_exit'] = False
        
        # Copy exit signals
        filtered['exit_buy'] = exit_signals['exit_buy']
        filtered['exit_sell'] = exit_signals['exit_sell']
        filtered['exit_reason'] = exit_signals['exit_reason']
        filtered['atr_stop'] = exit_signals['atr_stop']
        
        # Set filtered signals based on positions
        for idx, pos in positions.items():
            if not pos:  # Skip empty positions
                continue
                
            if not isinstance(pos, str):
                continue
                
            if pos == 'entry_long':
                filtered.at[idx, 'filtered_buy_signal'] = True
                filtered.at[idx, 'filtered_sell_signal'] = False
                filtered.at[idx, 'pending_exit'] = False
                
            elif pos == 'entry_short':
                filtered.at[idx, 'filtered_buy_signal'] = False
                filtered.at[idx, 'filtered_sell_signal'] = True
                filtered.at[idx, 'pending_exit'] = False
                
            elif 'exit_long' in pos:
                filtered.at[idx, 'exit_buy'] = True
                filtered.at[idx, 'filtered_buy_signal'] = False
                filtered.at[idx, 'pending_exit'] = False
                
                # Extract exit reason if available in position string
                if '_' in pos and len(pos.split('_')) > 2:
                    reason_parts = pos.split('_')[2:]
                    filtered.at[idx, 'exit_reason'] = ' '.join(word.capitalize() for word in reason_parts)
                    
            elif 'exit_short' in pos:
                filtered.at[idx, 'exit_sell'] = True
                filtered.at[idx, 'filtered_sell_signal'] = False
                filtered.at[idx, 'pending_exit'] = False
                
                # Extract exit reason if available in position string
                if '_' in pos and len(pos.split('_')) > 2:
                    reason_parts = pos.split('_')[2:]
                    filtered.at[idx, 'exit_reason'] = ' '.join(word.capitalize() for word in reason_parts)
                    
            elif 'pending_exit' in pos:
                filtered.at[idx, 'pending_exit'] = True
                
                # Mark the appropriate exit column based on position type
                if 'long' in pos:
                    filtered.at[idx, 'exit_buy'] = True
                    filtered.at[idx, 'exit_reason'] = 'Position Conflict'
                elif 'short' in pos:
                    filtered.at[idx, 'exit_sell'] = True
                    filtered.at[idx, 'exit_reason'] = 'Position Conflict'
        
        return filtered

# Improved configuration for exit strategy
IMPROVED_EXIT_CONFIG = {
    'rsi_overbought': 70,  # Now dynamically adjusted
    'rsi_oversold': 30,    # Now dynamically adjusted
    'atr_multiplier': 2.2, # Now dynamic based on volatility
    'volume_threshold': 0.8,
    'min_holding_minutes': 12,  # Matches SPY's 5m bar rhythm
    'confirmation_count': 2,  # Require at least 2 signals
    'vix_sensitivity': 0.25,  # From research paper
    'session_adjustments': {
        'pre_open': (0.8, '10min'),
        'power_hour': (2.2, '5min')
    }
}

# Keep the old configuration for backward compatibility
SPY_EXIT_CONFIG = {
    'atr_period': 14,  # More responsive to recent volatility
    'min_holding_period': pd.Timedelta('12min'),  # Matches SPY's 5m bar rhythm
    'base_k': 1.5,  # Base K multiplier from backtesting research
    'vix_sensitivity': 0.25,  # From research paper
    'session_adjustments': {
        'pre_open': {
            'k_multiplier': 0.8,
            'min_move': 0.0008,
            'min_holding': '10min'
        },
        'regular': {
            'k_multiplier': 1.5,
            'min_move': 0.0012,
            'min_holding': '12min'
        },
        'power_hour': {
            'k_multiplier': 2.2,
            'min_move': 0.0015,
            'min_holding': '5min'
        }
    },
    # Position-specific adjustments with new profit-taking parameters
    'position_specific': {
        'long': {
            'exit_score_threshold': 45,  # Higher threshold for long exits (more conservative)
            'profit_taking': True,  # Enable automatic profit taking
            'profit_targets': {
                'conservative': 0.5,  # 0.5% profit target (conservative)
                'moderate': 0.8,      # 0.8% profit target (moderate)
                'aggressive': 1.2     # 1.2% profit target (aggressive)
            },
            'scale_out': {
                'conservative': 50,   # Take 50% off at conservative target
                'moderate': 25,       # Take 25% off at moderate target
                'aggressive': 25      # Take final 25% off at aggressive target
            }
        },
        'short': {
            'exit_score_threshold': 40,  # Lower threshold for short exits (more sensitive)
            'profit_taking': True,
            'profit_targets': {
                'conservative': 0.4,  # 0.4% profit target (conservative)
                'moderate': 0.7,      # 0.7% profit target (moderate)
                'aggressive': 1.0     # 1.0% profit target (aggressive)
            },
            'scale_out': {
                'conservative': 50,   # Take 50% off at conservative target
                'moderate': 25,       # Take 25% off at moderate target
                'aggressive': 25      # Take final 25% off at aggressive target
            }
        }
    }
}

def apply_exit_strategy(data, signals, symbol=None):
    """
    Apply the enhanced exit strategy to signals with symbol-specific optimizations
    
    Args:
        data (pd.DataFrame): OHLCV price data
        signals (pd.DataFrame): Trading signals
        symbol (str, optional): Trading symbol for specific optimizations
    
    Returns:
        pd.DataFrame: Updated signals with exit signals added
    """
    try:
        # Initialize exit strategy based on symbol
        strategy = EnhancedExitStrategy(symbol=symbol)
        
        # Execute strategy with symbol information
        result = strategy.execute_strategy(data, signals, symbol=symbol)
        
        return result['filtered_signals']
    except Exception as e:
        print(f"Error applying exit strategy: {str(e)}")
        traceback.print_exc()
        return signals  # Return original signals if error occurs

# Legacy support - keep the old ExitSignalGenerator available for backward compatibility
# The rest of the original code remains unchanged below 

class ExitSignalGenerator:
    def __init__(self):
        self.current_position = None
        self.entry_price = None
        self.entry_time = None

    def generate_exit_signals(self, data, entry_signals):
        """
        Generate dedicated exit signals for both long and short positions
        with integrated momentum, volume, and volatility checks
        
        Args:
            data (pd.DataFrame): OHLCV price data
            entry_signals (pd.DataFrame): DataFrame with buy/sell entry signals
            
        Returns:
            pd.DataFrame: DataFrame with exit signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['exit_buy'] = False
        signals['exit_sell'] = False
        
        # Calculate technical indicators
        if 'rsi' not in data.columns:
            data['rsi'] = calculate_rsi(data)
        
        # Calculate MACD if not already present
        if 'macd' not in data.columns:
            macd, signal, macd_hist = calculate_macd(data)
            data['macd'] = macd
            data['macd_signal'] = signal
            data['macd_hist'] = macd_hist
        else:
            macd_hist = data['macd_hist']
        
        # Ensure ATR is calculated
        if 'atr' not in data.columns:
            data['atr'] = calculate_atr(data)
        
        # Calculate volume MA for volume confirmation
        data['vol_ma'] = data['volume'].rolling(20).mean()

        # Momentum weakening detection
        for i in range(1, len(data)):
            current = data.iloc[i]
            prev = data.iloc[i-1]
            current_idx = data.index[i]

            # Long exit conditions
            exit_long = False
            try:
                exit_long = (
                    (current['rsi'] < 65 and prev['rsi'] >= 65) or          # RSI dropping from overbought
                    (current['macd_hist'] < 0 and prev['macd_hist'] >= 0) or # MACD histogram turning negative
                    (current['volume'] < 0.8 * current['vol_ma']) or         # Volume dropping
                    (current['close'] < current['high'] - 1.5 * current['atr'])  # ATR-based trailing stop
                )
            except Exception as e:
                print(f"Error calculating long exit at {current_idx}: {str(e)}")
                
            # Short exit conditions
            exit_short = False
            try:
                exit_short = (
                    (current['rsi'] > 35 and prev['rsi'] <= 35) or          # RSI rising from oversold
                    (current['macd_hist'] > 0 and prev['macd_hist'] <= 0) or # MACD histogram turning positive
                    (current['volume'] < 0.8 * current['vol_ma']) or         # Volume confirmation
                    (current['close'] > current['low'] + 1.5 * current['atr'])  # ATR-based trailing stop
                )
            except Exception as e:
                print(f"Error calculating short exit at {current_idx}: {str(e)}")

            signals.at[current_idx, 'exit_buy'] = exit_long
            signals.at[current_idx, 'exit_sell'] = exit_short

        return signals

    def manage_positions(self, data, entry_signals, exit_signals):
        """
        Position management with strict entry/exit sequencing
        Implements state machine to enforce exit-before-entry
        
        Args:
            data (pd.DataFrame): OHLCV price data
            entry_signals (pd.DataFrame): DataFrame with buy/sell entry signals
            exit_signals (pd.DataFrame): DataFrame with exit signals
            
        Returns:
            pd.Series: Series with position states (entry_long, exit_long, etc.)
        """
        positions = pd.Series(index=data.index, dtype=object)
        current_pos = None
        
        for idx in data.index:
            # Exit existing position first
            if current_pos:
                if (current_pos == 'long' and exit_signals.at[idx, 'exit_buy']) or \
                   (current_pos == 'short' and exit_signals.at[idx, 'exit_sell']):
                    positions[idx] = 'exit_' + current_pos
                    current_pos = None
                    continue  # Skip to next bar after exit

            # Enter new position only if flat
            if not current_pos:
                if entry_signals.at[idx, 'buy_signal']:
                    current_pos = 'long'
                    positions[idx] = 'entry_long'
                elif entry_signals.at[idx, 'sell_signal']:
                    current_pos = 'short'
                    positions[idx] = 'entry_short'

        return positions 

def dynamic_k_multiplier(vix: float) -> float:
    """
    VIX-adjusted ATR multiplier for more accurate volatility-based stops
    
    Args:
        vix (float): VIX volatility index value
        
    Returns:
        float: Adjusted K multiplier for ATR calculation
    """
    if vix < 15: 
        return 1.3  # Low volatility
    elif 15 <= vix <= 25: 
        return 1.8  # Normal volatility
    else: 
        return 2.5  # High volatility

def calculate_effective_move(entry_price: float, current_price: float, 
                           highs: pd.Series, lows: pd.Series) -> float:
    """
    Calculate maximum favorable excursion to get true price path movement
    
    Args:
        entry_price (float): Position entry price
        current_price (float): Current price
        highs (pd.Series): Series of high prices since entry
        lows (pd.Series): Series of low prices since entry
        
    Returns:
        float: Maximum favorable excursion value
    """
    if entry_price < current_price:  # Long position
        return max(highs) - entry_price
    else:  # Short position
        return entry_price - min(lows)

def session_adjustments(data: pd.DataFrame) -> dict:
    """
    Time-based volatility adjustments for different market sessions
    
    Args:
        data (pd.DataFrame): Price data with timestamps as index
        
    Returns:
        dict: Parameters adjusted for current market session
    """
    # Default values
    result = {
        'k_multiplier': 1.5,
        'min_move': 0.0012
    }
    
    # Early check to avoid errors
    if not isinstance(data.index, pd.DatetimeIndex) or len(data) == 0:
        return result
        
    # Try to get current time
    try:
        current_time = data.index[-1].time()
        
        # Convert to Eastern time if timezone info is available
        if hasattr(data.index[-1], 'tzinfo') and data.index[-1].tzinfo is not None:
            eastern = pytz.timezone('US/Eastern')
            current_time = data.index[-1].astimezone(eastern).time()
            
        # Market open (9:30-10:30): Higher volatility, lower threshold
        if time(9, 30) <= current_time < time(10, 30):
            result['k_multiplier'] = 1.2
            result['min_move'] = 0.0008
            
        # Mid-day (10:30-14:30): Normal parameters
        elif time(10, 30) <= current_time < time(14, 30):
            result['k_multiplier'] = 1.5
            result['min_move'] = 0.0012
            
        # Power hour (14:30-16:00): Higher volatility, higher threshold
        elif time(14, 30) <= current_time <= time(16, 0):
            result['k_multiplier'] = 2.0
            result['min_move'] = 0.0015
    except:
        # In case of any error, return default values
        pass
        
    return result 

def get_vix_data(data=None, fallback_value=15.0):
    """
    Get the latest VIX value or use a fallback
    
    Args:
        data: DataFrame that might contain VIX data
        fallback_value: Default VIX value if no data available
        
    Returns:
        float: VIX value
    """
    if data is not None and 'vix' in data.columns:
        return data['vix'].iloc[-1]
    
    try:
        # Try to get VIX from Yahoo Finance
        import yfinance as yf
        vix_data = yf.download("^VIX", period="1d")
        if not vix_data.empty:
            return vix_data['Close'].iloc[-1]
    except:
        pass
    
    # Return fallback value if no VIX data available
    return fallback_value 