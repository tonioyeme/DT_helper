"""
SPY-specific day trading strategy.
Implements optimized strategies for intraday SPY trading based on the SPY configuration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pytz
from datetime import datetime, time

from app.signals.generator import analyze_single_day
from app.indicators import (
    calculate_ema,
    calculate_vwap, 
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_atr
)

try:
    from app.config.instruments.spy import SPY_CONFIG
    print("Successfully loaded SPY configuration in spy_strategy.py")
except ImportError:
    print("SPY configuration import failed in spy_strategy.py. Using built-in defaults.")
    # Default configuration if the SPY_CONFIG is not available
    SPY_CONFIG = {
        'indicators': {
            'ema': {'fast_period': 5, 'slow_period': 13},
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'vwap': {'reset_period': 'day'},
            'bollinger': {'period': 20, 'std_dev': 2.0}
        },
        'strategies': {
            'orb': {'minutes': 5, 'confirmation_candles': 1},
            'vwap_bollinger': {'enabled': True},
            'ema_vwap': {'enabled': True}
        },
        'signals': {
            'threshold': 0.6,
            'confirmation_required': False
        }
    }

def analyze_spy_day_trading(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze SPY intraday data using specialized SPY day trading configuration.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Dictionary with analysis results and signals
    """
    print(f"Starting SPY day trading analysis with {len(data)} data points")
    
    # Make a copy to avoid modifying original data
    df = data.copy()
    
    # Ensure we're analyzing a single day of data
    if isinstance(df.index[0], pd.Timestamp):
        # Get Eastern time zone for market hours
        eastern_tz = pytz.timezone('US/Eastern')
        
        # Convert to Eastern time if timezone info exists
        if df.index[0].tzinfo is not None:
            first_date = df.index[0].tz_convert(eastern_tz).date()
        else:
            first_date = df.index[0].date()
        
        # Filter to just the current day
        if df.index[-1].date() != first_date:
            df = df[df.index.date == first_date]
    
    # Ensure we have enough data points
    if len(df) < 10:
        return {
            "success": False,
            "error": "Insufficient data for SPY analysis",
            "signals": pd.DataFrame()
        }
    
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=df.index)
    signals["buy_signal"] = False
    signals["sell_signal"] = False
    signals["buy_score"] = 0.0
    signals["sell_score"] = 0.0
    signals["signal_strength"] = 0.0
    signals["signal_price"] = df["close"]
    signals["target_price"] = None
    signals["stop_loss"] = None
    
    # Get indicator config
    indicator_config = SPY_CONFIG['indicators']
    strategy_config = SPY_CONFIG['strategies']
    signal_config = SPY_CONFIG['signals']
    
    # Calculate indicators with SPY-specific settings
    try:
        # 1. EMA with SPY-specific periods
        ema_fast_period = indicator_config['ema'].get('fast_period', 5)
        ema_slow_period = indicator_config['ema'].get('slow_period', 13)
        
        ema_fast = calculate_ema(df, period=ema_fast_period)
        ema_slow = calculate_ema(df, period=ema_slow_period)
        
        df['ema_fast'] = ema_fast
        df['ema_slow'] = ema_slow
        
        # 2. VWAP calculation (reset daily)
        df['vwap'] = calculate_vwap(df)
        
        # 3. RSI with SPY settings
        rsi_period = indicator_config['rsi'].get('period', 14)
        rsi_overbought = indicator_config['rsi'].get('overbought', 70)
        rsi_oversold = indicator_config['rsi'].get('oversold', 30)
        
        df['rsi'] = calculate_rsi(df, period=rsi_period)
        
        # Apply adaptive RSI thresholds if configured
        if indicator_config['rsi'].get('adaptive', False):
            # Calculate volatility to adjust RSI thresholds
            atr = calculate_atr(df, period=14)
            atr_percent = atr / df['close']
            
            # Adjust thresholds based on volatility
            median_atr_pct = atr_percent.median()
            recent_atr_pct = atr_percent.iloc[-1]
            
            volatility_ratio = recent_atr_pct / median_atr_pct if median_atr_pct > 0 else 1.0
            
            # Adjust thresholds - higher volatility means wider thresholds
            if volatility_ratio > 1.5:
                rsi_overbought = 75
                rsi_oversold = 25
            elif volatility_ratio < 0.7:
                rsi_overbought = 65
                rsi_oversold = 35
        
        # 4. Bollinger Bands
        bb_period = indicator_config['bollinger'].get('period', 20)
        bb_std_dev = indicator_config['bollinger'].get('std_dev', 2.0)
        
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, period=bb_period, std_dev=bb_std_dev)
        
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # 5. ATR for stop loss and target calculation
        atr_period = indicator_config['atr'].get('period', 14)
        df['atr'] = calculate_atr(df, period=atr_period)
        
        # Calculate session phase (morning, midday, closing)
        if isinstance(df.index[0], pd.Timestamp) and df.index[0].tzinfo is not None:
            eastern_time = df.index.tz_convert(eastern_tz)
            
            # Define market session time ranges
            market_open = time(9, 30)
            morning_end = time(11, 30)
            midday_end = time(14, 30)
            market_close = time(16, 0)
            
            # Create session markers
            df['session'] = 'unknown'
            
            for i, idx in enumerate(df.index):
                et_time = eastern_time[i].time()
                
                if et_time < market_open:
                    df.loc[idx, 'session'] = 'pre_market'
                elif et_time < morning_end:
                    df.loc[idx, 'session'] = 'morning'
                elif et_time < midday_end:
                    df.loc[idx, 'session'] = 'midday'
                elif et_time <= market_close:
                    df.loc[idx, 'session'] = 'closing'
                else:
                    df.loc[idx, 'session'] = 'post_market'
            
        # Implement SPY-specific strategies
        
        # Strategy 1: Opening Range Breakout (ORB)
        if strategy_config['orb'].get('enabled', True):
            # Get opening range minutes 
            orb_minutes = strategy_config['orb'].get('minutes', 5)
            
            # Calculate opening range
            eastern_tz = pytz.timezone('US/Eastern')
            market_open_time = time(9, 30)
            
            if isinstance(df.index[0], pd.Timestamp) and df.index[0].tzinfo is not None:
                # Find market open in the data
                df_idx_et = df.index.tz_convert(eastern_tz)
                market_open_mask = [idx.time() >= market_open_time for idx in df_idx_et]
                
                if any(market_open_mask):
                    # Find first candle after market open
                    market_open_idx = df.index[market_open_mask][0]
                    # Use Timedelta properly for timestamp arithmetic
                    if len(df.index[df.index > market_open_idx]) > 0:
                        # Find indexes after market open + opening range minutes
                        opening_range_end_time = market_open_idx + pd.Timedelta(minutes=orb_minutes)
                        opening_range_end_idx = df.index[df.index > opening_range_end_time]
                        
                        if len(opening_range_end_idx) > 0:
                            opening_range_end = opening_range_end_idx[0]
                            opening_range = df[(df.index >= market_open_idx) & (df.index < opening_range_end)]
                            
                            if not opening_range.empty:
                                or_high = opening_range['high'].max()
                                or_low = opening_range['low'].min()
                                
                                # After opening range, check for breakouts
                                for idx in df.index[df.index >= opening_range_end]:
                                    # ORB Upside breakout
                                    if df.loc[idx, 'high'] > or_high:
                                        signals.loc[idx, 'buy_score'] += strategy_config['orb'].get('weight', 1.8)
                                        if idx > 0 and signals.loc[idx, 'buy_score'] > signal_config.get('threshold', 0.6):
                                            signals.loc[idx, 'buy_signal'] = True
                                    
                                    # ORB Downside breakout
                                    if df.loc[idx, 'low'] < or_low:
                                        signals.loc[idx, 'sell_score'] += strategy_config['orb'].get('weight', 1.8)
                                        if idx > 0 and signals.loc[idx, 'sell_score'] > signal_config.get('threshold', 0.6):
                                            signals.loc[idx, 'sell_signal'] = True
        
        # Strategy 2: VWAP Bollinger Band Strategy
        if strategy_config.get('vwap_bollinger', {}).get('enabled', True):
            strategy_weight = strategy_config.get('vwap_bollinger', {}).get('weight', 1.5)
            
            for i in range(1, len(df)):
                idx = df.index[i]
                prev_idx = df.index[i-1]
                
                # VWAP crossing up through Bollinger lower band (buy signal)
                if (df.loc[prev_idx, 'vwap'] <= df.loc[prev_idx, 'bb_lower'] and
                    df.loc[idx, 'vwap'] > df.loc[idx, 'bb_lower']):
                    signals.loc[idx, 'buy_score'] += strategy_weight
                    if signals.loc[idx, 'buy_score'] > signal_config.get('threshold', 0.6):
                        signals.loc[idx, 'buy_signal'] = True
                
                # VWAP crossing down through Bollinger upper band (sell signal)
                if (df.loc[prev_idx, 'vwap'] >= df.loc[prev_idx, 'bb_upper'] and
                    df.loc[idx, 'vwap'] < df.loc[idx, 'bb_upper']):
                    signals.loc[idx, 'sell_score'] += strategy_weight
                    if signals.loc[idx, 'sell_score'] > signal_config.get('threshold', 0.6):
                        signals.loc[idx, 'sell_signal'] = True
                        
                # Price bouncing off VWAP from below
                if (df.loc[prev_idx, 'low'] <= df.loc[prev_idx, 'vwap'] and
                    df.loc[idx, 'close'] > df.loc[idx, 'vwap'] and
                    df.loc[idx, 'close'] > df.loc[idx, 'open']):  # Green candle
                    signals.loc[idx, 'buy_score'] += strategy_weight * 0.8
                    if signals.loc[idx, 'buy_score'] > signal_config.get('threshold', 0.6):
                        signals.loc[idx, 'buy_signal'] = True
                
                # Price bouncing off VWAP from above
                if (df.loc[prev_idx, 'high'] >= df.loc[prev_idx, 'vwap'] and
                    df.loc[idx, 'close'] < df.loc[idx, 'vwap'] and
                    df.loc[idx, 'close'] < df.loc[idx, 'open']):  # Red candle
                    signals.loc[idx, 'sell_score'] += strategy_weight * 0.8
                    if signals.loc[idx, 'sell_score'] > signal_config.get('threshold', 0.6):
                        signals.loc[idx, 'sell_signal'] = True
        
        # Strategy 3: EMA-VWAP Strategy
        if strategy_config.get('ema_vwap', {}).get('enabled', True):
            strategy_weight = strategy_config.get('ema_vwap', {}).get('weight', 1.3)
            
            for i in range(1, len(df)):
                idx = df.index[i]
                prev_idx = df.index[i-1]
                
                # Fast EMA crossing up through VWAP (buy signal)
                if (df.loc[prev_idx, 'ema_fast'] <= df.loc[prev_idx, 'vwap'] and
                    df.loc[idx, 'ema_fast'] > df.loc[idx, 'vwap']):
                    signals.loc[idx, 'buy_score'] += strategy_weight
                    if signals.loc[idx, 'buy_score'] > signal_config.get('threshold', 0.6):
                        signals.loc[idx, 'buy_signal'] = True
                
                # Fast EMA crossing down through VWAP (sell signal)
                if (df.loc[prev_idx, 'ema_fast'] >= df.loc[prev_idx, 'vwap'] and
                    df.loc[idx, 'ema_fast'] < df.loc[idx, 'vwap']):
                    signals.loc[idx, 'sell_score'] += strategy_weight
                    if signals.loc[idx, 'sell_score'] > signal_config.get('threshold', 0.6):
                        signals.loc[idx, 'sell_signal'] = True
                
                # Bull trend confirmation: Fast EMA > Slow EMA > VWAP
                if (df.loc[idx, 'ema_fast'] > df.loc[idx, 'ema_slow'] > df.loc[idx, 'vwap']):
                    signals.loc[idx, 'buy_score'] += strategy_weight * 0.5
                    if signals.loc[idx, 'buy_score'] > signal_config.get('threshold', 0.6):
                        signals.loc[idx, 'buy_signal'] = True
                
                # Bear trend confirmation: Fast EMA < Slow EMA < VWAP
                if (df.loc[idx, 'ema_fast'] < df.loc[idx, 'ema_slow'] < df.loc[idx, 'vwap']):
                    signals.loc[idx, 'sell_score'] += strategy_weight * 0.5
                    if signals.loc[idx, 'sell_score'] > signal_config.get('threshold', 0.6):
                        signals.loc[idx, 'sell_signal'] = True
        
        # Strategy 4: Volume-Price Action
        if strategy_config.get('volume_price', {}).get('enabled', True):
            strategy_weight = strategy_config.get('volume_price', {}).get('weight', 1.2)
            volume_threshold = indicator_config.get('volume', {}).get('relative_volume_threshold', 1.5)
            
            # Calculate relative volume (current volume / average volume)
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['relative_volume'] = df['volume'] / df['volume_ma']
            
            for i in range(1, len(df)):
                idx = df.index[i]
                
                # High volume with price action (volume confirmation)
                if df.loc[idx, 'relative_volume'] >= volume_threshold:
                    # Volume spike on up candle
                    if df.loc[idx, 'close'] > df.loc[idx, 'open']:
                        signals.loc[idx, 'buy_score'] += strategy_weight * 0.7
                        if signals.loc[idx, 'buy_score'] > signal_config.get('threshold', 0.6):
                            signals.loc[idx, 'buy_signal'] = True
                    
                    # Volume spike on down candle
                    elif df.loc[idx, 'close'] < df.loc[idx, 'open']:
                        signals.loc[idx, 'sell_score'] += strategy_weight * 0.7
                        if signals.loc[idx, 'sell_score'] > signal_config.get('threshold', 0.6):
                            signals.loc[idx, 'sell_signal'] = True
        
        # Apply session weights
        session_weights = signal_config.get('session_weights', {})
        if 'session' in df.columns and session_weights:
            for idx in df.index:
                session = df.loc[idx, 'session']
                if session in session_weights:
                    weight = session_weights[session]
                    signals.loc[idx, 'buy_score'] *= weight
                    signals.loc[idx, 'sell_score'] *= weight
        
        # Calculate signal strength based on score
        for idx in signals.index:
            if signals.loc[idx, 'buy_signal']:
                signals.loc[idx, 'signal_strength'] = min(1.0, signals.loc[idx, 'buy_score'] / 2.0)
            elif signals.loc[idx, 'sell_signal']:
                signals.loc[idx, 'signal_strength'] = min(1.0, signals.loc[idx, 'sell_score'] / 2.0)
        
        # Calculate target prices and stop losses
        atr_target_mult = indicator_config['atr'].get('multiplier_target', 2.0)
        atr_stop_mult = indicator_config['atr'].get('multiplier_stop', 1.0)
        
        for idx in signals.index:
            if signals.loc[idx, 'buy_signal']:
                atr_value = df.loc[idx, 'atr']
                strength = signals.loc[idx, 'signal_strength']
                
                signals.loc[idx, 'target_price'] = df.loc[idx, 'close'] + (atr_value * atr_target_mult * strength)
                signals.loc[idx, 'stop_loss'] = df.loc[idx, 'close'] - (atr_value * atr_stop_mult)
                
            elif signals.loc[idx, 'sell_signal']:
                atr_value = df.loc[idx, 'atr']
                strength = signals.loc[idx, 'signal_strength']
                
                signals.loc[idx, 'target_price'] = df.loc[idx, 'close'] - (atr_value * atr_target_mult * strength)
                signals.loc[idx, 'stop_loss'] = df.loc[idx, 'close'] + (atr_value * atr_stop_mult)
        
        # Get only rows with signals
        signal_rows = signals[(signals['buy_signal'] == True) | (signals['sell_signal'] == True)]
        
        # Create result dictionary
        result = {
            "success": True,
            "data": {
                "signals": signals,
                "signal_rows": signal_rows,
                "last_close": df['close'].iloc[-1] if not df.empty else None,
                "last_data_point": df.index[-1] if not df.empty else None,
                "indicators": {
                    "ema_fast": df['ema_fast'],
                    "ema_slow": df['ema_slow'],
                    "vwap": df['vwap'],
                    "rsi": df['rsi'],
                    "bb_upper": df['bb_upper'],
                    "bb_middle": df['bb_middle'],
                    "bb_lower": df['bb_lower'],
                    "atr": df['atr']
                }
            }
        }
        
        print(f"SPY analysis complete: {signal_rows['buy_signal'].sum()} buy signals, {signal_rows['sell_signal'].sum()} sell signals")
        return result
        
    except Exception as e:
        import traceback
        print(f"Error in SPY day trading analysis: {str(e)}")
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "signals": pd.DataFrame()
        }

def render_spy_signals(results: Dict[str, Any], symbol: str = "SPY"):
    """
    Prepare SPY signals data for visualization.
    This function formats SPY-specific signals for display in the UI.
    
    Args:
        results: Results from analyze_spy_day_trading
        symbol: Trading symbol (default: SPY)
        
    Returns:
        Dictionary with formatted signal data for UI rendering
    """
    if not results.get("success", False):
        return {
            "error": results.get("error", "Unknown error in SPY analysis"),
            "signals": pd.DataFrame()
        }
    
    data = results.get("data", {})
    signals = data.get("signals", pd.DataFrame())
    signal_rows = data.get("signal_rows", pd.DataFrame())
    
    # Format timestamps and signal details
    if not signal_rows.empty:
        formatted_signals = []
        
        for idx, row in signal_rows.iterrows():
            # Format timestamp in Eastern Time
            if hasattr(idx, 'tz_convert'):
                eastern_tz = pytz.timezone('US/Eastern')
                # Convert to Eastern time and format correctly
                timestamp = idx.tz_convert(eastern_tz).strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp = str(idx)
            
            # Double-check timestamp is not in the future
            if isinstance(idx, pd.Timestamp):
                now = pd.Timestamp.now(tz='UTC')
                if idx > now:
                    # If timestamp is in the future, use current date with the time from the index
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    if hasattr(idx, 'tz_convert'):
                        time_part = idx.tz_convert(eastern_tz).strftime("%H:%M:%S")
                    else:
                        time_part = idx.strftime("%H:%M:%S")
                    timestamp = f"{current_date} {time_part}"
            
            # Determine signal type
            signal_type = "BUY" if row.get('buy_signal', False) else "SELL" if row.get('sell_signal', False) else "NEUTRAL"
            
            # Get values safely with None checks
            signal_price = row.get('signal_price')
            target_price = row.get('target_price')
            stop_loss = row.get('stop_loss')
            
            # Calculate risk-reward ratio with thorough None/zero checks
            risk_reward = None
            if signal_type == "BUY" and signal_price is not None and target_price is not None and stop_loss is not None:
                if signal_price > 0 and target_price > 0 and stop_loss > 0:
                    risk = signal_price - stop_loss
                    reward = target_price - signal_price
                    if risk > 0:
                        risk_reward = reward / risk
            elif signal_type == "SELL" and signal_price is not None and target_price is not None and stop_loss is not None:
                if signal_price > 0 and target_price > 0 and stop_loss > 0:
                    risk = stop_loss - signal_price
                    reward = signal_price - target_price
                    if risk > 0:
                        risk_reward = reward / risk
            
            formatted_signal = {
                "timestamp": timestamp,
                "type": signal_type,
                "price": signal_price if signal_price is not None else 0,
                "strength": row.get('signal_strength', 0),
                "score": row.get('buy_score' if signal_type == "BUY" else 'sell_score', 0),
                "target": target_price,
                "stop_loss": stop_loss,
                "risk_reward": risk_reward
            }
            
            formatted_signals.append(formatted_signal)
        
        return {
            "success": True,
            "signals": formatted_signals,
            "counts": {
                "buy": signal_rows['buy_signal'].sum(),
                "sell": signal_rows['sell_signal'].sum(),
                "total": len(signal_rows)
            },
            "last_close": data.get("last_close", None),
            "last_time": data.get("last_data_point", None)
        }
    else:
        return {
            "success": True,
            "signals": [],
            "counts": {
                "buy": 0,
                "sell": 0,
                "total": 0
            },
            "last_close": data.get("last_close", None),
            "last_time": data.get("last_data_point", None)
        } 