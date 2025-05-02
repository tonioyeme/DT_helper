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

# Import position manager
from app.signals.position_manager import (
    get_position_manager, calculate_exit_timing, get_vix_level, 
    SPY_CONFIG as POSITION_CONFIG
)

# Import position manager processing
from app.signals.processing import process_signals_with_position_manager

# Import the new exit strategy
from app.signals.exit_signals import apply_exit_strategy, EnhancedExitStrategy, SPY_EXIT_CONFIG

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

def apply_exit_priority_flags(signals: pd.DataFrame, data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add exit priority flags to ensure positions are exited before entering opposite positions
    Uses enhanced sequential position manager for SPY-optimized execution
    
    Args:
        signals: DataFrame with buy and sell signals
        data: DataFrame with OHLCV data for execution optimization
        
    Returns:
        DataFrame with exit priority flags added
    """
    # Create a copy to avoid modifying original
    processed_signals = signals.copy()
    
    # Initialize exit flag columns if they don't exist
    if 'exit_buy' not in processed_signals.columns:
        processed_signals['exit_buy'] = False
    if 'exit_sell' not in processed_signals.columns:
        processed_signals['exit_sell'] = False
    
    # If we don't have any signals, just return
    if len(processed_signals) == 0:
        return processed_signals
    
    # Get position manager and reset it for clean processing
    position_manager = get_position_manager(reset=True)
    
    # Process through every signal in chronological order to build proper signal sequence
    current_position = None
    pending_exit = False
    
    # Track processed signals in a new DataFrame to ensure chronological order
    chronological_signals = []
    
    # First pass to collect all raw signals
    for idx, row in processed_signals.iterrows():
        # Extract price
        price = row.get('close', None) if 'close' in row else row.get('signal_price', None)
        
        # Skip if we don't have price data
        if price is None:
            continue
            
        # Default signal is neutral
        signal = 'neutral'
        
        # Determine signal type
        if row.get('buy_signal', False):
            signal = 'buy'
        elif row.get('sell_signal', False):
            signal = 'sell'
            
        # Process the signal through position manager for sequential exit-entry enforcement
        result = position_manager.process_signal(signal, data, price, idx)
        
        # Track key state
        current_position = position_manager.current_position
        pending_exit = position_manager.pending_exit
        
        # Record this chronological signal with extra metadata about position state
        signal_record = {
            'index': idx,
            'raw_signal': signal,
            'filtered_signal': signal if result['accepted'] else 'neutral',
            'price': price,
            'exit_required': pending_exit,
            'current_position': current_position.direction if current_position else None,
            'reason': result['reason']
        }
        
        chronological_signals.append(signal_record)
    
    # Build the processed signals with forced exit-before-entry logic
    for i, record in enumerate(chronological_signals):
        idx = record['index']
        
        # Handle exit flags - mark them BEFORE new entries of opposite direction
        if record['exit_required'] and record['current_position']:
            pos_type = record['current_position']
            # Add exit flag
            if pos_type == 'buy':
                processed_signals.loc[idx, 'exit_buy'] = True
            elif pos_type == 'sell':
                processed_signals.loc[idx, 'exit_sell'] = True
        
        # Update the original row's filtered signals
        if record['filtered_signal'] == 'buy':
            processed_signals.loc[idx, 'filtered_buy_signal'] = True
            processed_signals.loc[idx, 'filtered_sell_signal'] = False
        elif record['filtered_signal'] == 'sell':
            processed_signals.loc[idx, 'filtered_buy_signal'] = False
            processed_signals.loc[idx, 'filtered_sell_signal'] = True
        else:
            processed_signals.loc[idx, 'filtered_buy_signal'] = False
            processed_signals.loc[idx, 'filtered_sell_signal'] = False
            
    return processed_signals

def apply_volatility_adjusted_timing(signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply volatility-adjusted timing for signal execution with SPY-specific optimizations
    
    Args:
        signals: DataFrame with signals
        data: DataFrame with OHLCV data for volatility calculation
        
    Returns:
        DataFrame with volatility-adjusted timing
    """
    # Calculate implied volatility from recent price data if available
    if data is not None and len(data) >= 20:
        try:
            # Calculate log returns
            log_returns = np.log(data['close'] / data['close'].shift(1)).iloc[-20:]
            
            # Calculate annualized volatility
            realized_vol = log_returns.std() * np.sqrt(252) * 100
            
            # Convert to implied volatility (typically 10% higher than realized)
            vix = realized_vol * 1.1
        except Exception:
            # Fall back to global VIX level if calculation fails
            vix = get_vix_level()
    else:
        # Use global VIX level if no price data available
        vix = get_vix_level()
    
    # Calculate dynamic timing based on VIX
    execution_delay = calculate_exit_timing(vix)
    
    # Add delay information to signals
    signals['exit_delay'] = execution_delay
    signals['implied_volatility'] = vix
    
    # Add emergency exit flag for end-of-day positions
    signals['emergency_exit'] = False
    
    # Set emergency exit flag for positions near market close
    if isinstance(signals.index[0], pd.Timestamp):
        # Convert to Eastern timezone if needed
        if signals.index[0].tzinfo is not None:
            eastern_tz = pytz.timezone('US/Eastern')
            times = signals.index.tz_convert(eastern_tz).strftime('%H:%M')
        else:
            times = signals.index.strftime('%H:%M')
        
        # Flag positions near market close for emergency exit
        emergency_time = POSITION_CONFIG['emergency_clear_time']
        signals['emergency_exit'] = [t >= emergency_time for t in times]
    
    return signals

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
        
        # Get only rows with signals for statistics
        signal_rows = signals[(signals['buy_signal'] == True) | (signals['sell_signal'] == True)]
        
        # Apply the new exit strategy
        signals = apply_exit_strategy(df, signals)
        
        # Apply volatility-adjusted timing
        signals = apply_volatility_adjusted_timing(signals, df)
        
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
        
        # Output signal statistics
        buy_signals_count = signals['filtered_buy_signal'].sum() if 'filtered_buy_signal' in signals.columns else signals['buy_signal'].sum()
        sell_signals_count = signals['filtered_sell_signal'].sum() if 'filtered_sell_signal' in signals.columns else signals['sell_signal'].sum()
        exit_buy_count = signals['exit_buy'].sum() if 'exit_buy' in signals.columns else 0
        exit_sell_count = signals['exit_sell'].sum() if 'exit_sell' in signals.columns else 0
        
        print(f"SPY analysis complete: {buy_signals_count} buy signals, {sell_signals_count} sell signals")
        print(f"Exit signals: {exit_buy_count} exit buys, {exit_sell_count} exit sells")
        
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
    
    # Instead of using only signal_rows, use the full signals DataFrame with both entries and exits
    all_signals = []
    
    # Track the current position for sequential ordering validation
    current_position = None
    
    # Process entry signals (buy/sell)
    for idx, row in signals.iterrows():
        price = row.get('close', 0)
        
        # Check if this is an entry signal
        if row.get('filtered_buy_signal', False) or row.get('filtered_sell_signal', False):
            signal_type = 'BUY' if row.get('filtered_buy_signal', False) else 'SELL'
            
            # Verify this doesn't violate sequential ordering
            if current_position is not None and current_position != 'neutral' and signal_type != current_position:
                # If signals aren't properly sequenced, we need an exit first
                all_signals.append({
                    'time': idx,
                    'signal_type': f'EXIT_{current_position}',
                    'price': price,
                    'explanation': 'Forced exit to maintain sequential ordering'
                })
            
            # Add the entry signal
            all_signals.append({
                'time': idx,
                'signal_type': signal_type,
                'price': price,
                'explanation': row.get('signal_explanation', f'{signal_type} signal') 
            })
            
            # Update current position
            current_position = signal_type
        
        # Check if this is an exit signal
        if row.get('exit_buy', False) or row.get('exit_sell', False):
            exit_type = 'BUY' if row.get('exit_buy', False) else 'SELL'
            
            all_signals.append({
                'time': idx,
                'signal_type': f'EXIT_{exit_type}',
                'price': price,
                'explanation': 'Sequential exit before new entry'
            })
            
            # Update current position to neutral after exit
            current_position = 'neutral'
    
    # Sort signals by time
    all_signals.sort(key=lambda x: x['time'])
    
    # Build signal rows for UI display
    signal_rows = []
    
    for signal in all_signals:
        formatted_time = signal['time']
        if hasattr(formatted_time, 'strftime'):
            formatted_time = formatted_time.strftime('%Y-%m-%d %H:%M:%S')
        
        signal_rows.append({
            'time': formatted_time,
            'signal': signal['signal_type'],
            'price': f"${signal['price']:.2f}",
            'explanation': signal['explanation']
        })
    
    # Additional processing for UI rendering
    ui_data = {
        "symbol": symbol,
        "signals": pd.DataFrame(signal_rows),
        "summary": {
            "total_signals": len(signal_rows),
            "buy_signals": len([s for s in all_signals if s['signal_type'] == 'BUY']),
            "sell_signals": len([s for s in all_signals if s['signal_type'] == 'SELL']),
            "exit_signals": len([s for s in all_signals if 'EXIT' in s['signal_type']])
        }
    }
    
    return ui_data 