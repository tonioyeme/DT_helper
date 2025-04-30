import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

def filter_signals_by_strategy(signals: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Filter signals based on strategy type
    
    Args:
        signals: DataFrame with trading signals
        strategy: Strategy name to filter by (e.g., 'ema_vwap', 'mm_vol', 'multi_indicator')
        
    Returns:
        DataFrame with filtered signals
    """
    filtered_signals = signals.copy()
    
    # Reset buy and sell signals
    filtered_signals['buy_signal'] = False
    filtered_signals['sell_signal'] = False
    
    if strategy == 'ema_vwap':
        # Filter for EMA + VWAP strategy
        mask = (signals['ema_vwap_bullish'] | signals['ema_vwap_bearish'])
        filtered_signals.loc[mask & signals['ema_vwap_bullish'], 'buy_signal'] = True
        filtered_signals.loc[mask & signals['ema_vwap_bearish'], 'sell_signal'] = True
        
    elif strategy == 'mm_vol':
        # Filter for Measured Move + Volume strategy
        mask = (signals['mm_vol_bullish'] | signals['mm_vol_bearish'])
        filtered_signals.loc[mask & signals['mm_vol_bullish'], 'buy_signal'] = True
        filtered_signals.loc[mask & signals['mm_vol_bearish'], 'sell_signal'] = True
        
    elif strategy == 'ema_cross':
        # Filter for EMA crossover signals
        mask = (signals['price_cross_above_ema20'] | signals['price_cross_below_ema20'])
        ema50_condition = signals['price_above_ema50']
        
        filtered_signals.loc[mask & signals['price_cross_above_ema20'] & ema50_condition, 'buy_signal'] = True
        filtered_signals.loc[mask & signals['price_cross_below_ema20'] & ~ema50_condition, 'sell_signal'] = True
        
    elif strategy == 'macd':
        # Filter for MACD signals
        mask = (signals['macd_cross_above_signal'] | signals['macd_cross_below_signal'])
        filtered_signals.loc[mask & signals['macd_cross_above_signal'], 'buy_signal'] = True
        filtered_signals.loc[mask & signals['macd_cross_below_signal'], 'sell_signal'] = True
        
    elif strategy == 'vwap':
        # Filter for VWAP signals
        mask = (signals['price_cross_above_vwap'] | signals['price_cross_below_vwap'])
        filtered_signals.loc[mask & signals['price_cross_above_vwap'], 'buy_signal'] = True
        filtered_signals.loc[mask & signals['price_cross_below_vwap'], 'sell_signal'] = True
        
    elif strategy == 'multi_indicator':
        # For multi-indicator, we keep the original signals but ensure they're based on multiple indicators
        # We define "multiple" as having at least 2 confirming indicators
        filtered_signals.loc[(signals['buy_signal']) & (signals['buy_count'] >= 2), 'buy_signal'] = True
        filtered_signals.loc[(signals['sell_signal']) & (signals['sell_count'] >= 2), 'sell_signal'] = True
        
    elif strategy == 'strong_signals':
        # Filter for strong signals (signal strength >= 3)
        filtered_signals.loc[(signals['buy_signal']) & (signals['signal_strength'] >= 3), 'buy_signal'] = True
        filtered_signals.loc[(signals['sell_signal']) & (signals['signal_strength'] >= 3), 'sell_signal'] = True
        
    elif strategy == 'strong_buy_only':
        # Filter for only strong buy signals
        filtered_signals.loc[(signals['buy_signal']) & (signals['signal_strength'] >= 3), 'buy_signal'] = True
        
    elif strategy == 'strong_sell_only':
        # Filter for only strong sell signals
        filtered_signals.loc[(signals['sell_signal']) & (signals['signal_strength'] >= 3), 'sell_signal'] = True
        
    elif strategy == 'all':
        # Keep all signals
        filtered_signals['buy_signal'] = signals['buy_signal']
        filtered_signals['sell_signal'] = signals['sell_signal']
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    return filtered_signals

def filter_signals_by_date(signals: pd.DataFrame, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Filter signals based on date range
    
    Args:
        signals: DataFrame with trading signals
        start_date: Start date in 'YYYY-MM-DD' format (inclusive)
        end_date: End date in 'YYYY-MM-DD' format (inclusive)
        
    Returns:
        DataFrame with filtered signals
    """
    filtered_signals = signals.copy()
    
    if start_date:
        filtered_signals = filtered_signals[filtered_signals.index >= start_date]
        
    if end_date:
        filtered_signals = filtered_signals[filtered_signals.index <= end_date]
        
    return filtered_signals

def create_strategy_comparison(data: pd.DataFrame, signals: pd.DataFrame, strategies: List[str], 
                              initial_capital: float = 10000.0, position_size: float = 0.1,
                              commission: float = 0.0, slippage: float = 0.0) -> pd.DataFrame:
    """
    Compare performance of different strategies
    
    Args:
        data: DataFrame with OHLCV price data
        signals: DataFrame with trading signals
        strategies: List of strategy names to compare
        initial_capital: Initial capital for each backtest
        position_size: Position size as a fraction of capital
        commission: Commission per trade as a percentage
        slippage: Slippage per trade as a percentage
        
    Returns:
        DataFrame with performance metrics for each strategy
    """
    from app.backtest.engine import BacktestEngine
    
    results = []
    
    for strategy in strategies:
        # Filter signals for the current strategy
        try:
            filtered_signals = filter_signals_by_strategy(signals, strategy)
            
            # Run backtest
            backtest = BacktestEngine(
                data=data,
                signals=filtered_signals,
                initial_capital=initial_capital,
                position_size=position_size,
                commission=commission,
                slippage=slippage,
                strategy_name=strategy
            )
            
            metrics = backtest.run()
            results.append(metrics)
        except Exception as e:
            print(f"Error backtesting {strategy}: {str(e)}")
            continue
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def create_signal_strength_comparison(data: pd.DataFrame, signals: pd.DataFrame, 
                                     initial_capital: float = 10000.0, position_size: float = 0.1,
                                     commission: float = 0.0, slippage: float = 0.0) -> pd.DataFrame:
    """
    Compare performance based on signal strength filtering
    
    Args:
        data: DataFrame with OHLCV price data
        signals: DataFrame with trading signals
        initial_capital: Initial capital for each backtest
        position_size: Position size as a fraction of capital
        commission: Commission per trade as a percentage
        slippage: Slippage per trade as a percentage
        
    Returns:
        DataFrame with performance metrics for each signal strength level
    """
    from app.backtest.engine import BacktestEngine
    
    strength_levels = {
        "All Signals": 0,
        "Weak+": 1,
        "Moderate+": 2,
        "Strong+": 3,
        "Very Strong": 4
    }
    
    results = []
    
    for name, strength in strength_levels.items():
        # Run backtest with the specified signal strength filter
        try:
            backtest = BacktestEngine(
                data=data,
                signals=signals,
                initial_capital=initial_capital,
                position_size=position_size,
                commission=commission,
                slippage=slippage,
                strategy_name=name
            )
            
            metrics = backtest.run(filter_strength=strength)
            results.append(metrics)
        except Exception as e:
            print(f"Error backtesting {name}: {str(e)}")
            continue
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def optimize_position_size(data: pd.DataFrame, signals: pd.DataFrame, 
                          position_sizes: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                          strategy: str = "all", filter_strength: int = 0,
                          initial_capital: float = 10000.0,
                          commission: float = 0.0, slippage: float = 0.0) -> pd.DataFrame:
    """
    Optimize position size for a given strategy
    
    Args:
        data: DataFrame with OHLCV price data
        signals: DataFrame with trading signals
        position_sizes: List of position sizes to test
        strategy: Strategy name to optimize
        filter_strength: Minimum signal strength to consider
        initial_capital: Initial capital for each backtest
        commission: Commission per trade as a percentage
        slippage: Slippage per trade as a percentage
        
    Returns:
        DataFrame with performance metrics for each position size
    """
    from app.backtest.engine import BacktestEngine
    
    # Filter signals for the specified strategy
    if strategy != "all":
        signals = filter_signals_by_strategy(signals, strategy)
    
    results = []
    
    for pos_size in position_sizes:
        try:
            backtest = BacktestEngine(
                data=data,
                signals=signals,
                initial_capital=initial_capital,
                position_size=pos_size,
                commission=commission,
                slippage=slippage,
                strategy_name=f"{strategy} (Size: {pos_size:.2f})"
            )
            
            metrics = backtest.run(filter_strength=filter_strength)
            metrics['position_size'] = pos_size
            results.append(metrics)
        except Exception as e:
            print(f"Error backtesting position size {pos_size}: {str(e)}")
            continue
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def get_available_strategies() -> List[str]:
    """
    Get a list of available strategy names for filtering
    
    Returns:
        List of strategy names
    """
    return [
        "all",
        "ema_vwap", 
        "mm_vol", 
        "ema_cross", 
        "macd", 
        "vwap", 
        "multi_indicator",
        "strong_signals",
        "strong_buy_only",
        "strong_sell_only"
    ]

def calculate_sharpe(signals, risk_free_rate=0.0):
    """
    Calculate Sharpe ratio based on signals and returns
    
    Args:
        signals (pd.DataFrame): DataFrame with trading signals and returns
        risk_free_rate (float): Risk-free rate (annualized decimal)
        
    Returns:
        float: Sharpe ratio
    """
    if 'returns' not in signals.columns:
        # If returns not available, try to calculate from price data
        if all(col in signals.columns for col in ['buy_signal', 'sell_signal', 'close']):
            # Simple returns calculation based on signals
            signals['returns'] = 0.0
            signals.loc[signals['buy_signal'] == True, 'returns'] = signals['close'].pct_change()
            signals.loc[signals['sell_signal'] == True, 'returns'] = -signals['close'].pct_change()
        else:
            return 0.0  # Can't calculate without necessary data
            
    # Calculate average return and standard deviation
    avg_return = signals['returns'].mean() * 252  # Annualize
    std_return = signals['returns'].std() * np.sqrt(252)  # Annualize
    
    # Handle case where std is zero
    if std_return == 0:
        return 0.0
        
    # Calculate Sharpe ratio
    sharpe = (avg_return - risk_free_rate) / std_return
    
    return sharpe

def walk_forward_test(data, strategy=None, n_splits=5, train_ratio=0.7, 
                     initial_capital=10000.0, position_size=0.1, 
                     commission=0.1, slippage=0.05):
    """
    Perform walk-forward optimization to validate strategy robustness
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV price data
        strategy (str): Strategy name to test (None for all signals)
        n_splits (int): Number of testing periods
        train_ratio (float): Ratio of training data in each split
        initial_capital (float): Initial capital for each test
        position_size (float): Position size as fraction of capital
        commission (float): Commission percentage
        slippage (float): Slippage percentage
        
    Returns:
        dict: Dictionary with walk-forward test results
    """
    from app.signals import generate_signals
    
    results = []
    total_length = len(data)
    split_size = total_length // n_splits
    
    # Create summary
    summary = {
        'splits': [],
        'train_sharpe': [],
        'test_sharpe': [],
        'train_return': [],
        'test_return': [],
        'robustness_score': 0
    }
    
    for i in range(n_splits):
        # Calculate split indices
        start_idx = i * split_size
        train_end_idx = start_idx + int(split_size * train_ratio)
        test_end_idx = min(train_end_idx + split_size, total_length)
        
        # Skip if we don't have enough data
        if train_end_idx >= total_length or test_end_idx >= total_length:
            continue
            
        # Get train and test data
        train_data = data.iloc[start_idx:train_end_idx]
        test_data = data.iloc[train_end_idx:test_end_idx]
        
        # Skip small datasets
        if len(train_data) < 20 or len(test_data) < 10:
            continue
            
        # Generate signals
        train_signals = generate_signals(train_data)
        test_signals = generate_signals(test_data)
        
        # Filter by strategy if needed
        if strategy and strategy != "all":
            train_signals = filter_signals_by_strategy(train_signals, strategy)
            test_signals = filter_signals_by_strategy(test_signals, strategy)
        
        # Initialize backtest for train data
        train_backtest = BacktestEngine(
            data=train_data,
            signals=train_signals,
            initial_capital=initial_capital,
            position_size=position_size,
            commission=commission,
            slippage=slippage,
            strategy_name=strategy if strategy else "all"
        )
        
        # Initialize backtest for test data
        test_backtest = BacktestEngine(
            data=test_data,
            signals=test_signals,
            initial_capital=initial_capital,
            position_size=position_size,
            commission=commission,
            slippage=slippage,
            strategy_name=strategy if strategy else "all"
        )
        
        # Run backtests
        train_metrics = train_backtest.run()
        test_metrics = test_backtest.run()
        
        # Calculate performance metrics
        train_sharpe = calculate_sharpe(train_signals)
        test_sharpe = calculate_sharpe(test_signals)
        
        # Get returns
        train_return = train_metrics.get('total_return', 0)
        test_return = test_metrics.get('total_return', 0)
        
        # Store results
        split_result = {
            'split': i + 1,
            'train_start': train_data.index[0] if len(train_data) > 0 else None,
            'train_end': train_data.index[-1] if len(train_data) > 0 else None,
            'test_start': test_data.index[0] if len(test_data) > 0 else None,
            'test_end': test_data.index[-1] if len(test_data) > 0 else None,
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'train_return': train_return,
            'test_return': test_return,
            'consistency': 1 if (train_return > 0 and test_return > 0) or (train_return < 0 and test_return < 0) else 0
        }
        
        results.append(split_result)
        
        # Update summary
        summary['splits'].append(i + 1)
        summary['train_sharpe'].append(train_sharpe)
        summary['test_sharpe'].append(test_sharpe)
        summary['train_return'].append(train_return)
        summary['test_return'].append(test_return)
    
    # Calculate robustness score (percentage of test periods where performance is consistent with training)
    robustness_score = sum(r['consistency'] for r in results) / len(results) if results else 0
    summary['robustness_score'] = robustness_score
    
    return {
        'results': results,
        'summary': summary
    } 