import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional

class BacktestEngine:
    """
    Engine for backtesting trading strategies based on generated signals
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        signals: pd.DataFrame, 
        initial_capital: float = 10000.0,
        position_size: float = 0.1,  # 10% of capital per trade
        commission: float = 0.0,     # Commission per trade in percentage
        slippage: float = 0.0,       # Slippage in percentage
        strategy_name: str = "All Signals"
    ):
        """
        Initialize the backtesting engine
        
        Args:
            data: DataFrame with OHLCV price data
            signals: DataFrame with trading signals
            initial_capital: Initial capital for the backtest
            position_size: Size of each position as a fraction of capital
            commission: Commission cost per trade as a percentage
            slippage: Slippage per trade as a percentage
            strategy_name: Name of the strategy being tested
        """
        self.data = data.copy()
        self.signals = signals.copy()
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.slippage = slippage
        self.strategy_name = strategy_name
        
        # Ensure we have required columns
        required_cols = ['buy_signal', 'sell_signal', 'target_price', 'stop_loss']
        for col in required_cols:
            if col not in self.signals.columns:
                raise ValueError(f"Signals DataFrame must contain '{col}' column")
        
        # Initialize results
        self.trades = []
        self.equity_curve = None
        self.performance_metrics = {}
        
    def run(self, filter_strength: int = 0) -> Dict:
        """
        Run the backtest
        
        Args:
            filter_strength: Minimum signal strength to consider (0-4)
                             0 = All signals, 1 = Weak or better, 2 = Moderate or better, etc.
        
        Returns:
            Dict of performance metrics
        """
        # Reset state
        self.trades = []
        capital = self.initial_capital
        equity = [capital]
        dates = [self.data.index[0]]
        
        # Current position state
        in_position = False
        entry_price = 0.0
        entry_date = None
        position_type = None  # 'long' or 'short'
        position_size_usd = 0.0
        shares = 0
        target = 0.0
        stop = 0.0
        
        # Performance tracking
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        break_even_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        max_drawdown = 0.0
        peak_equity = capital
        
        # Main backtest loop
        for i, idx in enumerate(self.data.index):
            current_price = self.data.loc[idx, 'close']
            
            # Skip if we don't have signal data for this timestamp
            if idx not in self.signals.index:
                continue
                
            # Update equity curve with current position value
            if in_position:
                if position_type == 'long':
                    current_value = shares * current_price
                else:  # short position
                    # For shorts, we calculate profit/loss differently
                    current_value = position_size_usd + (entry_price - current_price) * shares
                    
                new_equity = (capital - position_size_usd) + current_value
            else:
                new_equity = capital
                
            equity.append(new_equity)
            dates.append(idx)
            
            # Check for updated max drawdown
            if new_equity > peak_equity:
                peak_equity = new_equity
            drawdown = (peak_equity - new_equity) / peak_equity
            max_drawdown = max(max_drawdown, drawdown)
            
            # Check for target or stop hits if in a position
            if in_position:
                # For long positions
                if position_type == 'long':
                    # Check if high price hit target
                    if self.data.loc[idx, 'high'] >= target:
                        # Exit at target
                        exit_price = target * (1 - self.slippage/100)  # Account for slippage
                        exit_date = idx
                        profit = (exit_price - entry_price) * shares - (2 * position_size_usd * self.commission/100)
                        
                        # Record trade
                        self.trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'shares': shares,
                            'pnl': profit,
                            'pnl_pct': profit / position_size_usd * 100,
                            'type': position_type,
                            'exit_reason': 'target'
                        })
                        
                        # Update performance metrics
                        total_trades += 1
                        if profit > 0:
                            winning_trades += 1
                            total_profit += profit
                        elif profit < 0:
                            losing_trades += 1
                            total_loss += abs(profit)
                        else:
                            break_even_trades += 1
                            
                        # Update capital and reset position
                        capital += profit
                        in_position = False
                        
                    # Check if low price hit stop loss
                    elif self.data.loc[idx, 'low'] <= stop:
                        # Exit at stop
                        exit_price = stop * (1 - self.slippage/100)  # Account for slippage
                        exit_date = idx
                        profit = (exit_price - entry_price) * shares - (2 * position_size_usd * self.commission/100)
                        
                        # Record trade
                        self.trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'shares': shares,
                            'pnl': profit,
                            'pnl_pct': profit / position_size_usd * 100,
                            'type': position_type,
                            'exit_reason': 'stop'
                        })
                        
                        # Update performance metrics
                        total_trades += 1
                        if profit > 0:
                            winning_trades += 1
                            total_profit += profit
                        elif profit < 0:
                            losing_trades += 1
                            total_loss += abs(profit)
                        else:
                            break_even_trades += 1
                            
                        # Update capital and reset position
                        capital += profit
                        in_position = False
                
                # For short positions
                else:  # position_type == 'short'
                    # Check if low price hit target
                    if self.data.loc[idx, 'low'] <= target:
                        # Exit at target
                        exit_price = target * (1 + self.slippage/100)  # Account for slippage
                        exit_date = idx
                        profit = (entry_price - exit_price) * shares - (2 * position_size_usd * self.commission/100)
                        
                        # Record trade
                        self.trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'shares': shares,
                            'pnl': profit,
                            'pnl_pct': profit / position_size_usd * 100,
                            'type': position_type,
                            'exit_reason': 'target'
                        })
                        
                        # Update performance metrics
                        total_trades += 1
                        if profit > 0:
                            winning_trades += 1
                            total_profit += profit
                        elif profit < 0:
                            losing_trades += 1
                            total_loss += abs(profit)
                        else:
                            break_even_trades += 1
                            
                        # Update capital and reset position
                        capital += profit
                        in_position = False
                        
                    # Check if high price hit stop loss    
                    elif self.data.loc[idx, 'high'] >= stop:
                        # Exit at stop
                        exit_price = stop * (1 + self.slippage/100)  # Account for slippage
                        exit_date = idx
                        profit = (entry_price - exit_price) * shares - (2 * position_size_usd * self.commission/100)
                        
                        # Record trade
                        self.trades.append({
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'shares': shares,
                            'pnl': profit,
                            'pnl_pct': profit / position_size_usd * 100,
                            'type': position_type,
                            'exit_reason': 'stop'
                        })
                        
                        # Update performance metrics
                        total_trades += 1
                        if profit > 0:
                            winning_trades += 1
                            total_profit += profit
                        elif profit < 0:
                            losing_trades += 1
                            total_loss += abs(profit)
                        else:
                            break_even_trades += 1
                            
                        # Update capital and reset position
                        capital += profit
                        in_position = False
            
            # Check for new signals if not in a position
            if not in_position:
                signal_row = self.signals.loc[idx]
                
                # Check signal strength if filter is applied
                if filter_strength > 0 and 'signal_strength' in signal_row and signal_row['signal_strength'] < filter_strength:
                    continue
                
                # Check for buy signal
                if signal_row['buy_signal']:
                    # Calculate position size based on capital
                    position_size_usd = capital * self.position_size
                    entry_price = current_price * (1 + self.slippage/100)  # Account for slippage
                    shares = position_size_usd / entry_price
                    entry_date = idx
                    position_type = 'long'
                    in_position = True
                    
                    # Set target and stop loss
                    target = signal_row['target_price'] if pd.notna(signal_row['target_price']) else entry_price * 1.05
                    stop = signal_row['stop_loss'] if pd.notna(signal_row['stop_loss']) else entry_price * 0.97
                
                # Check for sell signal (short position)
                elif signal_row['sell_signal']:
                    # Calculate position size based on capital
                    position_size_usd = capital * self.position_size
                    entry_price = current_price * (1 - self.slippage/100)  # Account for slippage
                    shares = position_size_usd / entry_price
                    entry_date = idx
                    position_type = 'short'
                    in_position = True
                    
                    # Set target and stop loss
                    target = signal_row['target_price'] if pd.notna(signal_row['target_price']) else entry_price * 0.95
                    stop = signal_row['stop_loss'] if pd.notna(signal_row['stop_loss']) else entry_price * 1.03
        
        # Close any open position at the end of the backtest with the last price
        if in_position:
            exit_price = self.data.iloc[-1]['close']
            exit_date = self.data.index[-1]
            
            if position_type == 'long':
                profit = (exit_price - entry_price) * shares - (2 * position_size_usd * self.commission/100)
            else:  # short position
                profit = (entry_price - exit_price) * shares - (2 * position_size_usd * self.commission/100)
            
            # Record final trade
            self.trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'pnl': profit,
                'pnl_pct': profit / position_size_usd * 100,
                'type': position_type,
                'exit_reason': 'end_of_data'
            })
            
            # Update performance metrics
            total_trades += 1
            if profit > 0:
                winning_trades += 1
                total_profit += profit
            elif profit < 0:
                losing_trades += 1
                total_loss += abs(profit)
            else:
                break_even_trades += 1
                
            # Update final capital
            capital += profit
            
        # Create equity curve DataFrame
        self.equity_curve = pd.DataFrame({
            'date': dates,
            'equity': equity
        }).set_index('date')
        
        # Create trades DataFrame
        self.trades_df = pd.DataFrame(self.trades)
        
        # Calculate performance metrics
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Total return
        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        
        # Annualized return
        start_date = self.data.index[0]
        end_date = self.data.index[-1]
        days = (end_date - start_date).days
        if days > 0:
            annualized_return = ((1 + total_return/100) ** (365/days) - 1) * 100
        else:
            annualized_return = 0
            
        # Store performance metrics
        self.performance_metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'break_even_trades': break_even_trades,
            'win_rate': win_rate * 100,
            'loss_rate': loss_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown * 100,
            'profit_factor': profit_factor,
            'strategy': self.strategy_name
        }
        
        return self.performance_metrics
    
    def get_trades(self) -> pd.DataFrame:
        """
        Get all trades from the backtest
        
        Returns:
            DataFrame of all trades
        """
        if not hasattr(self, 'trades_df') or self.trades_df is None:
            return pd.DataFrame()
        return self.trades_df
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get the equity curve from the backtest
        
        Returns:
            DataFrame with equity curve
        """
        if self.equity_curve is None:
            return pd.DataFrame()
        return self.equity_curve
    
    def plot_equity_curve(self) -> plt.Figure:
        """
        Plot the equity curve
        
        Returns:
            Matplotlib figure
        """
        if self.equity_curve is None:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.equity_curve.index, self.equity_curve['equity'])
        ax.set_title(f"Equity Curve - {self.strategy_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.grid(True)
        return fig
    
    def plot_drawdown(self) -> plt.Figure:
        """
        Plot the drawdown curve
        
        Returns:
            Matplotlib figure
        """
        if self.equity_curve is None:
            return None
            
        # Calculate drawdown series
        equity = self.equity_curve['equity'].values
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        
        drawdown_series = pd.Series(drawdown, index=self.equity_curve.index)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(drawdown_series.index, 0, drawdown_series, color='red', alpha=0.3)
        ax.plot(drawdown_series.index, drawdown_series, color='red', linewidth=1)
        ax.set_title(f"Drawdown - {self.strategy_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True)
        ax.invert_yaxis()
        return fig
    
    def plot_monthly_returns(self) -> plt.Figure:
        """
        Plot monthly returns
        
        Returns:
            Matplotlib figure
        """
        if self.equity_curve is None or len(self.equity_curve) < 30:
            return None
            
        # Calculate monthly returns
        monthly_returns = self.equity_curve['equity'].resample('M').last().pct_change() * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_returns.plot(kind='bar', ax=ax)
        ax.set_title(f"Monthly Returns - {self.strategy_name}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Return (%)")
        ax.grid(True, axis='y')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        return fig
    
    def get_performance_summary(self) -> str:
        """
        Get a formatted performance summary
        
        Returns:
            Formatted string with performance metrics
        """
        if not self.performance_metrics:
            return "No backtest results available. Run the backtest first."
            
        summary = f"Performance Summary for {self.strategy_name}\n"
        summary += "=" * 50 + "\n"
        summary += f"Initial Capital: ${self.performance_metrics['initial_capital']:,.2f}\n"
        summary += f"Final Capital: ${self.performance_metrics['final_capital']:,.2f}\n"
        summary += f"Total Return: {self.performance_metrics['total_return']:.2f}%\n"
        summary += f"Annualized Return: {self.performance_metrics['annualized_return']:.2f}%\n"
        summary += f"Max Drawdown: {self.performance_metrics['max_drawdown']:.2f}%\n\n"
        
        summary += f"Total Trades: {self.performance_metrics['total_trades']}\n"
        summary += f"Winning Trades: {self.performance_metrics['winning_trades']} ({self.performance_metrics['win_rate']:.2f}%)\n"
        summary += f"Losing Trades: {self.performance_metrics['losing_trades']} ({self.performance_metrics['loss_rate']:.2f}%)\n"
        summary += f"Break-Even Trades: {self.performance_metrics['break_even_trades']}\n\n"
        
        summary += f"Average Win: ${self.performance_metrics['avg_win']:,.2f}\n"
        summary += f"Average Loss: ${self.performance_metrics['avg_loss']:,.2f}\n"
        summary += f"Profit Factor: {self.performance_metrics['profit_factor']:.2f}\n"
        
        return summary 