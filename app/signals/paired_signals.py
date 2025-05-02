import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import pytz
from datetime import datetime, timedelta

from app.signals.signal_functions import generate_standard_signals
from app.signals.filters import apply_atr_filter_to_signals, get_spy_atr_filter_config
from app.signals.exit_signals import ImprovedExitSignalGenerator

class PositionPairManager:
    """
    Manages and pairs entry and exit signals into complete trades, tracking performance
    """
    def __init__(self):
        self.open_positions = []
        self.closed_positions = []
        self.current_position = None
        self.min_profit = 0.0015  # 0.15% minimum profit (covers fees + spread)

    def process_signals(self, data, entry_signals, exit_signals):
        """
        Pair entry and exit signals into complete trades
        
        Args:
            data (pd.DataFrame): OHLCV price data
            entry_signals (pd.DataFrame): DataFrame with entry signals
            exit_signals (pd.DataFrame): DataFrame with exit signals
            
        Returns:
            List[Dict]: List of paired trades with entry and exit information
        """
        paired = []
        
        for idx in data.index:
            # Check for new entry signals
            if idx in entry_signals.index and idx in exit_signals.index:
                # Handle new entries if no position is open
                if entry_signals.at[idx, 'buy_signal'] and not self.current_position:
                    self.current_position = {
                        'entry_type': 'long',
                        'entry_price': data.at[idx, 'close'],
                        'entry_time': idx,
                        'exit_price': None,
                        'exit_time': None
                    }
                    
                elif entry_signals.at[idx, 'sell_signal'] and not self.current_position:
                    self.current_position = {
                        'entry_type': 'short',
                        'entry_price': data.at[idx, 'close'],
                        'entry_time': idx,
                        'exit_price': None,
                        'exit_time': None
                    }
                
                # Check for exit signals for current position
                if self.current_position:
                    exit_condition = (
                        (self.current_position['entry_type'] == 'long' and exit_signals.at[idx, 'exit_buy']) or
                        (self.current_position['entry_type'] == 'short' and exit_signals.at[idx, 'exit_sell'])
                    )
                    
                    if exit_condition:
                        # Calculate profit and validate
                        exit_price = data.at[idx, 'close']
                        profit_pct = self._calculate_profit(
                            self.current_position['entry_price'], 
                            exit_price,
                            self.current_position['entry_type']
                        )
                        
                        if profit_pct >= self.min_profit:
                            self.current_position['exit_price'] = exit_price
                            self.current_position['exit_time'] = idx
                            self.current_position['profit_pct'] = profit_pct
                            
                            paired.append(self.current_position)
                            self.closed_positions.append(self.current_position)
                            self.current_position = None
        
        return paired

    def _calculate_profit(self, entry, exit, position_type):
        """
        Calculate percentage profit with position type awareness
        
        Args:
            entry (float): Entry price
            exit (float): Exit price
            position_type (str): 'long' or 'short'
            
        Returns:
            float: Percentage profit (positive is gain, negative is loss)
        """
        if position_type == 'long':
            return (exit - entry) / entry
        else:  # short
            return (entry - exit) / entry
            
    def get_performance_metrics(self):
        """
        Calculate performance metrics for all closed positions
        
        Returns:
            Dict: Performance metrics
        """
        if not self.closed_positions:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "max_profit": 0.0,
                "max_loss": 0.0
            }
            
        profits = [pos['profit_pct'] for pos in self.closed_positions]
        winning_trades = sum(1 for p in profits if p > 0)
        
        return {
            "total_trades": len(self.closed_positions),
            "win_rate": winning_trades / len(self.closed_positions) if self.closed_positions else 0,
            "avg_profit": sum(profits) / len(profits) if profits else 0,
            "max_profit": max(profits) if profits else 0,
            "max_loss": min(profits) if profits else 0
        }

class PairedSignalGenerator:
    """
    Generates validated entry-exit pairs for trading signals
    """
    def __init__(self):
        self.position_manager = PositionPairManager()
        self.min_holding = pd.Timedelta('2min')  # Prevent immediate exits
        
    def generate_paired_signals(self, data):
        """
        Generate validated entry-exit pairs
        
        Args:
            data (pd.DataFrame): OHLCV price data
            
        Returns:
            Dict: Results containing paired signals and other signal information
        """
        # Generate raw signals
        raw_signals = generate_standard_signals(data)
        
        # Apply ATR filter
        filtered_signals = apply_atr_filter_to_signals(
            data, 
            raw_signals, 
            **get_spy_atr_filter_config()
        )
        
        # Generate exit signals
        exit_gen = ImprovedExitSignalGenerator()
        exit_signals = exit_gen.generate_exit_signals(data, filtered_signals)
        
        # Pair signals
        pairs = self.position_manager.process_signals(data, filtered_signals, exit_signals)
        
        # Create paired signal DataFrame
        paired_df = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'type', 
            'entry_price', 'exit_price', 'profit_pct'
        ])
        
        for pair in pairs:
            # Convert to proper format for the DataFrame
            pair_record = {
                'entry_time': pair['entry_time'],
                'exit_time': pair['exit_time'],
                'type': pair['entry_type'],
                'entry_price': pair['entry_price'],
                'exit_price': pair['exit_price'],
                'profit_pct': pair['profit_pct']
            }
            
            # Validate minimum holding period
            if (pair['exit_time'] - pair['entry_time']) < self.min_holding:
                continue
                
            paired_df = pd.concat([paired_df, pd.DataFrame([pair_record])], ignore_index=True)
        
        # Merge with original signals
        final_signals = filtered_signals.copy()
        final_signals['paired_entry'] = False
        final_signals['paired_exit'] = False
        
        for _, pair in paired_df.iterrows():
            if pair['entry_time'] in final_signals.index:
                final_signals.loc[pair['entry_time'], 'paired_entry'] = True
            if pair['exit_time'] in final_signals.index:
                final_signals.loc[pair['exit_time'], 'paired_exit'] = True
        
        # Calculate performance metrics
        metrics = self.position_manager.get_performance_metrics()
        
        return {
            'paired_signals': paired_df,
            'signal_df': final_signals,
            'raw_signals': raw_signals,
            'exit_signals': exit_signals,
            'metrics': metrics
        } 