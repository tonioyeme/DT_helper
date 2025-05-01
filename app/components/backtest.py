import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
from typing import Dict, List, Tuple, Union, Optional

from app.backtest import (
    BacktestEngine,
    filter_signals_by_strategy,
    filter_signals_by_date,
    create_strategy_comparison,
    create_signal_strength_comparison,
    optimize_position_size,
    get_available_strategies
)

def render_backtest(data, symbol=None):
    """
    Alias for render_backtest_ui for compatibility
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV price data
        symbol (str, optional): Trading symbol
    """
    return render_backtest_ui(data, symbol)

def render_backtest_ui(data, symbol=None):
    """
    Render the backtesting UI component
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV price data
        symbol (str, optional): Trading symbol
    """
    st.header(f"Strategy Backtesting {f'- {symbol}' if symbol else ''}")
    
    if data is None or data.empty:
        st.warning("No data available for backtesting. Please load data first.")
        return
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Data must contain these columns: {required_cols}")
        return
    
    # Check if signals are already generated
    from app.signals import generate_signals
    with st.spinner("Generating signals for backtesting..."):
        signals = generate_signals(data)
    
    # If no signals were generated, show warning
    if signals.empty:
        st.warning("No signals were generated from the data.")
        return
        
    # Get signal counts
    buy_count = signals['buy_signal'].sum()
    sell_count = signals['sell_signal'].sum()
    total_signals = buy_count + sell_count
    
    if total_signals == 0:
        st.warning("No buy or sell signals were generated from the data.")
        return
    
    st.info(f"Found {total_signals} signals ({buy_count} buy, {sell_count} sell) for backtesting.")
    
    # Create tabs for different backtest views
    backtest_tabs = st.tabs([
        "Single Strategy", 
        "Compare Strategies", 
        "Signal Strength Analysis", 
        "Position Sizing",
        "Walk-Forward Validation"
    ])
    
    # Tab 1: Single Strategy Backtest
    with backtest_tabs[0]:
        st.subheader("Backtest a Single Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Strategy selection
            available_strategies = get_available_strategies()
            selected_strategy = st.selectbox(
                "Select Strategy", 
                available_strategies,
                index=0,
                help="Select a specific strategy or use all signals"
            )
            
            # Signal strength filter
            min_strength = st.selectbox(
                "Minimum Signal Strength",
                ["All Signals", "Weak or Better", "Moderate or Better", "Strong or Better", "Very Strong Only"],
                index=0,
                help="Filter signals by minimum strength"
            )
            
            # Map strength selection to filter value
            strength_map = {
                "All Signals": 0,
                "Weak or Better": 1,
                "Moderate or Better": 2,
                "Strong or Better": 3,
                "Very Strong Only": 4
            }
            filter_strength = strength_map[min_strength]
            
        with col2:
            # Backtest parameters
            initial_capital = st.number_input(
                "Initial Capital ($)", 
                min_value=1000.0, 
                max_value=1000000.0,
                value=10000.0,
                step=1000.0,
                help="Starting capital for the backtest"
            )
            
            position_size = st.slider(
                "Position Size (% of Capital)", 
                min_value=1.0, 
                max_value=100.0,
                value=10.0,
                step=1.0,
                help="Percentage of capital to use per trade"
            ) / 100.0
            
            commission = st.slider(
                "Commission (%)", 
                min_value=0.0, 
                max_value=2.0,
                value=0.1,
                step=0.01,
                help="Commission per trade (percentage)"
            )
            
            slippage = st.slider(
                "Slippage (%)", 
                min_value=0.0, 
                max_value=2.0,
                value=0.05,
                step=0.01,
                help="Slippage per trade (percentage)"
            )
        
        # Date range filter
        start_date = data.index[0].strftime('%Y-%m-%d') if hasattr(data.index[0], 'strftime') else str(data.index[0])
        end_date = data.index[-1].strftime('%Y-%m-%d') if hasattr(data.index[-1], 'strftime') else str(data.index[-1])
        
        # Only show date range if we have datetime index
        if hasattr(data.index[0], 'strftime'):
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                selected_start = st.date_input("Start Date", datetime.strptime(start_date, '%Y-%m-%d'))
                start_date = selected_start.strftime('%Y-%m-%d')
            with date_col2:
                selected_end = st.date_input("End Date", datetime.strptime(end_date, '%Y-%m-%d'))
                end_date = selected_end.strftime('%Y-%m-%d')
        
        # Run backtest button
        if st.button("Run Backtest", key="run_single_backtest"):
            with st.spinner("Running backtest..."):
                try:
                    # Filter signals by strategy and date
                    filtered_signals = signals.copy()
                    if selected_strategy != "all":
                        filtered_signals = filter_signals_by_strategy(filtered_signals, selected_strategy)
                    filtered_signals = filter_signals_by_date(filtered_signals, start_date, end_date)
                    
                    # Filter data by date range
                    filtered_data = data.copy()
                    if hasattr(filtered_data.index[0], 'strftime'):
                        filtered_data = filtered_data[(filtered_data.index >= start_date) & (filtered_data.index <= end_date)]
                    
                    # Initialize and run backtest
                    backtest = BacktestEngine(
                        data=filtered_data,
                        signals=filtered_signals,
                        initial_capital=initial_capital,
                        position_size=position_size,
                        commission=commission,
                        slippage=slippage,
                        strategy_name=selected_strategy
                    )
                    
                    metrics = backtest.run(filter_strength=filter_strength)
                    
                    # Display performance summary
                    st.subheader("Performance Summary")
                    summary = backtest.get_performance_summary()
                    st.text(summary)
                    
                    # Display plots in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Equity curve
                        equity_curve_fig = backtest.plot_equity_curve()
                        if equity_curve_fig:
                            st.pyplot(equity_curve_fig)
                    
                    with col2:
                        # Drawdown
                        drawdown_fig = backtest.plot_drawdown()
                        if drawdown_fig:
                            st.pyplot(drawdown_fig)
                    
                    # Monthly returns
                    monthly_returns_fig = backtest.plot_monthly_returns()
                    if monthly_returns_fig:
                        st.pyplot(monthly_returns_fig)
                    
                    # Display trades
                    st.subheader("Trades")
                    trades_df = backtest.get_trades()
                    
                    if not trades_df.empty:
                        # Format trades for display
                        display_trades = trades_df.copy()
                        
                        # Format dates
                        if hasattr(display_trades['entry_date'].iloc[0], 'strftime'):
                            display_trades['entry_date'] = display_trades['entry_date'].dt.strftime('%Y-%m-%d %H:%M')
                            display_trades['exit_date'] = display_trades['exit_date'].dt.strftime('%Y-%m-%d %H:%M')
                        
                        # Format numeric columns
                        display_trades['entry_price'] = display_trades['entry_price'].apply(lambda x: f'${x:,.2f}' if x is not None else 'N/A')
                        display_trades['exit_price'] = display_trades['exit_price'].apply(lambda x: f'${x:,.2f}' if x is not None else 'N/A')
                        display_trades['shares'] = display_trades['shares'].apply(lambda x: f'{x:,.2f}' if x is not None else 'N/A')
                        display_trades['pnl'] = display_trades['pnl'].apply(lambda x: f'${x:,.2f}' if x is not None else 'N/A')
                        display_trades['pnl_pct'] = display_trades['pnl_pct'].apply(lambda x: f'{x:,.2f}%' if x is not None else 'N/A')
                        
                        # Add trade number
                        display_trades.insert(0, 'Trade #', range(1, len(display_trades) + 1))
                        
                        # Display table
                        st.dataframe(display_trades)
                        
                        # Add download button for trades
                        csv = trades_df.to_csv(index=False)
                        st.download_button(
                            label="Download Trades CSV",
                            data=csv,
                            file_name=f"backtest_trades_{selected_strategy}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No trades were executed in this backtest.")
                    
                except Exception as e:
                    st.error(f"Error running backtest: {str(e)}")
    
    # Tab 2: Compare Strategies
    with backtest_tabs[1]:
        st.subheader("Compare Trading Strategies")
        
        # Select strategies to compare
        available_strategies = get_available_strategies()
        selected_strategies = st.multiselect(
            "Select Strategies to Compare", 
            available_strategies,
            default=["all", "ema_vwap", "mm_vol", "strong_signals"],
            help="Select multiple strategies to compare"
        )
        
        # Backtest parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_capital = st.number_input(
                "Initial Capital ($)", 
                min_value=1000.0, 
                max_value=1000000.0,
                value=10000.0,
                step=1000.0,
                key="compare_initial_capital"
            )
        
        with col2:
            position_size = st.slider(
                "Position Size (%)", 
                min_value=1.0, 
                max_value=50.0,
                value=10.0,
                step=1.0,
                key="compare_position_size"
            ) / 100.0
        
        with col3:
            commission = st.slider(
                "Commission (%)", 
                min_value=0.0, 
                max_value=2.0,
                value=0.1,
                step=0.01,
                key="compare_commission"
            )
            
            slippage = st.slider(
                "Slippage (%)", 
                min_value=0.0, 
                max_value=2.0,
                value=0.05,
                step=0.01,
                key="compare_slippage"
            )
        
        # Run comparison button
        if st.button("Compare Strategies", key="run_strategy_comparison") and selected_strategies:
            with st.spinner("Comparing strategies..."):
                try:
                    # Run comparison
                    comparison_results = create_strategy_comparison(
                        data=data,
                        signals=signals,
                        strategies=selected_strategies,
                        initial_capital=initial_capital,
                        position_size=position_size,
                        commission=commission,
                        slippage=slippage
                    )
                    
                    if not comparison_results.empty:
                        # Display results table
                        st.subheader("Strategy Comparison Results")
                        
                        # Format for display
                        display_results = comparison_results.copy()
                        display_results = display_results.round({
                            'total_return': 2,
                            'annualized_return': 2,
                            'win_rate': 2,
                            'loss_rate': 2,
                            'max_drawdown': 2,
                            'profit_factor': 2
                        })
                        
                        # Set index to strategy name
                        display_results.set_index('strategy', inplace=True)
                        
                        # Select columns to display
                        display_cols = [
                            'total_return', 'annualized_return', 'max_drawdown',
                            'total_trades', 'win_rate', 'profit_factor',
                            'avg_win', 'avg_loss'
                        ]
                        
                        # Rename columns for display
                        column_names = {
                            'total_return': 'Total Return (%)',
                            'annualized_return': 'Annual Return (%)',
                            'max_drawdown': 'Max Drawdown (%)',
                            'total_trades': 'Total Trades',
                            'win_rate': 'Win Rate (%)',
                            'profit_factor': 'Profit Factor',
                            'avg_win': 'Avg Win ($)',
                            'avg_loss': 'Avg Loss ($)'
                        }
                        
                        # Display formatted table
                        st.dataframe(display_results[display_cols].rename(columns=column_names))
                        
                        # Create bar chart of returns
                        st.subheader("Performance Comparison")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        colors = ['green' if x >= 0 else 'red' for x in display_results['total_return']]
                        display_results['total_return'].plot(kind='bar', ax=ax, color=colors)
                        ax.set_title("Total Return by Strategy")
                        ax.set_ylabel("Return (%)")
                        ax.grid(True, axis='y')
                        
                        # Add return values on bars
                        for i, v in enumerate(display_results['total_return']):
                            ax.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", 
                                   ha='center', va='bottom' if v >= 0 else 'top', 
                                   fontweight='bold')
                        
                        st.pyplot(fig)
                        
                        # Create table of drawdowns
                        fig2, ax2 = plt.subplots(figsize=(12, 6))
                        display_results['max_drawdown'].plot(kind='bar', ax=ax2, color='red')
                        ax2.set_title("Maximum Drawdown by Strategy")
                        ax2.set_ylabel("Drawdown (%)")
                        ax2.grid(True, axis='y')
                        ax2.invert_yaxis()  # Invert to show drawdowns as negative
                        
                        # Add drawdown values on bars
                        for i, v in enumerate(display_results['max_drawdown']):
                            ax2.text(i, -v - 1, f"{v:.1f}%", ha='center', va='top', fontweight='bold')
                        
                        st.pyplot(fig2)
                        
                        # Add download button for comparison results
                        csv = comparison_results.to_csv(index=False)
                        st.download_button(
                            label="Download Comparison CSV",
                            data=csv,
                            file_name="strategy_comparison.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No comparison results available. Please check if selected strategies generated any trades.")
                
                except Exception as e:
                    st.error(f"Error comparing strategies: {str(e)}")
    
    # Tab 3: Signal Strength Analysis
    with backtest_tabs[2]:
        st.subheader("Signal Strength Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Backtest parameters
            initial_capital = st.number_input(
                "Initial Capital ($)", 
                min_value=1000.0, 
                max_value=1000000.0,
                value=10000.0,
                step=1000.0,
                key="strength_initial_capital"
            )
            
            position_size = st.slider(
                "Position Size (%)", 
                min_value=1.0, 
                max_value=50.0,
                value=10.0,
                step=1.0,
                key="strength_position_size"
            ) / 100.0
        
        with col2:
            commission = st.slider(
                "Commission (%)", 
                min_value=0.0, 
                max_value=2.0,
                value=0.1,
                step=0.01,
                key="strength_commission"
            )
            
            slippage = st.slider(
                "Slippage (%)", 
                min_value=0.0, 
                max_value=2.0,
                value=0.05,
                step=0.01,
                key="strength_slippage"
            )
        
        # Run analysis button
        if st.button("Analyze Signal Strength", key="run_strength_analysis"):
            with st.spinner("Analyzing signal strength..."):
                try:
                    # Run comparison
                    strength_results = create_signal_strength_comparison(
                        data=data,
                        signals=signals,
                        initial_capital=initial_capital,
                        position_size=position_size,
                        commission=commission,
                        slippage=slippage
                    )
                    
                    if not strength_results.empty:
                        # Display results table
                        st.subheader("Signal Strength Results")
                        
                        # Format for display
                        display_results = strength_results.copy()
                        display_results = display_results.round({
                            'total_return': 2,
                            'annualized_return': 2,
                            'win_rate': 2,
                            'loss_rate': 2,
                            'max_drawdown': 2,
                            'profit_factor': 2
                        })
                        
                        # Set index to strategy name
                        display_results.set_index('strategy', inplace=True)
                        
                        # Select columns to display
                        display_cols = [
                            'total_return', 'annualized_return', 'max_drawdown',
                            'total_trades', 'win_rate', 'profit_factor',
                            'avg_win', 'avg_loss'
                        ]
                        
                        # Rename columns for display
                        column_names = {
                            'total_return': 'Total Return (%)',
                            'annualized_return': 'Annual Return (%)',
                            'max_drawdown': 'Max Drawdown (%)',
                            'total_trades': 'Total Trades',
                            'win_rate': 'Win Rate (%)',
                            'profit_factor': 'Profit Factor',
                            'avg_win': 'Avg Win ($)',
                            'avg_loss': 'Avg Loss ($)'
                        }
                        
                        # Display formatted table
                        st.dataframe(display_results[display_cols].rename(columns=column_names))
                        
                        # Create bar chart of returns
                        st.subheader("Performance by Signal Strength")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        colors = ['green' if x >= 0 else 'red' for x in display_results['total_return']]
                        display_results['total_return'].plot(kind='bar', ax=ax, color=colors)
                        ax.set_title("Total Return by Signal Strength")
                        ax.set_ylabel("Return (%)")
                        ax.grid(True, axis='y')
                        
                        # Add return values on bars
                        for i, v in enumerate(display_results['total_return']):
                            ax.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", 
                                   ha='center', va='bottom' if v >= 0 else 'top', 
                                   fontweight='bold')
                        
                        st.pyplot(fig)
                        
                        # Create bar chart of win rates
                        fig2, ax2 = plt.subplots(figsize=(12, 6))
                        display_results['win_rate'].plot(kind='bar', ax=ax2, color='blue')
                        ax2.set_title("Win Rate by Signal Strength")
                        ax2.set_ylabel("Win Rate (%)")
                        ax2.grid(True, axis='y')
                        
                        # Add win rate values on bars
                        for i, v in enumerate(display_results['win_rate']):
                            ax2.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')
                        
                        st.pyplot(fig2)
                    else:
                        st.warning("No strength analysis results available.")
                
                except Exception as e:
                    st.error(f"Error analyzing signal strength: {str(e)}")
    
    # Tab 4: Position Sizing
    with backtest_tabs[3]:
        st.subheader("Position Size Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Strategy selection
            available_strategies = get_available_strategies()
            selected_strategy = st.selectbox(
                "Select Strategy",
                available_strategies,
                index=0,
                key="sizing_strategy",
                help="Select a strategy to optimize position sizing"
            )
            
            # Signal strength filter
            min_strength = st.selectbox(
                "Minimum Signal Strength",
                ["All Signals", "Weak or Better", "Moderate or Better", "Strong or Better", "Very Strong Only"],
                index=0,
                key="sizing_strength",
                help="Filter signals by minimum strength"
            )
            
            # Map strength selection to filter value
            strength_map = {
                "All Signals": 0,
                "Weak or Better": 1,
                "Moderate or Better": 2,
                "Strong or Better": 3,
                "Very Strong Only": 4
            }
            filter_strength = strength_map[min_strength]
        
        with col2:
            # Backtest parameters
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000.0,
                max_value=1000000.0,
                value=10000.0,
                step=1000.0,
                key="sizing_initial_capital"
            )
            
            commission = st.slider(
                "Commission (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.1,
                step=0.01,
                key="sizing_commission"
            )
            
            slippage = st.slider(
                "Slippage (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.05,
                step=0.01,
                key="sizing_slippage"
            )
        
        # Position size range
        st.subheader("Position Size Range")
        
        min_size = st.slider(
            "Minimum Position Size (%)",
            min_value=1.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            key="min_position_size"
        )
        
        max_size = st.slider(
            "Maximum Position Size (%)",
            min_value=min_size,
            max_value=100.0,
            value=min(min_size + 25.0, 100.0),
            step=1.0,
            key="max_position_size"
        )
        
        step_size = st.slider(
            "Step Size (%)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=1.0,
            key="step_size"
        )
        
        # Create position size range
        position_sizes = [size / 100.0 for size in np.arange(min_size, max_size + step_size, step_size)]
        
        # Run optimization button
        if st.button("Optimize Position Size", key="run_position_optimization"):
            with st.spinner("Optimizing position size..."):
                try:
                    # Run optimization
                    optimization_results = optimize_position_size(
                        data=data,
                        signals=signals,
                        position_sizes=position_sizes,
                        strategy=selected_strategy,
                        filter_strength=filter_strength,
                        initial_capital=initial_capital,
                        commission=commission,
                        slippage=slippage
                    )
                    
                    if not optimization_results.empty:
                        # Display results table
                        st.subheader("Position Size Optimization Results")
                        
                        # Format for display
                        display_results = optimization_results.copy()
                        display_results = display_results.round({
                            'total_return': 2,
                            'annualized_return': 2,
                            'win_rate': 2,
                            'loss_rate': 2,
                            'max_drawdown': 2,
                            'profit_factor': 2,
                            'position_size': 2
                        })
                        
                        # Set index to strategy name
                        display_results.set_index('strategy', inplace=True)
                        
                        # Select columns to display
                        display_cols = [
                            'position_size', 'total_return', 'annualized_return', 'max_drawdown',
                            'total_trades', 'win_rate', 'profit_factor'
                        ]
                        
                        # Rename columns for display
                        column_names = {
                            'position_size': 'Position Size',
                            'total_return': 'Total Return (%)',
                            'annualized_return': 'Annual Return (%)',
                            'max_drawdown': 'Max Drawdown (%)',
                            'total_trades': 'Total Trades',
                            'win_rate': 'Win Rate (%)',
                            'profit_factor': 'Profit Factor'
                        }
                        
                        # Display formatted table
                        st.dataframe(display_results[display_cols].rename(columns=column_names))
                        
                        # Create line chart of returns vs position size
                        st.subheader("Returns vs Position Size")
                        
                        fig, ax1 = plt.subplots(figsize=(12, 6))
                        
                        # Plot metrics
                        ax1.plot(display_results['position_size'] * 100, display_results['total_return'], 'b-', label='Total Return (%)')
                        ax1.set_xlabel('Position Size (%)')
                        ax1.set_ylabel('Total Return (%)', color='b')
                        ax1.tick_params(axis='y', labelcolor='b')
                        ax1.grid(True)
                        
                        # Create a secondary y-axis for drawdown (inverted)
                        ax2 = ax1.twinx()
                        ax2.plot(display_results['position_size'] * 100, -display_results['max_drawdown'], 'r--', label='Max Drawdown (%)')
                        ax2.set_ylabel('Max Drawdown (%)', color='r')
                        ax2.tick_params(axis='y', labelcolor='r')
                        ax2.invert_yaxis()  # Invert to show drawdown as negative
                        
                        # Add a title
                        plt.title(f"Return vs Risk for {selected_strategy} Strategy")
                        
                        # Add a legend
                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
                        
                        st.pyplot(fig)
                        
                        # Find the best position size based on different metrics
                        best_return_idx = display_results['total_return'].idxmax()
                        best_return_size = display_results.loc[best_return_idx, 'position_size']
                        best_return = display_results.loc[best_return_idx, 'total_return']
                        
                        # Calculate the risk-adjusted return (return / drawdown)
                        display_results['risk_adjusted'] = display_results['total_return'] / display_results['max_drawdown']
                        best_risk_adj_idx = display_results['risk_adjusted'].idxmax()
                        best_risk_adj_size = display_results.loc[best_risk_adj_idx, 'position_size']
                        best_risk_adj_return = display_results.loc[best_risk_adj_idx, 'total_return']
                        best_risk_adj_drawdown = display_results.loc[best_risk_adj_idx, 'max_drawdown']
                        
                        # Display recommendations
                        st.subheader("Position Size Recommendations")
                        
                        st.info(f"Best Position Size for Maximum Return: {best_return_size:.0%} "
                               f"(Return: {best_return:.2f}%)")
                        
                        st.info(f"Best Position Size for Risk-Adjusted Return: {best_risk_adj_size:.0%} "
                               f"(Return: {best_risk_adj_return:.2f}%, Drawdown: {best_risk_adj_drawdown:.2f}%)")
                        
                    else:
                        st.warning("No optimization results available.")
                
                except Exception as e:
                    st.error(f"Error optimizing position size: {str(e)}")
    
    # Tab 5: Walk-Forward Validation (add after Tab 4)
    with backtest_tabs[4]:
        st.subheader("Walk-Forward Validation")
        st.markdown("""
        Walk-forward validation tests the strategy's robustness across multiple time periods, 
        helping to identify potential overfitting and confirm strategy effectiveness.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Strategy selection
            available_strategies = get_available_strategies()
            selected_strategy = st.selectbox(
                "Select Strategy",
                available_strategies,
                index=0,
                key="wf_strategy",
                help="Select a strategy to validate"
            )
            
            # Number of splits
            n_splits = st.slider(
                "Number of Testing Periods",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                key="wf_splits",
                help="More periods give more comprehensive validation"
            )
            
            # Train ratio
            train_ratio = st.slider(
                "Training Data Ratio",
                min_value=0.5,
                max_value=0.9,
                value=0.7,
                step=0.05,
                key="wf_train_ratio",
                help="Portion of each period used for training (vs testing)"
            )
        
        with col2:
            # Backtest parameters
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000.0,
                max_value=1000000.0,
                value=10000.0,
                step=1000.0,
                key="wf_initial_capital"
            )
            
            position_size = st.slider(
                "Position Size (%)",
                min_value=1.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                key="wf_position_size"
            ) / 100.0
            
            commission = st.slider(
                "Commission (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.1,
                step=0.01,
                key="wf_commission"
            )
            
            slippage = st.slider(
                "Slippage (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.05,
                step=0.01,
                key="wf_slippage"
            )
        
        # Run validation button
        if st.button("Run Walk-Forward Validation", key="run_wf_validation"):
            with st.spinner("Running walk-forward validation (this may take a while)..."):
                try:
                    # Run walk-forward test
                    wf_results = walk_forward_test(
                        data=data,
                        strategy=selected_strategy if selected_strategy != "all" else None,
                        n_splits=n_splits,
                        train_ratio=train_ratio,
                        initial_capital=initial_capital,
                        position_size=position_size,
                        commission=commission,
                        slippage=slippage
                    )
                    
                    # Get results and summary
                    results = wf_results['results']
                    summary = wf_results['summary']
                    
                    # Display results if we have any
                    if results:
                        # Display overall robustness score
                        robustness = summary['robustness_score'] * 100
                        if robustness >= 70:
                            score_color = "green"
                            robustness_text = "High"
                        elif robustness >= 50:
                            score_color = "orange"
                            robustness_text = "Medium"
                        else:
                            score_color = "red"
                            robustness_text = "Low"
                        
                        st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: {score_color}; color: white; font-size: 20px; font-weight: bold; text-align: center; margin-bottom: 20px;">
                            Strategy Robustness: {robustness_text} ({robustness:.0f}%)
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create a table of results
                        results_df = pd.DataFrame(results)
                        
                        # Format dates if they're datetime objects
                        for col in ['train_start', 'train_end', 'test_start', 'test_end']:
                            if col in results_df.columns:
                                results_df[col] = results_df[col].apply(
                                    lambda x: x.strftime('%Y-%m-%d %H:%M') if hasattr(x, 'strftime') else str(x)
                                )
                        
                        # Format numerical columns
                        results_df['train_sharpe'] = results_df['train_sharpe'].round(2)
                        results_df['test_sharpe'] = results_df['test_sharpe'].round(2)
                        results_df['train_return'] = results_df['train_return'].round(2)
                        results_df['test_return'] = results_df['test_return'].round(2)
                        
                        # Add a status column
                        results_df['status'] = results_df.apply(
                            lambda x: "✅ Consistent" if x['consistency'] == 1 else "❌ Inconsistent", 
                            axis=1
                        )
                        
                        # Display the results table
                        st.subheader("Results by Period")
                        st.dataframe(results_df[['split', 'train_start', 'train_end', 'test_start', 'test_end', 
                                              'train_return', 'test_return', 'train_sharpe', 'test_sharpe', 'status']])
                        
                        # Create graphs for visualization
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                        
                        # Plot returns
                        ax1.set_title("Returns by Period")
                        ax1.plot(summary['splits'], summary['train_return'], 'b-', label='Train Return (%)')
                        ax1.plot(summary['splits'], summary['test_return'], 'r--', label='Test Return (%)')
                        ax1.set_xlabel('Period')
                        ax1.set_ylabel('Return (%)')
                        ax1.legend()
                        ax1.grid(True)
                        
                        # Plot Sharpe ratios
                        ax2.set_title("Sharpe Ratio by Period")
                        ax2.plot(summary['splits'], summary['train_sharpe'], 'g-', label='Train Sharpe')
                        ax2.plot(summary['splits'], summary['test_sharpe'], 'm--', label='Test Sharpe')
                        ax2.set_xlabel('Period')
                        ax2.set_ylabel('Sharpe Ratio')
                        ax2.legend()
                        ax2.grid(True)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Interpretation
                        st.subheader("Interpretation")
                        
                        avg_train_return = np.mean(summary['train_return'])
                        avg_test_return = np.mean(summary['test_return'])
                        avg_train_sharpe = np.mean(summary['train_sharpe'])
                        avg_test_sharpe = np.mean(summary['test_sharpe'])
                        
                        # Calculate the difference between train and test
                        return_degradation = ((avg_train_return - avg_test_return) / abs(avg_train_return)) * 100 if avg_train_return != 0 else 0
                        sharpe_degradation = ((avg_train_sharpe - avg_test_sharpe) / avg_train_sharpe) * 100 if avg_train_sharpe != 0 else 0
                        
                        st.markdown(f"""
                        - Average Training Return: **{avg_train_return:.2f}%**
                        - Average Testing Return: **{avg_test_return:.2f}%**
                        - Performance Degradation: **{return_degradation:.1f}%**
                        
                        - Average Training Sharpe: **{avg_train_sharpe:.2f}**
                        - Average Testing Sharpe: **{avg_test_sharpe:.2f}**
                        - Sharpe Degradation: **{sharpe_degradation:.1f}%**
                        """)
                        
                        # Conclusions based on results
                        if robustness >= 70 and return_degradation < 30:
                            st.success("The strategy shows good robustness and consistency across different time periods.")
                        elif robustness >= 50 and return_degradation < 50:
                            st.warning("The strategy shows moderate robustness, but there is some degradation in out-of-sample performance.")
                        else:
                            st.error("The strategy shows signs of overfitting. Performance degradation in out-of-sample periods is significant.")
                        
                        # Recommendations
                        st.subheader("Recommendations")
                        if robustness < 50:
                            st.markdown("""
                            - Consider simplifying the strategy to reduce potential overfitting
                            - Test with different parameters or additional filters
                            - Consider the market conditions during periods of inconsistency
                            """)
                        elif return_degradation > 50:
                            st.markdown("""
                            - The strategy shows directional consistency but significant performance degradation
                            - Consider using more conservative position sizing in live trading
                            - Implement additional risk management rules
                            """)
                        else:
                            st.markdown("""
                            - The strategy shows good robustness and can be considered for live trading
                            - Continue to monitor performance and compare to walk-forward benchmarks
                            - Consider further optimization of entry/exit criteria to improve consistency
                            """)
                    else:
                        st.warning("No validation results available. This could be due to insufficient data for the selected number of splits.")
                
                except Exception as e:
                    st.error(f"Error during walk-forward validation: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc()) 