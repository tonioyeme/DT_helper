import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

from app.signals.paired_signals import PairedSignalGenerator, PositionPairManager
from app.data.loader import load_symbol_data

def demonstrate_paired_signals(symbol='SPY', days=1):
    """
    Demonstrate paired entry-exit signals for a given symbol
    
    Args:
        symbol (str): Trading symbol (default: SPY)
        days (int): Number of trading days to analyze
        
    Returns:
        dict: Results containing paired signals
    """
    # Load data
    print(f"Loading {days} days of data for {symbol}...")
    data = load_symbol_data(symbol, days=days)
    
    if data is None or len(data) < 20:
        print(f"Insufficient data for {symbol}")
        return None
    
    # Create signal generator and generate signals
    print("Generating paired entry-exit signals...")
    generator = PairedSignalGenerator()
    results = generator.generate_paired_signals(data)
    
    # Display results
    paired_signals = results['paired_signals']
    signal_df = results['signal_df']
    metrics = results.get('metrics', {})
    
    print(f"\nPerformance Metrics:")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"Average Profit: {metrics.get('avg_profit', 0)*100:.2f}%")
    print(f"Max Profit: {metrics.get('max_profit', 0)*100:.2f}%")
    print(f"Max Loss: {metrics.get('max_loss', 0)*100:.2f}%")
    
    # Return results for further analysis
    return results

def visualize_paired_signals(results, symbol='SPY'):
    """
    Visualize paired entry-exit signals 
    
    Args:
        results (dict): Results from demonstrate_paired_signals function
        symbol (str): Trading symbol
        
    Returns:
        go.Figure: Plotly figure with visualized signals
    """
    if results is None:
        print("No results to visualize")
        return None
    
    data = results.get('raw_signals')
    signal_df = results.get('signal_df')
    paired_signals = results.get('paired_signals')
    
    # Create plot with price data
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'], 
            low=data['low'],
            close=data['close'],
            name='Price'
        )
    )
    
    # Add entry signals
    if 'paired_entry' in signal_df.columns:
        entries = signal_df[signal_df['paired_entry'] == True]
        
        # Add buy signals
        buy_entries = entries[entries['buy_signal'] == True]
        if not buy_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_entries.index,
                    y=buy_entries['close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Buy Entry'
                )
            )
        
        # Add sell signals
        sell_entries = entries[entries['sell_signal'] == True]
        if not sell_entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_entries.index,
                    y=sell_entries['close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Sell Entry'
                )
            )
    
    # Add exit signals
    if 'paired_exit' in signal_df.columns:
        exits = signal_df[signal_df['paired_exit'] == True]
        
        # Add exit signals for buy (long positions)
        if 'exit_buy' in signal_df.columns:
            exit_buys = exits[exits['exit_buy'] == True]
            if not exit_buys.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exit_buys.index,
                        y=exit_buys['close'],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=12,
                            color='orange',
                            line=dict(width=2, color='darkorange')
                        ),
                        name='Exit Long'
                    )
                )
        
        # Add exit signals for sell (short positions)
        if 'exit_sell' in signal_df.columns:
            exit_sells = exits[exits['exit_sell'] == True]
            if not exit_sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exit_sells.index,
                        y=exit_sells['close'],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=12,
                            color='purple',
                            line=dict(width=2, color='indigo')
                        ),
                        name='Exit Short'
                    )
                )
    
    # Connect paired signals with lines
    if not paired_signals.empty:
        for _, pair in paired_signals.iterrows():
            # Draw line connecting entry and exit
            fig.add_shape(
                type="line",
                x0=pair['entry_time'],
                y0=pair['entry_price'],
                x1=pair['exit_time'],
                y1=pair['exit_price'],
                line=dict(
                    color="green" if pair['type'] == 'long' else "red",
                    width=1,
                    dash="dot",
                )
            )
            
            # Add profit annotation
            profit_pct = pair['profit_pct'] * 100
            profit_color = "green" if profit_pct > 0 else "red"
            
            fig.add_annotation(
                x=pair['exit_time'],
                y=pair['exit_price'],
                text=f"{profit_pct:.2f}%",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-20,
                font=dict(
                    color=profit_color
                )
            )
    
    # Update layout
    fig.update_layout(
        title=f'Paired Entry-Exit Signals - {symbol}',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig

if __name__ == "__main__":
    # Example: Generate and visualize paired signals for SPY
    results = demonstrate_paired_signals(symbol='SPY', days=5)
    if results is not None:
        fig = visualize_paired_signals(results)
        # In a Jupyter notebook or Streamlit app, display fig 