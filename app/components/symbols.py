import streamlit as st
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
import yfinance as yf

def render_symbol_selection(
    default_symbols: List[str] = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
    allow_multiple: bool = False
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Render a symbol selection widget in Streamlit
    
    Args:
        default_symbols: List of default symbols to show
        allow_multiple: Whether to allow selecting multiple symbols
        
    Returns:
        Tuple containing the selected symbol and any additional metadata
    """
    st.sidebar.subheader("Symbol Selection")
    
    # Option to enter a custom symbol
    custom_symbol = st.sidebar.text_input("Enter Symbol:", "")
    
    if custom_symbol:
        selected_symbol = custom_symbol.strip().upper()
    else:
        # Select from default symbols
        selected_symbol = st.sidebar.selectbox(
            "Select Symbol:",
            options=default_symbols,
            index=0
        )
    
    # Get basic symbol info
    symbol_info = None
    try:
        if selected_symbol:
            ticker = yf.Ticker(selected_symbol)
            info = ticker.info
            
            # Create simplified info dict with most relevant fields
            symbol_info = {
                "symbol": selected_symbol,
                "name": info.get("shortName", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("currentPrice", 0),
                "volume": info.get("volume", 0),
                "exchange": info.get("exchange", "Unknown")
            }
            
            # Display basic info
            st.sidebar.markdown(f"**{symbol_info['name']}** ({selected_symbol})")
            st.sidebar.markdown(f"Sector: {symbol_info['sector']}")
            st.sidebar.markdown(f"Price: ${symbol_info['current_price']}")
    except Exception as e:
        st.sidebar.warning(f"Could not fetch info for {selected_symbol}: {str(e)}")
    
    return selected_symbol, symbol_info

def render_multi_symbol_selection(
    default_symbols: List[str] = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
    max_symbols: int = 5
) -> List[str]:
    """
    Render a multi-symbol selection widget in Streamlit
    
    Args:
        default_symbols: List of default symbols to show
        max_symbols: Maximum number of symbols that can be selected
        
    Returns:
        List of selected symbols
    """
    st.sidebar.subheader("Symbol Selection")
    
    # Option to enter custom symbols
    custom_symbols = st.sidebar.text_input("Enter Symbols (comma-separated):", "")
    
    if custom_symbols:
        # Process custom symbols
        symbols = [s.strip().upper() for s in custom_symbols.split(",")]
        # Limit to max_symbols
        if len(symbols) > max_symbols:
            st.sidebar.warning(f"Limited to {max_symbols} symbols. Only using the first {max_symbols}.")
            symbols = symbols[:max_symbols]
    else:
        # Select from default symbols
        symbols = st.sidebar.multiselect(
            "Select Symbols:",
            options=default_symbols,
            default=[default_symbols[0]]
        )
        
        # Limit to max_symbols
        if len(symbols) > max_symbols:
            st.sidebar.warning(f"Limited to {max_symbols} symbols. Only using the first {max_symbols}.")
            symbols = symbols[:max_symbols]
    
    return symbols 