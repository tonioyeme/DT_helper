import streamlit as st

def render_dashboard(data=None, symbol=None):
    """
    Render the main dashboard UI layout
    
    Args:
        data (pd.DataFrame, optional): DataFrame with OHLC data
        symbol (str, optional): Trading symbol
    """
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Charts", "Signals", "Risk Analysis"])
    
    # Get selected indicators from sidebar
    selected_indicators = []
    
    # Check if data exists
    if data is None:
        with tab1:
            st.info("Enter a symbol and fetch data to see charts.")
        with tab2:
            st.info("No signal data available. Fetch data first.")
        with tab3:
            st.info("Enter trade details to see risk analysis.")
    else:
        with tab1:
            st.subheader(f"Price Chart - {symbol}")
            # Chart visualization code would go here
        
        with tab2:
            st.subheader(f"Trading Signals - {symbol}")
            # Signal table would go here
            
        with tab3:
            st.subheader("Risk Calculator")
            # Risk calculator would go here
    
    # Return configuration dictionary
    return {
        "indicators": selected_indicators
    } 