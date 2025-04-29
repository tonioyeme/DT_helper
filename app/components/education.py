import streamlit as st

def render_indicator_combinations():
    """
    Render educational content about effective indicator combinations for day trading
    """
    st.header("Effective Indicator Combinations")
    
    st.write("""
    Combining indicators strategically can provide confirmation signals and filter out false positives. 
    The key is to select complementary indicators that address different aspects of market behavior.
    Here are some powerful combinations specifically for day traders:
    """)
    
    # First combination
    with st.expander("1. Moving Average + Momentum Oscillator", expanded=True):
        st.write("""
        The combination of moving averages (particularly the 5-8-13 SMA setup) with a momentum oscillator like RSI 
        creates a robust framework for day trading. The moving averages establish the trend direction while the RSI 
        identifies potential reversal points when the market becomes overextended.
        
        **Implementation:** Enter long when price is above the 5-8-13 SMA alignment (stacked in ascending order) and 
        RSI rebounds from oversold conditions (below 30). Enter short when price is below the SMAs (stacked in descending order) 
        and RSI drops from overbought readings (above 70).
        """)
        
        # Add example image or diagram here if available
        # st.image("path_to_image.png", caption="Moving Average + RSI Example")
    
    # Second combination
    with st.expander("2. Bollinger Bands + RSI/Stochastic", expanded=True):
        st.write("""
        This combination is particularly effective for identifying potential reversal points. Bollinger Bands show volatility 
        and potential support/resistance levels, while oscillators confirm whether the market is overextended.
        
        **Implementation:** Look for buying opportunities when price touches the lower Bollinger Band while RSI or Stochastic 
        shows oversold conditions. Similarly, consider selling when price reaches the upper band while oscillators indicate 
        overbought conditions.
        """)
        
        # Add example image or diagram here if available
        # st.image("path_to_image.png", caption="Bollinger Bands + RSI Example")
    
    # Third combination
    with st.expander("3. MACD + RSI", expanded=True):
        st.write("""
        This powerful combination helps confirm trend direction and momentum. MACD provides trend information while RSI 
        adds insight about momentum and potential reversals.
        
        **Implementation:** The strongest signals occur when both indicators align—for example, MACD crossing above its 
        signal line while RSI moves above 50 generates a more reliable buy signal than either indicator alone.
        """)
        
        # Add example image or diagram here if available
        # st.image("path_to_image.png", caption="MACD + RSI Example")
    
    # Fourth combination
    with st.expander("4. Volume + Price Action", expanded=True):
        st.write("""
        Combining volume analysis with price action patterns creates particularly robust signals. Volume confirms the 
        strength and legitimacy of price movements.
        
        **Implementation:** Focus on price action patterns that are accompanied by significant volume increases. For instance, 
        a breakout from a rectangle pattern with above-average volume has a higher probability of success than one occurring 
        on low volume.
        """)
        
        # Add example image or diagram here if available
        # st.image("path_to_image.png", caption="Volume + Price Action Example")
    
    # Add a note about practice
    st.info("""
    **Note:** These combinations work best when you've practiced with them extensively and understand their nuances.
    Test them in a paper trading environment before using them for real trades.
    """)

def render_ema_clouds():
    """
    Render educational content about EMA Clouds for day trading
    """
    st.header("EMA Clouds: Dynamic Support/Resistance Zones")
    
    st.write("""
    EMA Clouds, popularized by trader Ripster, visualize the interaction between two EMAs by shading the area between them. 
    This system transforms traditional moving averages into dynamic support/resistance zones that adapt to intraday volatility.
    """)
    
    # Conceptual Foundation
    st.subheader("Conceptual Foundation")
    st.write("""
    EMA Clouds create visual zones on your chart that help identify trend direction, potential reversal points, and 
    support/resistance levels. The cloud forms between two EMAs of different periods, with the space between them 
    shaded to create a "cloud" effect.
    """)
    
    # Key EMA Combinations
    st.subheader("Key EMA Combinations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 5-13 EMA Cloud")
        st.write("""
        Acts as a fluid trendline for day trading, responding rapidly to price changes. 
        This combination is especially useful for short-term momentum trades.
        """)
    
    with col2:
        st.markdown("##### 8-9 EMA Cloud")
        st.write("""
        Identifies pullback opportunities during trending markets. 
        The narrow cloud creates precise entry points during established trends.
        """)
    
    with col3:
        st.markdown("##### 34-50 EMA Cloud")
        st.write("""
        Establishes higher-timeframe bias - prices above this cloud indicate bullish momentum, 
        while prices below suggest bearish control.
        """)
    
    # Strategic Applications
    st.subheader("Trading Applications")
    
    with st.expander("Cloud Color Interpretation", expanded=True):
        st.write("""
        The cloud's color provides immediate visual cues about market sentiment:
        - **Green cloud**: Bullish alignment (faster EMA above slower EMA)
        - **Red cloud**: Bearish alignment (faster EMA below slower EMA)
        
        These color changes can signal potential trend reversals when they occur.
        """)
    
    with st.expander("Bounce Trades", expanded=True):
        st.write("""
        When price approaches the cloud boundaries, watch for potential reversals:
        - Price approaching from below and bouncing off the cloud bottom suggests resistance
        - Price approaching from above and bouncing off the cloud top indicates support
        
        Backtests show approximately 73% success rate for bounce trades when combined with confirming indicators.
        """)
    
    with st.expander("Breakout Trades", expanded=True):
        st.write("""
        When price moves through the cloud with conviction:
        - Upside breakouts: Price closes above the top of the cloud with increased volume
        - Downside breakouts: Price closes below the bottom of the cloud with increased volume
        
        These breakouts often signal the beginning of new trends in the breakout direction.
        """)
    
    # Implementation code
    with st.expander("Sample Implementation", expanded=True):
        st.code("""
# Calculate EMA Cloud
def calculate_ema_cloud(df, fast_period=5, slow_period=13):
    # Calculate EMAs
    df['fast_ema'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['slow_ema'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Determine cloud color/bias
    df['cloud_bullish'] = df['fast_ema'] > df['slow_ema']
    
    return df['fast_ema'], df['slow_ema'], df['cloud_bullish']
        """, language="python")

def render_vwap():
    """
    Render educational content about VWAP for day trading
    """
    st.header("Volume-Weighted Average Price (VWAP): The Institutional Benchmark")
    
    st.write("""
    VWAP is one of the most important indicators for day traders, especially those focusing on stocks. 
    It provides a volume-adjusted perspective of price movement throughout the trading session, offering 
    insights into institutional activity and true market value.
    """)
    
    # Core Calculations
    st.subheader("Core Calculations")
    st.write("""
    VWAP aggregates price and volume data to reflect the "true" average price for a session:
    
    **VWAP = cumulative sum(typical price * volume) / cumulative sum(volume)**
    
    Where typical price = (high + low + close) / 3
    """)
    
    # Formula visualization
    st.latex(r'''
    VWAP = \frac{\sum_{i=1}^{n} (H_i + L_i + C_i) \times V_i / 3}{\sum_{i=1}^{n} V_i}
    ''')
    
    # Strategic Applications
    st.subheader("Strategic Applications")
    
    with st.expander("1. Trend Confirmation", expanded=True):
        st.write("""
        VWAP serves as an intraday trend indicator with statistical significance:
        - **Prices above VWAP**: Signal bullish intraday bias (68% continuation rate)
        - **Prices below VWAP**: Indicate bearish control (62% continuation rate)
        
        Many institutional algorithms use VWAP as a trigger for buy/sell decisions, making it a self-fulfilling indicator.
        """)
    
    with st.expander("2. Mean Reversion", expanded=True):
        st.write("""
        VWAP acts as a magnet throughout the trading day:
        - 87% of SPY pullbacks to VWAP during trends result in continuation
        - Combining VWAP with standard deviation bands (±1σ, ±2σ) creates powerful overbought/oversold indicators
        
        Mean reversion trades are especially effective in the final two hours of the trading session as prices 
        tend to gravitate toward VWAP before the close.
        """)
    
    with st.expander("3. Anchored VWAP", expanded=True):
        st.write("""
        Standard VWAP resets each day, but Anchored VWAP allows you to:
        - Restart calculations from significant events (earnings announcements, major gaps)
        - Identify longer-term support/resistance zones based on volume-weighted activity
        - Track institutional cost basis after major news events
        
        Significant price reactions often occur when current price approaches Anchored VWAP from important market events.
        """)
    
    # Implementation code
    with st.expander("Sample Implementation", expanded=True):
        st.code("""
# Calculate VWAP
def calculate_vwap(df):
    df = df.copy()
    
    # Calculate typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate VWAP components
    df['tp_volume'] = df['typical_price'] * df['volume']
    df['cum_tp_volume'] = df['tp_volume'].cumsum()
    df['cum_volume'] = df['volume'].cumsum()
    
    # Calculate VWAP
    df['vwap'] = df['cum_tp_volume'] / df['cum_volume']
    
    # Calculate standard deviation bands
    price_vwap_diff = df['typical_price'] - df['vwap']
    df['stdev'] = price_vwap_diff.rolling(window=14).std()
    df['vwap_upper_1'] = df['vwap'] + 1 * df['stdev']
    df['vwap_lower_1'] = df['vwap'] - 1 * df['stdev']
    df['vwap_upper_2'] = df['vwap'] + 2 * df['stdev']
    df['vwap_lower_2'] = df['vwap'] - 2 * df['stdev']
    
    return df['vwap']
        """, language="python")

def render_measured_moves():
    """
    Render educational content about Measured Moves using Fibonacci for day trading
    """
    st.header("Measured Moves: Fibonacci-Based Target Projection")
    
    st.write("""
    Measured Moves are precise price projections based on Fibonacci ratios that help traders identify 
    high-probability targets for trades. This technique combines pattern recognition with mathematical 
    precision to anticipate where price is likely to reverse or consolidate.
    """)
    
    # Pattern Recognition
    st.subheader("Pattern Recognition")
    st.write("""
    Measured Moves occur when:
    1. Price retraces 50-61.8% of prior swing
    2. Subsequent move reaches -23.6% Fibonacci extension
    
    The TDU Measured Move Indicator automates detection of completed patterns, showing:
    - 78% accuracy in ES futures on 15-min charts
    - 2:1 average risk-reward ratio
    """)
    
    # Visual explanation
    st.subheader("Visual Guide to Measured Moves")
    
    cols = st.columns(2)
    
    with cols[0]:
        st.markdown("##### Bullish Measured Move")
        st.write("""
        1. Identify a strong upward swing (A to B)
        2. Wait for a retracement of 50-61.8% (B to C)
        3. Project a -23.6% extension target (C to D)
        
        Entry would be at point C with stop below the most recent low.
        """)
    
    with cols[1]:
        st.markdown("##### Bearish Measured Move")
        st.write("""
        1. Identify a strong downward swing (A to B)
        2. Wait for a retracement of 50-61.8% (B to C)
        3. Project a -23.6% extension target (C to D)
        
        Entry would be at point C with stop above the most recent high.
        """)
    
    # Implementation and strategy
    st.subheader("Implementation Strategy")
    
    with st.expander("Step 1: Identify Swing Points", expanded=True):
        st.write("""
        Use swing detection algorithms or manual identification to find:
        - Major swing high/low points
        - Clear directional moves with definitive peaks and troughs
        - Swings of at least 10 ATR (Average True Range) for reliability
        """)
    
    with st.expander("Step 2: Calculate Fibonacci Levels", expanded=True):
        st.write("""
        For each identified swing:
        - Draw Fibonacci retracement from swing start to end
        - Mark the 50% and 61.8% retracement levels
        - Calculate the -23.6% extension level for profit targets
        
        These levels create a framework for potential trade opportunities.
        """)
    
    with st.expander("Step 3: Trade Management", expanded=True):
        st.write("""
        When price reaches the 50-61.8% retracement zone:
        - Enter in the direction of the original swing
        - Place stop beyond the nearest swing point
        - Target the -23.6% extension for exits
        - Consider scaling out at 1:1 risk-reward ratio and letting remainder run to full target
        """)
    
    # Implementation code
    with st.expander("Sample Implementation", expanded=True):
        st.code("""
# Detect Measured Move pattern and calculate target
def detect_measured_move(df, window=20):
    df = df.copy()
    
    # Find swing highs and lows
    df['swing_high'] = df['high'].rolling(window=window, center=True).apply(
        lambda x: np.argmax(x) == len(x)//2, raw=True
    )
    df['swing_low'] = df['low'].rolling(window=window, center=True).apply(
        lambda x: np.argmin(x) == len(x)//2, raw=True
    )
    
    # Identify potential measured moves
    results = pd.DataFrame(index=df.index)
    results['potential_target'] = np.nan
    
    for i in range(window, len(df)):
        # Find the most recent swing high and low
        recent_high_idx = df.index[df.iloc[i-window:i]['swing_high']].max() if any(df.iloc[i-window:i]['swing_high']) else None
        recent_low_idx = df.index[df.iloc[i-window:i]['swing_low']].max() if any(df.iloc[i-window:i]['swing_low']) else None
        
        if recent_high_idx and recent_low_idx:
            # Calculate potential measured move
            if recent_high_idx > recent_low_idx:  # Bullish pattern
                swing_high = df.loc[recent_high_idx, 'high']
                swing_low = df.loc[recent_low_idx, 'low']
                
                # Check for 50-61.8% retracement
                current_price = df.iloc[i]['close']
                retracement_50 = swing_high - (swing_high - swing_low) * 0.5
                retracement_618 = swing_high - (swing_high - swing_low) * 0.618
                
                if retracement_618 <= current_price <= retracement_50:
                    # Calculate target (-23.6% extension)
                    target = swing_high + (swing_high - swing_low) * 0.236
                    results.iloc[i, results.columns.get_loc('potential_target')] = target
            
            else:  # Bearish pattern
                swing_high = df.loc[recent_high_idx, 'high']
                swing_low = df.loc[recent_low_idx, 'low']
                
                # Check for 50-61.8% retracement
                current_price = df.iloc[i]['close']
                retracement_50 = swing_low + (swing_high - swing_low) * 0.5
                retracement_618 = swing_low + (swing_high - swing_low) * 0.618
                
                if retracement_50 <= current_price <= retracement_618:
                    # Calculate target (-23.6% extension)
                    target = swing_low - (swing_high - swing_low) * 0.236
                    results.iloc[i, results.columns.get_loc('potential_target')] = target
    
    return results['potential_target']
        """, language="python")

def render_strategic_combinations():
    """
    Render educational content about strategic combinations of advanced trading concepts
    """
    st.header("Strategic Combinations")
    
    st.write("""
    These powerful combinations leverage multiple advanced concepts simultaneously to create high-probability trading setups.
    By combining complementary techniques, traders can create more robust systems with higher win rates.
    """)
    
    # First strategic combination
    with st.expander("1. EMA Cloud + VWAP Convergence", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("""
            This strategy combines the dynamic support/resistance of EMA Clouds with the institutional benchmark of VWAP
            to identify high-probability entry points.
            
            **Entry Rules:**
            - Price bounces off EMA cloud while above VWAP (bullish setup)
            - Enter long on the first candle close above the cloud after the bounce
            - For bearish setups, price must bounce down from cloud while below VWAP
            
            **Exit Rules:**
            - Primary exit: VWAP crossover in the opposite direction
            - Secondary exit: EMA cloud color change (bullish to bearish or vice versa)
            - Stop loss: Below the recent swing low (for longs) or above swing high (for shorts)
            
            **Statistical Edge:**
            - 63% win rate in NASDAQ 100 stocks (2024 backtest)
            - Average reward:risk ratio of 1.8:1
            - Most effective during regular market hours (9:30AM-3:30PM ET)
            """)
        
        with col2:
            st.markdown("##### Optimal Parameters")
            st.markdown("- EMA Cloud: 8-9 EMA")
            st.markdown("- VWAP: Standard daily")
            st.markdown("- Timeframe: 5-15 min")
            st.markdown("- Markets: US Equities")
            st.markdown("- Best during: Opening hour")
    
    # Second strategic combination
    with st.expander("2. Measured Move + Volume Confirmation", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("""
            This strategy enhances the reliability of Measured Move patterns by requiring volume confirmation,
            filtering out low-probability setups.
            
            **Entry Rules:**
            - Identify completed measured move pattern (50-61.8% retracement)
            - Confirmation: Volume > 20-period average at the breakout point
            - Enter in the direction of the original trend when price breaks out from the retracement zone
            
            **Exit Rules:**
            - Primary target: -23.6% Fibonacci extension level
            - Trailing stop: ATR(14) trailing stop once price moves in your favor
            - Scale out: Consider partial profit taking at 1:1 risk-reward ratio
            
            **Risk Management:**
            - Initial stop placed beyond the retracement zone
            - Position size limited to 1-2% account risk per trade
            - Avoid trading during major news events
            """)
        
        with col2:
            st.markdown("##### Optimal Parameters")
            st.markdown("- Fib levels: 50%, 61.8%")
            st.markdown("- Volume threshold: >120% of 20-period average")
            st.markdown("- ATR period: 14")
            st.markdown("- Timeframe: 15-60 min")
            st.markdown("- Markets: Futures, Forex, Stocks")
    
    # Implementation code example
    with st.expander("Sample Implementation (EMA Cloud + VWAP)", expanded=False):
        st.code("""
# Strategic combination: EMA Cloud + VWAP
def detect_ema_vwap_setup(df, fast_ema=8, slow_ema=9):
    df = df.copy()
    
    # Calculate EMA Cloud
    df['fast_ema'] = df['close'].ewm(span=fast_ema, adjust=False).mean()
    df['slow_ema'] = df['close'].ewm(span=slow_ema, adjust=False).mean()
    df['cloud_bullish'] = df['fast_ema'] > df['slow_ema']
    
    # Calculate VWAP
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['typical_price'] * df['volume']
    df['cum_tp_volume'] = df['tp_volume'].cumsum()
    df['cum_volume'] = df['volume'].cumsum()
    df['vwap'] = df['cum_tp_volume'] / df['cum_volume']
    
    # Detect setups
    bullish_setups = pd.Series(False, index=df.index)
    bearish_setups = pd.Series(False, index=df.index)
    
    for i in range(3, len(df)):
        # Bullish setup: price touches cloud from below while above VWAP
        cloud_bottom = min(df['fast_ema'].iloc[i-1], df['slow_ema'].iloc[i-1])
        if (df['low'].iloc[i-1] <= cloud_bottom and 
            df['close'].iloc[i-1] > cloud_bottom and
            df['close'].iloc[i-1] > df['vwap'].iloc[i-1] and
            df['cloud_bullish'].iloc[i-1]):
            bullish_setups.iloc[i] = True
            
        # Bearish setup: price touches cloud from above while below VWAP
        cloud_top = max(df['fast_ema'].iloc[i-1], df['slow_ema'].iloc[i-1])
        if (df['high'].iloc[i-1] >= cloud_top and 
            df['close'].iloc[i-1] < cloud_top and
            df['close'].iloc[i-1] < df['vwap'].iloc[i-1] and
            not df['cloud_bullish'].iloc[i-1]):
            bearish_setups.iloc[i] = True
    
    return bullish_setups, bearish_setups
        """, language="python")
    
    with st.expander("Sample Implementation (Measured Move + Volume)", expanded=False):
        st.code("""
# Strategic combination: Measured Move + Volume Confirmation
def detect_measured_move_volume(df, window=20, volume_threshold=1.2):
    df = df.copy()
    
    # Find swing highs and lows
    df['swing_high'] = df['high'].rolling(window=window, center=True).apply(
        lambda x: np.argmax(x) == len(x)//2, raw=True
    )
    df['swing_low'] = df['low'].rolling(window=window, center=True).apply(
        lambda x: np.argmin(x) == len(x)//2, raw=True
    )
    
    # Calculate volume average
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Calculate ATR for trailing stop
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Identify measured moves with volume confirmation
    results = pd.DataFrame(index=df.index)
    results['bullish_setup'] = False
    results['bearish_setup'] = False
    results['target'] = np.nan
    results['stop_loss'] = np.nan
    
    for i in range(window, len(df)):
        # Skip if volume is not above threshold
        if df['volume_ratio'].iloc[i] <= volume_threshold:
            continue
            
        # Find recent swing points
        recent_high_idx = df.index[df.iloc[i-window:i]['swing_high']].max() if any(df.iloc[i-window:i]['swing_high']) else None
        recent_low_idx = df.index[df.iloc[i-window:i]['swing_low']].max() if any(df.iloc[i-window:i]['swing_low']) else None
        
        if recent_high_idx and recent_low_idx:
            # Process potential setups
            if recent_high_idx > recent_low_idx:  # Bullish pattern
                swing_high = df.loc[recent_high_idx, 'high']
                swing_low = df.loc[recent_low_idx, 'low']
                
                # Check for 50-61.8% retracement with breakout
                retracement_50 = swing_high - (swing_high - swing_low) * 0.5
                retracement_618 = swing_high - (swing_high - swing_low) * 0.618
                
                if (retracement_618 <= df['low'].iloc[i-1] <= retracement_50 and
                    df['close'].iloc[i] > df['high'].iloc[i-1]):
                    # Calculate target and stop
                    target = swing_high + (swing_high - swing_low) * 0.236
                    stop_loss = min(df['low'].iloc[i-3:i]) - df['atr'].iloc[i]
                    
                    # Mark as bullish setup
                    results.loc[df.index[i], 'bullish_setup'] = True
                    results.loc[df.index[i], 'target'] = target
                    results.loc[df.index[i], 'stop_loss'] = stop_loss
            
            else:  # Bearish pattern
                swing_high = df.loc[recent_high_idx, 'high']
                swing_low = df.loc[recent_low_idx, 'low']
                
                # Check for 50-61.8% retracement with breakdown
                retracement_50 = swing_low + (swing_high - swing_low) * 0.5
                retracement_618 = swing_low + (swing_high - swing_low) * 0.618
                
                if (retracement_50 <= df['high'].iloc[i-1] <= retracement_618 and
                    df['close'].iloc[i] < df['low'].iloc[i-1]):
                    # Calculate target and stop
                    target = swing_low - (swing_high - swing_low) * 0.236
                    stop_loss = max(df['high'].iloc[i-3:i]) + df['atr'].iloc[i]
                    
                    # Mark as bearish setup
                    results.loc[df.index[i], 'bearish_setup'] = True
                    results.loc[df.index[i], 'target'] = target
                    results.loc[df.index[i], 'stop_loss'] = stop_loss
    
    return results
        """, language="python")
    
    # Success factors
    st.subheader("Keys to Successful Implementation")
    
    st.write("""
    When combining multiple advanced techniques, keep these principles in mind:
    
    1. **Simplicity over complexity**: Aim for clear rules that combine just 2-3 concepts effectively
    2. **Alignment across timeframes**: Confirm setups on multiple timeframes when possible
    3. **Volume validation**: Almost all high-probability setups show abnormal volume characteristics
    4. **Contextual awareness**: Market structure and session timing significantly impact results
    5. **Rigorous testing**: Backtest combinations thoroughly before trading real capital
    
    The most successful traders typically master a small number of these combinations rather than
    employing many different strategies inconsistently.
    """)

def render_advanced_concepts():
    """
    Master function to render all advanced trading concepts
    """
    st.title("Advanced Trading Concepts")
    
    # Create tabs for each concept
    tabs = st.tabs(["EMA Clouds", "VWAP", "Measured Moves", "Strategic Combinations"])
    
    with tabs[0]:
        render_ema_clouds()
        
    with tabs[1]:
        render_vwap()
        
    with tabs[2]:
        render_measured_moves()
        
    with tabs[3]:
        render_strategic_combinations() 