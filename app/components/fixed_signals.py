import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict
import plotly.express as px
import traceback
import matplotlib.pyplot as plt

from app.signals import generate_signals, SignalStrength
from app.signals.generator import create_default_signal_generator, generate_signals_advanced, analyze_exit_strategy
from app.signals.timeframes import TimeFrame, TimeFramePriority
from app.indicators import calculate_opening_range

def is_market_hours(timestamp):
    """
    Check if the given timestamp is during market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
    
    Args:
        timestamp: Datetime object or index
        
    Returns:
        bool: True if timestamp is during market hours, False otherwise
    """
    # If timestamp has no tzinfo, assume it's UTC and convert
    if hasattr(timestamp, 'tzinfo'):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=pytz.UTC)
        # Convert to Eastern Time
        eastern = pytz.timezone('US/Eastern')
        timestamp = timestamp.astimezone(eastern)
    else:
        # If not a datetime, just return True (can't determine)
        return True
    
    # Check if it's a weekday (0 = Monday, 4 = Friday)
    if timestamp.weekday() > 4:  # Saturday or Sunday
        return False
        
    # Check if within 9:30 AM - 4:00 PM ET
    market_open = time(9, 30, 0)
    market_close = time(16, 0, 0)
    
    return market_open <= timestamp.time() <= market_close 