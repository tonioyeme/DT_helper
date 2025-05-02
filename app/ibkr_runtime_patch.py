"""
IBKR Runtime Parameter Fix

This patch automatically fixes incompatible parameter combinations in IBKR API calls.
Import this at the beginning of your main application to prevent errors.

The issue: According to IBKR API documentation, you cannot use snapshot=True with 
generic tick types in the same reqMktData call.
"""

from ibapi.client import EClient
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ibkr_runtime_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ibkr_runtime_fix")

# Store the original method
original_req_mkt_data = EClient.reqMktData

# Define a wrapper function to detect and fix invalid combinations
def patched_req_mkt_data(self, reqId, contract, genericTickList="", snapshot=False, 
                       regulatorySnapshot=False, mktDataOptions=None):
    """
    Patched version of reqMktData that prevents incompatible parameter combinations
    
    This patch automatically detects and fixes the following issues:
    - Using snapshot=True with non-empty genericTickList (incompatible according to IBKR API)
    
    In case of incompatible parameters, this patch defaults to:
    - Keep the generic tick list (setting snapshot=False)
    """
    # Check for incompatible parameter combinations
    if genericTickList and snapshot:
        import traceback
        
        # Create a stack trace to identify the source of the call
        stack = traceback.extract_stack()
        calling_file = stack[-2].filename
        calling_line = stack[-2].lineno
        
        # Log the issue
        logger.warning(f"Incompatible parameter combination detected in {calling_file}:{calling_line}")
        logger.warning(f"  Contract: {contract.symbol}")
        logger.warning(f"  Generic ticks: '{genericTickList}'")
        logger.warning(f"  Snapshot: {snapshot}")
        logger.warning("  These parameters are incompatible according to IBKR API docs.")
        logger.warning("  Automatically fixing by setting snapshot=False")
        
        # Fix the incompatible combination by prioritizing generic ticks
        snapshot = False
    
    # Call the original method with fixed parameters
    return original_req_mkt_data(self, reqId, contract, genericTickList, snapshot, 
                               regulatorySnapshot, mktDataOptions)

# Apply the patch
logger.info("Applying IBKR parameter fix patch")
EClient.reqMktData = patched_req_mkt_data
logger.info("IBKR parameter fix patch applied successfully")

print("IBKR Parameter Fix: Patched reqMktData to prevent incompatible parameter combinations") 