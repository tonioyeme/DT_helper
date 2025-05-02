"""
Utility functions to ensure proper asyncio functionality with IB-insync and Streamlit.
"""

import asyncio
import functools
import threading
import logging
from typing import Callable, Any
import queue
import time

# Configure logging
logger = logging.getLogger(__name__)

def ensure_event_loop():
    """Ensure there is an event loop available in the current thread"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            logger.warning("Event loop was closed. Creating a new one.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        logger.info("No event loop in current thread. Creating a new one.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def with_event_loop(func):
    """
    Decorator to ensure a function runs with an event loop
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        loop = ensure_event_loop()
        if asyncio.iscoroutinefunction(func):
            return loop.run_until_complete(func(*args, **kwargs))
        else:
            return func(*args, **kwargs)
    return wrapper

# Task queue for Streamlit-safe background operations
class StreamlitBackgroundTaskManager:
    """
    Manages background tasks in a way that's compatible with Streamlit.
    
    Instead of directly accessing session_state from background threads,
    this class uses a queue system to safely transfer data between threads.
    """
    
    def __init__(self):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the background task manager thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.thread.start()
        logger.info("Background task manager started")
    
    def stop(self):
        """Stop the background task manager thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        logger.info("Background task manager stopped")
    
    def _process_tasks(self):
        """Process tasks from the queue"""
        while self.running:
            try:
                # Get task with timeout to allow checking if still running
                try:
                    task_id, task_func, args, kwargs = self.task_queue.get(timeout=0.5)
                    
                    # Process the task
                    try:
                        result = task_func(*args, **kwargs)
                        self.result_queue.put((task_id, result, None))  # Success
                    except Exception as e:
                        logger.error(f"Error in background task: {str(e)}")
                        self.result_queue.put((task_id, None, e))  # Error
                        
                    # Mark task as done
                    self.task_queue.task_done()
                    
                except queue.Empty:
                    # No tasks available, continue checking
                    continue
                    
            except Exception as e:
                logger.error(f"Error in task processor: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def schedule_task(self, task_func, *args, **kwargs):
        """
        Schedule a task to run in the background
        
        Args:
            task_func: The function to run
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            task_id: An identifier for the scheduled task
        """
        task_id = id(task_func) + len(args) + len(kwargs)  # Simple unique ID
        self.task_queue.put((task_id, task_func, args, kwargs))
        return task_id
    
    def get_results(self):
        """
        Get all available results from background tasks
        
        Returns:
            list: List of (task_id, result, error) tuples
        """
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
            self.result_queue.task_done()
        return results

# Create a global instance of the background task manager
task_manager = StreamlitBackgroundTaskManager()

def run_in_background(func):
    """
    Decorator to run a function in the background without Streamlit context
    
    Use this for functions that don't need to interact with Streamlit's session state
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Start the task manager if not already running
        if not task_manager.running:
            task_manager.start()
            
        # Schedule the task
        return task_manager.schedule_task(func, *args, **kwargs)
    
    return wrapper

def get_background_results():
    """
    Get results from background tasks
    
    Call this from the main Streamlit thread to process results
    """
    if task_manager.running:
        return task_manager.get_results()
    return []

# Initialize the task manager automatically
task_manager.start()

# Add a better background loop function
class BackgroundLoopManager:
    """
    Manages a background loop that runs a function at a regular interval.
    Designed to be safe with Streamlit by avoiding direct session_state access.
    """
    
    def __init__(self, interval=1.0):
        """
        Initialize the background loop manager
        
        Args:
            interval (float): Interval in seconds between function calls
        """
        self.interval = interval
        self.functions = {}  # {name: (func, args, kwargs)}
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
    
    def start(self):
        """Start the background loop thread"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Background loop manager started")
    
    def stop(self):
        """Stop the background loop thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        logger.info("Background loop manager stopped")
    
    def add_function(self, name, func, *args, **kwargs):
        """
        Add a function to be called periodically
        
        Args:
            name (str): Name of the function (for identification)
            func (callable): Function to call
            *args, **kwargs: Arguments to pass to the function
        """
        with self.lock:
            self.functions[name] = (func, args, kwargs)
        logger.info(f"Added function {name} to background loop")
    
    def remove_function(self, name):
        """
        Remove a function from the background loop
        
        Args:
            name (str): Name of the function to remove
        """
        with self.lock:
            if name in self.functions:
                del self.functions[name]
                logger.info(f"Removed function {name} from background loop")
    
    def _run_loop(self):
        """Run the loop that calls functions periodically"""
        while self.running:
            try:
                # Copy the functions dict to avoid modification during iteration
                with self.lock:
                    functions_copy = dict(self.functions)
                
                # Call each registered function
                for name, (func, args, kwargs) in functions_copy.items():
                    try:
                        func(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in background function {name}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # Sleep for the interval
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in background loop: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # Sleep briefly before continuing

# Create a global instance
background_loop = BackgroundLoopManager()

# Initialize the background loop
background_loop.start()

def run_periodically(name, interval=1.0):
    """
    Decorator to run a function periodically in the background
    
    Args:
        name (str): Unique name for this periodic function
        interval (float): How often to run the function in seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set the interval for the background loop
            background_loop.interval = min(background_loop.interval, interval)
            
            # Register the function
            background_loop.add_function(name, func, *args, **kwargs)
            
            # Return the original function for direct calling
            return func(*args, **kwargs)
        return wrapper
    return decorator 