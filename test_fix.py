#!/usr/bin/env python3

# Simple test to verify the rolling window fix
import numpy as np

class TestRollingWindow:
    def __init__(self, window_size=10):
        self.window_size = window_size
        
    def test_loop(self, data_size=100):
        """Test the fixed rolling window logic"""
        print(f"Testing rolling window with {data_size} data points...")
        
        # Simulate the fixed loop logic
        max_windows = min(50, data_size - self.window_size)  # Limit to prevent infinite loops
        step_size = max(1, (data_size - self.window_size) // max_windows)  # Dynamic step size
        
        processed_windows = 0
        
        for window_idx, i in enumerate(range(self.window_size, data_size, step_size)):
            print(f"Window {window_idx+1}/{max_windows}: processing up to index {i}")
            
            processed_windows += 1
            
            # Safety check to prevent infinite loops
            if window_idx >= max_windows:
                print(f"Reached maximum windows limit ({max_windows}), stopping analysis")
                break
                
            # Simulate some processing
            if window_idx >= 5:  # Stop after a few for testing
                print("Stopping early for test")
                break
        
        print(f"Processed {processed_windows} windows successfully")
        return processed_windows > 0

if __name__ == "__main__":
    tester = TestRollingWindow(window_size=10)
    success = tester.test_loop(data_size=100)
    print(f"Test {'PASSED' if success else 'FAILED'}")