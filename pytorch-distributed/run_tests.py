#!/usr/bin/env python
"""
Test runner script for the pytorch-distributed project.
This script discovers and runs all tests in the tests directory.
"""

import unittest
import sys
import os

if __name__ == '__main__':
    # Add the current directory to the path so we can import the gentrl module
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Run the tests
    result = unittest.TextTestRunner().run(test_suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful())