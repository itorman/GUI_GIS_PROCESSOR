"""
Test suite for the GIS Document Processing Application.
Provides basic test infrastructure for core modules.
"""

import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_test_suite():
    """Create test suite with all test cases"""
    test_suite = unittest.TestSuite()
    
    # Import and add test modules if available
    try:
        from test_services import TestServices
        test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestServices))
    except ImportError:
        print("Service tests not available")
    
    try:
        from test_config_manager import TestConfigManager
        test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConfigManager))
    except ImportError:
        print("Configuration manager tests not available")
    
    try:
        from test_error_handler import TestErrorHandler
        test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestErrorHandler))
    except ImportError:
        print("Error handler tests not available")
    
    return test_suite

def run_tests():
    """Run all tests and return results"""
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = create_test_suite()
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)