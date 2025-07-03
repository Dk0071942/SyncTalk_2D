#!/usr/bin/env python3
"""Test runner for SyncTalk 2D."""

import sys
import unittest
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Run SyncTalk 2D tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--pattern', '-p', type=str, default='test*.py',
                       help='Test file pattern (default: test*.py)')
    
    args = parser.parse_args()
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Determine which tests to run
    if args.unit:
        test_dirs = ['tests/unit']
    elif args.integration:
        test_dirs = ['tests/integration']
    else:
        test_dirs = ['tests/unit', 'tests/integration']
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_dir in test_dirs:
        test_path = project_root / test_dir
        if test_path.exists():
            discovered = loader.discover(
                start_dir=str(test_path),
                pattern=args.pattern,
                top_level_dir=str(project_root)
            )
            suite.addTests(discovered)
    
    # Run tests
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())