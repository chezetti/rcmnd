"""
Script to run all tests for the NFT recommendation system.
"""
import os
import sys
import unittest
import argparse
from pathlib import Path

# Add parent directory to path to import App modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_test_files():
    """Get all test files in the tests directory."""
    test_dir = Path(__file__).parent
    test_files = [f for f in test_dir.glob("test_*.py") if f.name != "test_data_generator.py"]
    return test_files


def run_tests(test_modules=None, verbosity=1):
    """
    Run the tests for the specified modules.
    
    Args:
        test_modules: List of test module names to run, or None for all tests
        verbosity: Verbosity level (1-3)
    """
    # If no modules specified, run all tests
    if not test_modules:
        test_files = get_test_files()
        test_modules = [f"tests.{f.stem}" for f in test_files]
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test modules to the suite
    for module_name in test_modules:
        try:
            # Import the module
            module = __import__(module_name, fromlist=["*"])
            # Add all tests from the module
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            print(f"Error importing {module_name}: {e}")
            continue
    
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def generate_test_data(count=10, output_dir="test_data"):
    """
    Generate test data for the tests.
    
    Args:
        count: Number of NFTs to generate
        output_dir: Directory to save the test data
    """
    from tests.test_data_generator import generate_test_dataset
    
    print(f"Generating {count} test NFTs in directory: {output_dir}")
    generate_test_dataset(count, output_dir)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the NFT recommendation system")
    parser.add_argument(
        "--modules", "-m", nargs="+", help="Test modules to run (e.g. tests.test_encoders)"
    )
    parser.add_argument(
        "--verbosity", "-v", type=int, choices=[1, 2, 3], default=2,
        help="Verbosity level (1-3, default: 2)"
    )
    parser.add_argument(
        "--generate-data", "-g", action="store_true",
        help="Generate test data before running tests"
    )
    parser.add_argument(
        "--count", "-c", type=int, default=10,
        help="Number of test NFTs to generate (default: 10)"
    )
    parser.add_argument(
        "--output-dir", "-o", default="test_data",
        help="Directory to save test data (default: test_data)"
    )
    
    args = parser.parse_args()
    
    # Generate test data if requested
    if args.generate_data:
        generate_test_data(args.count, args.output_dir)
    
    # Run tests
    result = run_tests(args.modules, args.verbosity)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful()) 