"""
Master test runner for all unit tests.

Runs all test suites and provides comprehensive reporting.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import time
import unittest
from io import StringIO

# Import all test modules
import test_train_utils
import test_train_advanced
import test_fft_layers_comprehensive
import test_data_pipeline_comprehensive


class TestResults:
    """Container for test results with detailed reporting."""

    def __init__(self, name):
        self.name = name
        self.result = None
        self.duration = 0
        self.stdout = ""
        self.stderr = ""


def run_test_suite(test_module, module_name):
    """Run a test suite and capture results."""
    print(f"\n{'='*70}")
    print(f"Running: {module_name}")
    print(f"{'='*70}")

    # Capture output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        start_time = time.time()
        result = test_module.run_tests()
        duration = time.time() - start_time

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    # Create results object
    test_results = TestResults(module_name)
    test_results.result = result
    test_results.duration = duration
    test_results.stdout = stdout_capture.getvalue()
    test_results.stderr = stderr_capture.getvalue()

    # Print results immediately
    print(test_results.stdout)
    if test_results.stderr:
        print("STDERR:")
        print(test_results.stderr)

    print(f"\nDuration: {duration:.2f}s")

    return test_results


def print_summary(all_results):
    """Print comprehensive summary of all tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*70)

    total_tests = 0
    total_successes = 0
    total_failures = 0
    total_errors = 0
    total_duration = 0

    print(f"\n{'Suite':<35} {'Tests':<8} {'Pass':<8} {'Fail':<8} {'Error':<8} {'Time':<8}")
    print("-" * 70)

    for test_result in all_results:
        result = test_result.result
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        successes = tests_run - failures - errors
        duration = test_result.duration

        total_tests += tests_run
        total_successes += successes
        total_failures += failures
        total_errors += errors
        total_duration += duration

        status = "✓" if result.wasSuccessful() else "✗"

        print(f"{test_result.name:<35} {tests_run:<8} {successes:<8} "
              f"{failures:<8} {errors:<8} {duration:<7.2f}s {status}")

    print("-" * 70)
    print(f"{'TOTAL':<35} {total_tests:<8} {total_successes:<8} "
          f"{total_failures:<8} {total_errors:<8} {total_duration:<7.2f}s")

    print("\n" + "="*70)
    print("DETAILED STATISTICS")
    print("="*70)
    print(f"Total Test Suites:     {len(all_results)}")
    print(f"Total Tests Run:       {total_tests}")
    print(f"Successful Tests:      {total_successes} ({total_successes/total_tests*100:.1f}%)")
    print(f"Failed Tests:          {total_failures}")
    print(f"Error Tests:           {total_errors}")
    print(f"Total Duration:        {total_duration:.2f}s")
    print(f"Average per Suite:     {total_duration/len(all_results):.2f}s")

    # Print failures and errors if any
    if total_failures > 0 or total_errors > 0:
        print("\n" + "="*70)
        print("FAILURES AND ERRORS")
        print("="*70)

        for test_result in all_results:
            result = test_result.result

            if result.failures:
                print(f"\n{test_result.name} - FAILURES:")
                for test, traceback in result.failures:
                    print(f"  {test}: {traceback}")

            if result.errors:
                print(f"\n{test_result.name} - ERRORS:")
                for test, traceback in result.errors:
                    print(f"  {test}: {traceback}")

    # Overall result
    print("\n" + "="*70)
    all_passed = all(r.result.wasSuccessful() for r in all_results)

    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
    else:
        print("❌ SOME TESTS FAILED")

    print("="*70 + "\n")

    return all_passed


def main():
    """Run all test suites."""
    print("="*70)
    print("COMPREHENSIVE UNIT TEST SUITE")
    print("Integration Branch: claude/integration-all-features-OkWhC")
    print("="*70)
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test_suites = [
        (test_train_utils, "train_utils.py"),
        (test_train_advanced, "train_advanced.py"),
        (test_fft_layers_comprehensive, "fft_layers.py"),
        (test_data_pipeline_comprehensive, "data_pipeline.py"),
    ]

    all_results = []

    for test_module, module_name in test_suites:
        try:
            result = run_test_suite(test_module, module_name)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR running {module_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print comprehensive summary
    all_passed = print_summary(all_results)

    print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
