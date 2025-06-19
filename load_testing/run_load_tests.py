#!/usr/bin/env python3
"""
Load testing runner script for Semantic Cache Service
"""
import subprocess
import time
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List
import requests


class LoadTestRunner:
    """Runner for different load test scenarios"""
    
    def __init__(self, host: str = "http://localhost:3000"):
        self.host = host
        self.results_dir = "load_testing/results"
        self.ensure_results_dir()
    
    def ensure_results_dir(self):
        """Ensure results directory exists"""
        os.makedirs(self.results_dir, exist_ok=True)
    
    def check_service_health(self) -> bool:
        """Check if the service is running and healthy"""
        try:
            response = requests.get(f"{self.host}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Service health check failed: {e}")
            return False
    
    def run_baseline_test(self, duration: int = 300) -> Dict[str, Any]:
        """
        Run baseline load test with normal user behavior
        
        Args:
            duration: Test duration in seconds (default 5 minutes)
        """
        print(f"Running baseline load test for {duration} seconds...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.results_dir}/baseline_test_{timestamp}.html"
        csv_file = f"{self.results_dir}/baseline_test_{timestamp}.csv"
        
        cmd = [
            "locust",
            "-f", "load_testing/baseline_test.py",
            "--host", self.host,
            "--users", "10",
            "--spawn-rate", "2",
            "--run-time", f"{duration}s",
            "--html", results_file,
            "--csv", csv_file.replace('.csv', ''),
            "--headless"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "test_type": "baseline",
            "duration": duration,
            "users": 10,
            "spawn_rate": 2,
            "results_file": results_file,
            "csv_file": csv_file,
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr
        }
    
    def run_high_load_test(self, duration: int = 300) -> Dict[str, Any]:
        """
        Run high load test with aggressive user behavior
        
        Args:
            duration: Test duration in seconds (default 5 minutes)
        """
        print(f"Running high load test for {duration} seconds...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.results_dir}/high_load_test_{timestamp}.html"
        csv_file = f"{self.results_dir}/high_load_test_{timestamp}.csv"
        
        cmd = [
            "locust",
            "-f", "load_testing/high_load_test.py",
            "--host", self.host,
            "--users", "50",
            "--spawn-rate", "5",
            "--run-time", f"{duration}s",
            "--html", results_file,
            "--csv", csv_file.replace('.csv', ''),
            "--headless"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "test_type": "high_load",
            "duration": duration,
            "users": 50,
            "spawn_rate": 5,
            "results_file": results_file,
            "csv_file": csv_file,
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr
        }
    
    def run_stress_test(self, duration: int = 180) -> Dict[str, Any]:
        """
        Run stress test to find breaking point
        
        Args:
            duration: Test duration in seconds (default 3 minutes)
        """
        print(f"Running stress test for {duration} seconds...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.results_dir}/stress_test_{timestamp}.html"
        csv_file = f"{self.results_dir}/stress_test_{timestamp}.csv"
        
        cmd = [
            "locust",
            "-f", "load_testing/stress_test.py",
            "--host", self.host,
            "--users", "100",
            "--spawn-rate", "10",
            "--run-time", f"{duration}s",
            "--html", results_file,
            "--csv", csv_file.replace('.csv', ''),
            "--headless"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "test_type": "stress",
            "duration": duration,
            "users": 100,
            "spawn_rate": 10,
            "results_file": results_file,
            "csv_file": csv_file,
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr
        }
    

    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all load test scenarios"""
        print("Starting comprehensive load testing suite...")
        
        if not self.check_service_health():
            print("ERROR: Service is not healthy. Please start the service before running tests.")
            return []
        
        results = []
        
        # Run different test scenarios (simplified - no circuit breaker testing)
        test_scenarios = [
            ("baseline", self.run_baseline_test, 300),
            ("high_load", self.run_high_load_test, 300),
            ("stress", self.run_stress_test, 180),
        ]
        
        for name, test_func, duration in test_scenarios:
            print(f"\n{'='*50}")
            print(f"Running {name} test...")
            print(f"{'='*50}")
            
            result = test_func(duration)
            results.append(result)
            
            # Print summary
            if result['success']:
                print(f"✅ {name} test completed successfully")
                print(f"   Results: {result['results_file']}")
            else:
                print(f"❌ {name} test failed")
                print(f"   Error: {result['error']}")
            
            # Wait between tests to let system recover
            if name != test_scenarios[-1][0]:  # Don't wait after last test
                print("Waiting 30 seconds for system recovery...")
                time.sleep(30)
        
        # Save overall summary
        self.save_test_summary(results)
        
        return results
    
    def save_test_summary(self, results: List[Dict[str, Any]]):
        """Save overall test summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"{self.results_dir}/test_summary_{timestamp}.json"
        
        summary = {
            "timestamp": timestamp,
            "host": self.host,
            "total_tests": len(results),
            "successful_tests": len([r for r in results if r['success']]),
            "failed_tests": len([r for r in results if not r['success']]),
            "tests": results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*50}")
        print("LOAD TESTING SUMMARY")
        print(f"{'='*50}")
        print(f"Total tests run: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Summary saved to: {summary_file}")
        
        # Print individual test results
        for result in results:
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            print(f"{result['test_type']:<12} | {status} | {result['users']} users | {result['duration']}s")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Load testing runner for Semantic Cache Service")
    parser.add_argument("--host", default="http://localhost:3000", help="Target host URL")
    parser.add_argument("--test", choices=["all", "baseline", "high_load", "stress"],
                        default="all", help="Test type to run")
    parser.add_argument("--duration", type=int, help="Test duration in seconds (overrides defaults)")
    
    args = parser.parse_args()
    
    runner = LoadTestRunner(host=args.host)
    
    if args.test == "all":
        runner.run_all_tests()
    elif args.test == "baseline":
        duration = args.duration or 300
        result = runner.run_baseline_test(duration)
        print(f"Baseline test {'completed' if result['success'] else 'failed'}")
    elif args.test == "high_load":
        duration = args.duration or 300
        result = runner.run_high_load_test(duration)
        print(f"High load test {'completed' if result['success'] else 'failed'}")
    elif args.test == "stress":
        duration = args.duration or 180
        result = runner.run_stress_test(duration)
        print(f"Stress test {'completed' if result['success'] else 'failed'}")



if __name__ == "__main__":
    main() 