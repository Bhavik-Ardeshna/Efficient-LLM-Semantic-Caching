#!/bin/bash

# Load Testing Quick Start Script for Semantic Cache Service
# This script helps you quickly run load tests and check system resilience

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HOST=${HOST:-"http://localhost:3000"}
TEST_TYPE=${1:-"all"}
DURATION=${2:-"default"}

echo -e "${BLUE}🚀 Semantic Cache Service - Load Testing Suite${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Function to check if service is running
check_service() {
    echo -e "${YELLOW}🔍 Checking service health...${NC}"
    
    if curl -s -f "$HOST/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is running at $HOST${NC}"
        return 0
    else
        echo -e "${RED}❌ Service is not reachable at $HOST${NC}"
        echo -e "${YELLOW}💡 Please ensure the service is running:${NC}"
        echo "   docker-compose up -d"
        echo "   # or"
        echo "   python run.py"
        return 1
    fi
}

# Function to check dependencies
check_dependencies() {
    echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is not installed${NC}"
        return 1
    fi
    
    # Check if required packages are installed
    if ! python3 -c "import locust" &> /dev/null; then
        echo -e "${YELLOW}⚠️  Locust not found. Installing dependencies...${NC}"
        pip install -r requirements.txt
    fi
    
    echo -e "${GREEN}✅ Dependencies are ready${NC}"
}

# Function to display help
show_help() {
    echo "Usage: $0 [TEST_TYPE] [DURATION]"
    echo ""
    echo "TEST_TYPE options:"
    echo "  all        - Run all test scenarios (default)"
    echo "  baseline   - Normal load test (10 users, 5 min)"
    echo "  high_load  - High load test (50 users, 5 min)"
    echo "  stress     - Stress test (100 users, 3 min)"
    echo "  help       - Show this help message"
    echo ""
    echo "DURATION: Test duration in seconds (optional, overrides defaults)"
    echo ""
    echo "Examples:"
    echo "  $0 baseline 300    # Run baseline test for 5 minutes"
    echo "  $0 stress          # Run stress test with default duration"
    echo "  $0 all             # Run all test scenarios"
    echo ""
    echo "Environment variables:"
    echo "  HOST=$HOST"
}

# Function to test system health endpoints
test_system_health() {
    echo -e "${BLUE}🔍 Testing System Health Endpoints${NC}"
    echo -e "${BLUE}==================================${NC}"
    
    # Test individual health endpoints
    endpoints=(
        "/health"
        "/api/cache/health"
    )
    
    for endpoint in "${endpoints[@]}"; do
        echo -n "  Testing $endpoint... "
        if curl -s -f "$HOST$endpoint" > /dev/null; then
            echo -e "${GREEN}✅${NC}"
        else
            echo -e "${RED}❌${NC}"
        fi
    done
    
    echo ""
    echo -e "${YELLOW}🎯 Load manager runs automatically...${NC}"
    
    echo ""
    echo -e "${GREEN}✅ System health check completed${NC}"
}

# Function to run performance summary
show_performance_summary() {
    echo ""
    echo -e "${BLUE}📊 Current System Performance${NC}"
    echo -e "${BLUE}=============================${NC}"
    
    echo -e "${YELLOW}Load Manager: Running automatically${NC}"
    
    echo ""
    echo -e "${YELLOW}Cache Statistics:${NC}"
    curl -s "$HOST/api/cache/stats" | python3 -m json.tool
}

# Main execution
main() {
    case $TEST_TYPE in
        "help"|"-h"|"--help")
            show_help
            exit 0
            ;;
        "health")
            check_service || exit 1
            test_system_health
            show_performance_summary
            exit 0
            ;;
    esac
    
    # Check prerequisites
    check_dependencies || exit 1
    check_service || exit 1
    
    echo ""
    echo -e "${YELLOW}🏃 Starting load tests...${NC}"
    echo "  Test type: $TEST_TYPE"
    echo "  Target: $HOST"
    if [ "$DURATION" != "default" ]; then
        echo "  Duration: $DURATION seconds"
    fi
    echo ""
    
    # Create results directory
    mkdir -p load_testing/results
    
    # Run the load tests
    if [ "$DURATION" = "default" ]; then
        python3 load_testing/run_load_tests.py --host "$HOST" --test "$TEST_TYPE"
    else
        python3 load_testing/run_load_tests.py --host "$HOST" --test "$TEST_TYPE" --duration "$DURATION"
    fi
    
    # Show final performance summary
    show_performance_summary
    
    echo ""
    echo -e "${GREEN}🎉 Load testing completed!${NC}"
    echo -e "${BLUE}📁 Results saved to: load_testing/results/${NC}"
    echo ""
    echo -e "${YELLOW}📊 To view detailed results:${NC}"
    echo "  1. Open the HTML reports in load_testing/results/"
    echo "  2. Check CSV files for raw data"
    echo "  3. Review the test summary JSON files"
    echo ""
    echo -e "${YELLOW}🔍 Monitor ongoing performance:${NC}"
    echo "  curl $HOST/api/cache/stats"
    echo "  curl $HOST/api/cache/health"
}

# Run main function
main "$@" 