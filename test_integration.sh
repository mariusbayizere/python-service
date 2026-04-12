#!/bin/bash
# test_integration.sh - Quick test script for DriveGuardAI integration
# Save as: /home/marius/python-service/test_integration.sh
# Run with: bash test_integration.sh

echo "========================================"
echo "DriveGuardAI Integration Test Script"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Java Backend
echo -e "${YELLOW}[1/5] Testing Java Backend...${NC}"
if curl -s http://localhost:8080/api/v1/drivers > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Java backend is running on port 8080${NC}"
else
    echo -e "${RED}✗ Java backend is NOT running!${NC}"
    echo "Start it with: cd ~/DriveGuardAI- && mvn spring-boot:run"
    exit 1
fi
echo ""

# Test 2: Python Service
echo -e "${YELLOW}[2/5] Testing Python AI Service...${NC}"
if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Python service is running on port 5000${NC}"
else
    echo -e "${RED}✗ Python service is NOT running!${NC}"
    echo "Start it with: cd /home/marius/python-service && python api_server.py"
    exit 1
fi
echo ""

# Test 3: Connection between services
echo -e "${YELLOW}[3/5] Testing Python → Java connection...${NC}"
CONNECTION_TEST=$(curl -s http://localhost:5000/api/test-java-connection)
if echo "$CONNECTION_TEST" | grep -q "\"connected\": true"; then
    echo -e "${GREEN}✓ Python can connect to Java backend${NC}"
else
    echo -e "${RED}✗ Python CANNOT connect to Java!${NC}"
    echo "Response: $CONNECTION_TEST"
    exit 1
fi
echo ""

# Test 4: Database data
echo -e "${YELLOW}[4/5] Checking database has required data...${NC}"
DRIVERS=$(curl -s http://localhost:8080/api/v1/drivers)
if echo "$DRIVERS" | grep -q "licenseNumber"; then
    echo -e "${GREEN}✓ Drivers exist in database${NC}"
else
    echo -e "${RED}✗ No drivers found! Add test data to database${NC}"
fi
echo ""

# Test 5: Start monitoring (OPTIONAL - comment out if you don't want auto-start)
echo -e "${YELLOW}[5/5] Testing monitoring start...${NC}"
read -p "Do you want to start a test monitoring session? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting monitoring with driver_id=1, vehicle_id=1, trip_id=1..."
    MONITOR_RESULT=$(curl -s -X POST http://localhost:5000/api/monitoring/start \
      -H "Content-Type: application/json" \
      -d '{"driver_id": 1, "vehicle_id": 1, "trip_id": 1}')
    
    if echo "$MONITOR_RESULT" | grep -q "Monitoring started successfully"; then
        echo -e "${GREEN}✓ Monitoring started! Camera should open now.${NC}"
        echo "To stop monitoring: curl -X POST http://localhost:5000/api/monitoring/stop"
    else
        echo -e "${RED}✗ Failed to start monitoring${NC}"
        echo "Response: $MONITOR_RESULT"
    fi
else
    echo "Skipped monitoring test"
fi
echo ""

echo "========================================"
echo -e "${GREEN}Integration Test Complete!${NC}"
echo "========================================"
echo ""
echo "📋 Quick Commands:"
echo "  Start monitoring: curl -X POST http://localhost:5000/api/monitoring/start -H 'Content-Type: application/json' -d '{\"driver_id\": 1, \"vehicle_id\": 1, \"trip_id\": 1}'"
echo "  Stop monitoring:  curl -X POST http://localhost:5000/api/monitoring/stop"
echo "  Check status:     curl http://localhost:5000/api/monitoring/status"
echo "  View incidents:   curl http://localhost:8080/api/v1/incidents"
echo ""
