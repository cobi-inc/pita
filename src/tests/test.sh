#!/bin/bash

# Run benchmarking_test.py
python -m src.tests.benchmarking_test

# Start serve_power_sampling.py in the background
python -m src.api.serve_power_sampling &
SERVER_PID=$!

# Wait for the server to be ready (listening on localhost:8000)
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/ > /dev/null; then
        echo "Server is up!"
        break
    fi
    sleep 1
done
# Run api_test.py
python -m src.tests.api_test

# Kill the serve_power_sampling.py process
kill $SERVER_PID