#!/bin/bash

# Run benchmarking_test.py
python -m src.tests.benchmarking_test

# Start serve_power_sampling.py in the background
python -m src.api.serve_power_sampling &
SERVER_PID=$!

# Run api_test.py
python -m src.tests.api_test

# Kill the serve_power_sampling.py process
kill $SERVER_PID