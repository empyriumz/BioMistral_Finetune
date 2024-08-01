#!/bin/bash
# Start time
# Define log file name with date and time
log_file="logs/binary_classification_no_structured_variable_$(date +%m%d%Y_%H%M%S).log"

start_time=$(date +%s)
cmd="python -u train_binary_classification.py --config config.yaml"
# Write the command to the terminal and log file
echo "Running command: $cmd" | tee -a "$log_file"

# Run the python script and redirect both stdout and stderr to log file
eval $cmd 2>&1 | tee -a "$log_file"

# Capture the end time
end_time=$(date +%s)
# Calculate and display the duration
duration=$((end_time - start_time))
echo "Job finished in $duration seconds" | tee -a "$log_file"

# Log the current date and time
date | tee -a "$log_file"
