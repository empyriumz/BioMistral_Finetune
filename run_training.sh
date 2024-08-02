#!/bin/bash

# Create a unique folder name with timestamp
run_folder="runs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_folder"

# Define log file name in the new folder
log_file="$run_folder/training.log"

# Start time
start_time=$(date +%s)

# Modify the command to save output to the new folder
cmd="python -u train_binary_classification.py --config config.yaml --output_dir $run_folder"

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

echo "All outputs saved in: $run_folder"