#!/bin/bash

# Define the base directory where config files are stored
CONFIG_DIR="configs"

# Loop through each subdirectory (small, base, large)
for subdir in "$CONFIG_DIR"/*; do
    # Ensure it's a directory
    if [ -d "$subdir" ]; then
        # Loop through each YAML file in the subdirectory
        for config in "$subdir"/*.yaml; do
            # Ensure it's a file
            if [ -f "$config" ]; then
                echo "Running: python main.py --config $config"
                python main.py --config "$config"
            fi
        done
    fi
done
