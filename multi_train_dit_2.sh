#!/bin/bash

# Define the folder containing the YAML files
CONFIG_FOLDER="config/DiTTrainerActScene"
# Define the default trainer type
DEFAULT_TRAINER="DiTTrainerActScene"

# Check if the directory exists
if [ ! -d "$CONFIG_FOLDER" ]; then
    echo "Directory $CONFIG_FOLDER does not exist."
    exit 1
fi

# Find all YAML files in the folder
yaml_files=($CONFIG_FOLDER/*.yaml)

# Check if any YAML files were found
if [ ${#yaml_files[@]} -eq 0 ]; then
    echo "No YAML files found in $CONFIG_FOLDER."
    exit 1
fi

for config_file in "${yaml_files[@]}"; do
    # Extract the base name of the file (without path)
    config_filename=$(basename "$config_file")
    echo "Processing $config_file..."
    python3 -m torch.distributed.launch run_training_diff.py -c "$config_file" -t "$DEFAULT_TRAINER"
done

echo "Finished Training"
exit 1