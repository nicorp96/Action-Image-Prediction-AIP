#!/bin/bash

# Define the folder containing the YAML files
CONFIG_FOLDER="config/DiTTrainerActFrames"
# Define the default trainer type
DEFAULT_TRAINER="DiTTrainerActFrames"

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
    torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12355 run_training_diff.py -c "$config_file" -t "$DEFAULT_TRAINER"
done

echo "Finished Training"
exit 1