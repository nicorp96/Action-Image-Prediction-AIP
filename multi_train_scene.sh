#!/bin/bash

# Define the folder containing the YAML files
CONFIG_FILE="config/DiTTrainerSceneMC/dit_mod_seq_scene.yaml"
# Define the default trainer type
DEFAULT_TRAINER="DiTTrainerSceneMC"

# Define the folder containing the YAML files
CONFIG_FILE2="config/DiTTrainerScene/dit_mod_seq_scene_bridge.yaml"
# Define the default trainer type
DEFAULT_TRAINER2="DiTTrainerScene"

echo "Processing $CONFIG_FILE..."
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12355 run_training_diff.py -c "$CONFIG_FILE" -t "$DEFAULT_TRAINER"

echo "Processing $CONFIG_FILE2..."
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12355 run_training_diff.py -c "$CONFIG_FILE2" -t "$DEFAULT_TRAINER2"