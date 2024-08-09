# Action-Image-Prediction-AIP

## Training with multiple GPUS:

### Launch Distributed Training on Multiple GPUs

To train the model using all GPUs available on your server, you need to launch the training script using PyTorch's `torchrun` utility. Follow these steps:

#### 1. Ensure Your Configuration is Set Correctly

Make sure your configuration file (`config.yaml`) has the correct settings for distributed training. Specifically, update the `world_size` under the `distributed` section to match the number of GPUs you want to use:

```yaml
distributed:
  backend: "nccl"
  world_size: 2  # Number of GPUs to use
```
#### 2. Run the Training Script Using torchrun

```bash
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12355 run_training_diff.py -c config/DiTTrainerSceneMC/dit_mod_seq_scene.yaml -t DiTTrainerSceneMC
```

- **`--nproc_per_node=2`**: Specifies that 2 processes should be started, one for each GPU.
- **`--nnodes=1`**: Indicates that you are running on a single node (server).
- **`--node_rank=0`**: The rank of this node (useful if you are training across multiple nodes).
- **`--master_addr="localhost"`**: The address of the master node (use `localhost` for single-node setups).
- **`--master_port=12355`**: The port used for communication between processes (can be any available port).


## Script multi GPU training

```bash
# 1.
chmod +x multi_train_dit_multi_gpu.sh
# 2. 
./multi_train_dit.sh
```