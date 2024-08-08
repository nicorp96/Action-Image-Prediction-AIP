# Action-Image-Prediction-AIP

```bash
python3 -m torch.distributed.launch run_training_diff.py -c config/DiTTrainerActScene/dit_mod_seq_scene.yaml -t DiTTrainerScene
```

## Script multi training

```bash
# 1.
chmod +x multi_train_dit.sh
# 2. 
./multi_train_dit.sh
```