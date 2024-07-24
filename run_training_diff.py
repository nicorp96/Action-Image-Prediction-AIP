from src.trainer_dit import DiTTrainer
from src.trainer_dit_mod import DiTTrainerMod
from src.trainer_dit_seq import DiTTrainerV
from src.trainer_dit_seq_act import DiTTrainerScene
from src.trainer_dit_seq_act_frames import DiTTrainerActFrames
import os
import wandb


def get_trainer(trainer_type, config):
    trainers = {
        "DiTTrainer": DiTTrainer,
        "DiTTrainerMod": DiTTrainerMod,
        "DiTTrainerV": DiTTrainerV,
        "DiTTrainerScene": DiTTrainerScene,
        "DiTTrainerActFrames": DiTTrainerActFrames,
    }
    if trainer_type in trainers:
        return trainers[trainer_type](config)
    else:
        raise ValueError(f"Trainer type '{trainer_type}' is not recognized.")


def main():
    try:
        config = "config/dit_mod_seq_act_frames.yaml"

        trainer_name = "DiTTrainerActFrames"
        base_dir = os.getcwd()

        # Default configuration for demonstration
        config_dir = os.path.join(base_dir, config)

        trainer = get_trainer(trainer_name, config_dir)

        trainer.train()

    except ValueError as exc:
        print("Exception Occurred")
        print(exc)
    finally:
        print("Program was successfully finalized")


if __name__ == "__main__":
    main()
