from src.trainer_dit import DiTTrainer
from src.trainer_dit_mod import DiTTrainerMod
from src.trainer_dit_seq import DiTTrainerV
from src.trainer_dit_seq_act import DiTTrainerActScene
from src.trainer_dit_seq_act_frames import DiTTrainerActFrames
from src.trainer_dit_seq_act_frames_att import DiTTrainerActFramesAtt
from src.trainer_dit_seq_scene import DiTTrainerScene
import os
import argparse


def get_trainer(trainer_type, config):
    trainers = {
        "DiTTrainer": DiTTrainer,
        "DiTTrainerMod": DiTTrainerMod,
        "DiTTrainerV": DiTTrainerV,
        "DiTTrainerActScene": DiTTrainerActScene,
        "DiTTrainerActFrames": DiTTrainerActFrames,
        "DiTTrainerScene": DiTTrainerScene,
        "DiTTrainerActFramesAtt": DiTTrainerActFramesAtt,
    }
    if trainer_type in trainers:
        return trainers[trainer_type](config)
    else:
        raise ValueError(f"Trainer type '{trainer_type}' is not recognized.")


def main():
    try:

        base_dir = os.getcwd()

        parser = argparse.ArgumentParser(description="Simulation Enviroment UR5")
        parser.add_argument(
            "-c", "--config", help="name of config file", default="dit_mod_seq.yaml"
        )
        parser.add_argument(
            "-t",
            "--trainer",
            help="Trainer",
            default="DiTTrainerActScene",
            choices=[
                "DiTTrainer",
                "DiTTrainerMod",
                "DiTTrainerV",
                "DiTTrainerActScene",
                "DiTTrainerActFrames",
                "DiTTrainerActFramesAtt",
            ],
        )

        parser.add_argument("--local-rank")
        args = parser.parse_args()
        trainer_name = args.trainer
        config = f"{args.config}"

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
