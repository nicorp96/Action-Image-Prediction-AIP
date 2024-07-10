from src.trainer_gan import TrainerGAN
from src.trainer_gan_aux import TrainerGANAux
from src.trainer_gan_fk import TrainerGANFk
from src.trainer_gan_d_g_c import TrainerGANCAGD
from src.trainer_gan_res import TrainerGANReS
import os
import argparse


def get_trainer(trainer_type, config):
    trainers = {
        "TrainerGAN": TrainerGAN,
        "TrainerGANAux": TrainerGANAux,
        "TrainerGANFk": TrainerGANFk,
        "TrainerGANCAGD": TrainerGANCAGD,
        "TrainerGANReS": TrainerGANReS,
    }
    if trainer_type in trainers:
        return trainers[trainer_type](config)
    else:
        raise ValueError(f"Trainer type '{trainer_type}' is not recognized.")


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Training GANs Action Conditioned Image Prediction"
        )
        parser.add_argument(
            "-c",
            "--config",
            help="path to json config file",
            default="config/gan_condition_gd.yaml",
        )
        parser.add_argument(
            "-t",
            "--trainer",
            help="Different Trainer for GANs",
            default="TrainerGANCAGD",
            choices=[
                "TrainerGAN",
                "TrainerGANAux",
                "TrainerGANFk",
                "TrainerGANCAGD",
                "TrainerGANReS",
            ],
        )

        args = parser.parse_args()
        base_dir = os.getcwd()

        # Default configuration for demonstration
        config_dir = os.path.join(base_dir, args.config)

        trainer = get_trainer(args.trainer, config_dir)
        trainer.train()

    except ValueError as exc:
        print("Exception Occurred")
        print(exc)
    finally:
        print("Program was successfully finalized")


if __name__ == "__main__":
    main()
