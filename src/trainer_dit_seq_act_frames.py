import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset.data_set_seq_trj import RobotDatasetSeqTrj
from src.models.diffusion_transformer_action import (
    DiTActionFramesSeq,
    DiTActionFramesSeq2,
    DiTActionFramesSeq3,
    DiTActionFramesSeq4,
    DiTActionFramesSeq5,
    DiTActionFramesSeq6,
)
from copy import deepcopy
from src.models.difussion_utils.schedule import create_diffusion_seq_act
from diffusers.models import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler
from src.trainer_base import TrainerBase
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import wandb
import yaml
from einops import rearrange
from src.utils import unnormilize_action_seq__torch
from diffusers.optimization import get_scheduler
from src.metrics.video_metrics import VideoMetrics
import os
from .utils import update_ema, requires_grad


def model_factory(model_config):
    model_classes = {
        "DiTActionFramesSeq": DiTActionFramesSeq,
        "DiTActionFramesSeq2": DiTActionFramesSeq2,
        "DiTActionFramesSeq3": DiTActionFramesSeq3,
        "DiTActionFramesSeq4": DiTActionFramesSeq4,
        "DiTActionFramesSeq5": DiTActionFramesSeq5,
        "DiTActionFramesSeq6": DiTActionFramesSeq6,
    }
    type = model_config["type"]
    if type not in model_classes:
        raise ValueError(f"Unknown model type: {type}")

    model_class = model_classes[type]
    return model_class(model_config)


class DiTTrainerActFrames(TrainerBase):
    def __init__(self, config_dir, val_dataset=None):
        super().__init__(config_dir)
        self.__load_config__()
        if self.config["trainer"]["wandb_log"]:
            wandb.init()
        # Trainer settings
        base = os.getcwd()
        self.batch_size = self.config["trainer"]["batch_size"]
        self.global_seed = self.config["trainer"]["global_seed"]
        self.image_size = self.config["trainer"]["image_size"]
        self.n_epochs = self.config["trainer"]["n_epochs"]
        self.data_path = os.path.join(base, self.config["trainer"]["data_path"])
        self.cuda_num = self.config["trainer"]["cuda_num"]

        self.__setup__DDP(self.config["distributed"])
        # Model settings
        model_config = self.config["model"]
        self.model_dit = model_factory(model_config)
        self.eval_save_real_dir = os.path.join(
            base, self.config["trainer"]["eval_save_real"]
        )
        self.eval_save_gen_dir = os.path.join(
            base, self.config["trainer"]["eval_save_gen"]
        )
        self.eval_act_save_gen_dir = os.path.join(
            base, self.config["trainer"]["action_save_gen"]
        )
        self.eval_act_save_real_dir = os.path.join(
            base, self.config["trainer"]["action_save_real"]
        )
        self.mask_num = model_config["mask_n"]
        self.ema = deepcopy(self.model_dit).to(
            self.device
        )  # Create an EMA of the model for use after training
        requires_grad(self.ema, False)
        self.model_ddp = DDP(
            self.model_dit.to(self.device),
            device_ids=[self.rank],
            # find_unused_parameters=True,
        )

        self.diffusion_s = create_diffusion_seq_act(
            timestep_respacing=self.config["diffusion"]["timestep_respacing"],
            learn_sigma=self.config["model"]["learn_sigma"],
        )

        # Load VAE
        vae_config = self.config["vae"]
        self.vae = AutoencoderKL.from_pretrained(
            vae_config["path"], subfolder=vae_config["subfolder"]
        ).to(self.device)
        self.metrics = VideoMetrics(device=self.device, vae=self.vae)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Load dataset
        self.dataset = RobotDatasetSeqTrj(
            data_path=self.data_path, transform=transform, seq_l=model_config["seq_len"]
        )
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=dist.get_world_size(),
            rank=self.rank,
            shuffle=True,
            seed=self.global_seed,
        )

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=int(self.batch_size // dist.get_world_size()),
            shuffle=False,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            # collate_fn=collate_fn,
        )

        self.val_loader = (
            DataLoader(val_dataset, batch_size=32, shuffle=False)
            if val_dataset
            else None
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model_ddp.parameters(),
            lr=self.config["trainer"]["learning_rate"],
            weight_decay=self.config["trainer"]["weight_decay"],
        )
        self.lr_scheduler = get_scheduler(
            name="constant",
            optimizer=self.optimizer,
            num_training_steps=self.n_epochs,
        )

    def __load_config__(self):
        # Load config
        with open(self.config_dir, "r") as file:
            self.config = yaml.safe_load(file)

    def __setup__DDP(self, distributed_config):
        assert (
            torch.cuda.is_available()
        ), "Training currently requires at least one GPU."

        dist.init_process_group(distributed_config["backend"])
        assert (
            self.batch_size % dist.get_world_size() == 0
        ), f"Batch size must be divisible by the world size."
        self.rank = dist.get_rank()
        self.device = torch.device(self.cuda_num)
        self.seed = self.global_seed * dist.get_world_size() + self.rank
        torch.manual_seed(self.seed)
        torch.cuda.set_device(self.device)
        print(
            f"Starting rank={self.rank}, seed={self.seed}, world_size={dist.get_world_size()}."
        )

    def train(self):
        # Prepare models for training:
        update_ema(
            self.ema, self.model_ddp.module, decay=0
        )  # Ensure EMA is initialized with synced weights
        self.model_ddp.train()  # important! This enables embedding dropout for classifier-free guidance
        self.ema.eval()  # EMA model should always be in eval mode
        best_loss = float("inf")
        step = 0
        val_loss = None
        for epoch in range(self.n_epochs):
            train_loss = self._train_one_epoch(step)
            if step % self.config["trainer"]["val_num"] == 0:
                val_loss = self._validate_one_epoch() if self.val_loader else None
            print(
                f"Epoch {epoch+1}/{self.n_epochs}, Training Loss: {train_loss:.4f}",
                end="",
            )
            if val_loss is not None:
                print(f", Validation Loss: {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    self._save_checkpoint()
            else:
                print(" ")

            step += 1

        self.model_ddp.eval()
        self._save_checkpoint()
        print("Training finished.")

    def _train_one_epoch(self, step):
        self.model_ddp.train()
        running_loss = 0.0
        for current_img, next_seq, action in tqdm(self.data_loader, desc="Training"):
            current_img, next_seq, action = (
                current_img.to(device=self.device, dtype=torch.float32),
                next_seq.to(self.device, dtype=torch.float32),
                action.to(self.device, dtype=torch.float32),
            )

            with torch.no_grad():
                b, _, _, _, _ = next_seq.shape
                x = rearrange(next_seq, "b f c h w -> (b f) c h w").contiguous()
                x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, "(b f) c h w -> b f c h w", b=b).contiguous()
                goal_img = (
                    self.vae.encode(current_img[:1, :, :])
                    .latent_dist.sample()
                    .mul_(0.18215)
                )

            t = torch.randint(
                0, self.diffusion_s.num_timesteps, (x.shape[0],), device=self.device
            )

            model_kwargs = dict(a=action, img_c=goal_img, mask_frame_num=2)
            loss_dict = self.diffusion_s.training_losses(
                self.model_ddp, x, t, model_kwargs
            )
            loss = loss_dict["loss"].mean()
            loss_act = loss_dict["loss_act"].mean()
            if self.config["trainer"]["wandb_log"]:
                wandb.log({"loss": loss})
                wandb.log({"loss_act": loss_act})
            total_loss = loss + loss_act
            total_loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            update_ema(self.ema, self.model_ddp.module)
            running_loss += total_loss.item()

            if step % self.config["trainer"]["val_num"] == 0:
                self.save_image_actions(
                    step=step,
                    x=x,
                    next_seq=next_seq,
                    actions_real=action,
                    goal_img=goal_img,
                )

        return running_loss

    def _validate_one_epoch(self):
        self.model_ddp.eval()
        running_loss = 0.0
        with torch.no_grad():
            for current_img, next_seq, action in tqdm(self.val_loader, desc="Training"):
                current_img, next_seq, action = (
                    current_img.to(device=self.device, dtype=torch.float32),
                    next_seq.to(self.device, dtype=torch.float32),
                    action.to(self.device, dtype=torch.float32),
                )
                b, _, _, _, _ = next_seq.shape
                x = rearrange(next_seq, "b f c h w -> (b f) c h w").contiguous()
                x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, "(b f) c h w -> b f c h w", b=b).contiguous()
                goal_img = (
                    self.vae.encode(current_img[:1, :, :])
                    .latent_dist.sample()
                    .mul_(0.18215)
                )
                model_kwargs = dict(a=action, mask_frame_num=2, img_c=goal_img)
                z = torch.randn_like(
                    x,
                    device=self.device,
                )
                z[:, : self.mask_num, :, :] = x[:, : self.mask_num, :, :]
                samples, actions_pred = self.diffusion_s.p_sample_loop(
                    self.ema,
                    z.shape,
                    z,
                    clip_denoised=False,
                    progress=True,
                    model_kwargs=model_kwargs,
                    device=self.device,
                )
                samples = rearrange(samples, "b f c h w -> (b f) c h w").contiguous()
                samples = self.vae.decode(samples / 0.18215).sample
                samples = rearrange(
                    samples, "(b f) c h w -> b f c h w", b=b
                ).contiguous()
                batchsize = np.random.choice(b)
        return running_loss / len(self.val_loader.dataset)

    def _save_checkpoint(self):
        torch.save(self.model_ddp.state_dict(), "best_model.pth")

    def save_image_actions(self, step, x, next_seq, actions_real, goal_img):
        b, _, _, _, _ = next_seq.shape
        with torch.no_grad():
            # action_n = torch.randn_like(
            #     actions_real,
            #     device=self.device,
            # )
            model_kwargs = dict(
                a=actions_real[:, :2, :], mask_frame_num=2, img_c=goal_img
            )
            z = torch.randn_like(
                x,
                device=self.device,
            )
            z[:, : self.mask_num, :, :] = x[:, : self.mask_num, :, :]
            samples, actions_pred = self.diffusion_s.p_sample_loop(
                self.ema,
                z.shape,
                z,
                clip_denoised=False,
                progress=True,
                model_kwargs=model_kwargs,
                device=self.device,
            )
            samples = rearrange(samples, "b f c h w -> (b f) c h w").contiguous()
            samples = self.vae.decode(samples / 0.18215).sample
            samples = rearrange(samples, "(b f) c h w -> b f c h w", b=b).contiguous()
            batchsize = np.random.choice(b)
            save_image(
                samples[batchsize, :, :, :, :],
                self.eval_save_gen_dir + f"_{step}.png",
                nrow=4,
                normalize=True,
                value_range=(-1, 1),
            )
            save_image(
                next_seq[batchsize, :, :, :, :],
                self.eval_save_real_dir + f"_{step}.png",
                nrow=4,
                normalize=True,
                value_range=(-1, 1),
            )
            np.savetxt(
                self.eval_act_save_gen_dir + f"_{step}.csv",
                # unnormilize_action_seq__torch(
                #     actions[batchsize, :, :].detach().cpu().numpy(),
                #     [self.dataset.max, self.dataset.min],
                # ),
                actions_pred[batchsize, :, :].detach().cpu().numpy(),
                delimiter=",",
            )
            np.savetxt(
                self.eval_act_save_real_dir + f"_{step}.csv",
                # unnormilize_action_seq__torch(
                #     action[batchsize, :, :].detach().cpu().numpy(),
                #     [self.dataset.max, self.dataset.min],
                # ),
                actions_real[batchsize, :, :].detach().cpu().numpy(),
                delimiter=",",
            )
            self.log_accuracy(
                predicted_actions=actions_pred[:batchsize, :, :].detach().cpu().numpy(),
                real_actions=actions_real[:batchsize, :, :].detach().cpu().numpy(),
            )
            avg_psnr, avg_ssim = self.metrics.evaluate_video(
                samples[batchsize, :, :, :, :], next_seq[batchsize, :, :, :, :]
            )
            if self.config["trainer"]["wandb_log"]:
                wandb.log({"psnr": avg_psnr})
                wandb.log({"ssim": avg_ssim})

    def calculate_pose_error(self, predicted_pose, real_pose):
        # Compute the Euclidean distance error for position
        position_error = np.linalg.norm(
            np.array(predicted_pose[:3]) - np.array(real_pose[:3])
        )

        # Compute the error for orientation (roll, pitch, yaw)
        orientation_error = np.linalg.norm(
            np.array(predicted_pose[3:6]) - np.array(real_pose[3:6])
        )

        return position_error, orientation_error

    def calculate_gripper_accuracy(self, predicted_gripper, real_gripper):
        if predicted_gripper == real_gripper:
            return 1
        else:
            return 0

    def log_accuracy(self, predicted_actions, real_actions):
        total_position_error = 0
        total_orientation_error = 0
        total_gripper_accuracy = 0
        b_sz, seq_l, dim = predicted_actions.shape

        for i in range(b_sz):
            for act_i in range(seq_l):
                predicted_pose = predicted_actions[i, act_i][:6]
                real_pose = real_actions[i, act_i][:6]
                predicted_gripper = predicted_actions[i, act_i][6]
                real_gripper = real_actions[i, act_i][6]

                position_error, orientation_error = self.calculate_pose_error(
                    predicted_pose, real_pose
                )
                gripper_accuracy = self.calculate_gripper_accuracy(
                    predicted_gripper, real_gripper
                )

                total_position_error += position_error
                total_orientation_error += orientation_error
                total_gripper_accuracy += gripper_accuracy

        mean_position_error = total_position_error  # / b_sz
        mean_orientation_error = total_orientation_error  # / b_sz
        # mean_gripper_accuracy = total_gripper_accuracy / b_sz
        # print(f"\n mean_position_error: {mean_position_error}")
        # print(f"\n mean_orientation_error: {mean_orientation_error}")
        # print(f"\n mean_gripper_accuracy: {mean_gripper_accuracy}")

        if self.config["trainer"]["wandb_log"]:
            wandb.log({"mean_position_error": mean_position_error})
            wandb.log({"mean_orientation_error": mean_orientation_error})
            # wandb.log({"mean_gripper_accuracy": mean_gripper_accuracy})


if __name__ == "__main__":
    trainer = DiTTrainerActFrames(
        config_path="/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/config/dit.yaml"
    )
    # wandb.init(
    #     # project=trainer.config["wandb"]["project"],
    #     # entity=trainer.config["wandb"]["entity"],
    # )
    trainer.train()
