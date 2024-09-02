import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from src.dataset.data_set_seq_scene import RobotDatasetSeqScene, collate_fn
from src.dataset.data_set_bridge import BridgeDataset
from src.models.diffusion_frame_pred import (
    DiTActionSeqISim,
    LinearDiTActionSeqISim,
    DiTActionSeqFrameAtt,
    DiTActionSeqJoint,
)
from copy import deepcopy
from src.models.difussion_utils.schedule import create_diffusion_seq
from diffusers.models import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler
from src.trainer_base import TrainerBase
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import wandb
import yaml
from einops import rearrange
import os
from .utils import update_ema, requires_grad, NormalizeVideo
from src.metrics.video_metrics import VideoMetrics
from diffusers.optimization import get_scheduler
from diffusers.schedulers import PNDMScheduler
from .pipelines_video_gen import Trajectory2VideoGenPipeline
from accelerate import Accelerator


def model_factory(model_config):
    model_classes = {
        "DiTActionSeqISim": DiTActionSeqISim,
        "LinearDiTActionSeqISim": LinearDiTActionSeqISim,
        "DiTActionSeqFrameAtt": DiTActionSeqFrameAtt,
        "DiTActionSeqJoint": DiTActionSeqJoint,
    }
    type = model_config["type"]
    if type not in model_classes:
        raise ValueError(f"Unknown model type: {type}")

    model_class = model_classes[type]
    return model_class(model_config)


class DiTTrainerScene(TrainerBase):
    def __init__(self, config_dir, val_dataset=None):
        super().__init__(config_dir)
        self.__load_config__()
        self.accelerator = Accelerator()

        if self.config["trainer"]["wandb_log"]:
            wandb.init()

        # Trainer settings
        base = os.getcwd()
        self.batch_size = self.config["trainer"]["batch_size"]
        self.global_seed = self.config["trainer"]["global_seed"]
        self.image_size = self.config["trainer"]["image_size"]
        self.n_epochs = self.config["trainer"]["n_epochs"]
        self.data_path = os.path.join(base, self.config["trainer"]["data_path"])

        # Model settings
        model_config = self.config["model"]
        self.model_dit = torch.compile(model_factory(model_config))

        self.diffusion_s = create_diffusion_seq(
            timestep_respacing=self.config["diffusion"]["timestep_respacing"],
            learn_sigma=model_config["learn_sigma"],
        )
        self.metrics = VideoMetrics(device=self.accelerator.device)

        # Load VAE
        vae_config = self.config["vae"]
        self.vae = AutoencoderKL.from_pretrained(
            vae_config["path"], subfolder=vae_config["subfolder"]
        )

        self.eval_save_real_dir = os.path.join(
            base, self.config["trainer"]["eval_save_real"]
        )
        self.eval_save_gen_dir = os.path.join(
            base, self.config["trainer"]["eval_save_gen"]
        )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                NormalizeVideo(),
                transforms.Resize((self.image_size, self.image_size)),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Load dataset
        # self.dataset = RobotDatasetSeqScene(
        #     data_path=self.data_path,
        #     transform=transform,
        #     seq_l=model_config["seq_len"],
        # )
        self.dataset = BridgeDataset(
            json_dir=self.data_path,
            transform=transform,
            sequence_length=model_config["seq_len"],
        )

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        # self.val_dataset = RobotDatasetSeqScene(
        #     data_path=self.data_path,
        #     transform=transform,
        #     seq_l=model_config["seq_len"],
        # )
        self.val_dataset = BridgeDataset(
            json_dir=self.data_path,
            transform=transform,
            sequence_length=model_config["seq_len"],
        )
        if self.val_dataset is not None:
            self.val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)
        else:
            self.val_loader = None

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model_dit.parameters(),
            lr=self.config["trainer"]["learning_rate"],
            weight_decay=self.config["trainer"]["weight_decay"],
        )
        self.lr_scheduler = get_scheduler(
            name="constant",
            optimizer=self.optimizer,
            num_training_steps=self.n_epochs
            * self.config["trainer"]["gradient_accumulation_steps"],
        )

        # Prepare with Accelerator
        (
            self.model_dit,
            self.optimizer,
            self.data_loader,
            self.lr_scheduler,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.model_dit, self.optimizer, self.data_loader, self.lr_scheduler, self.val_loader
        )
        self.vae = self.vae.to(device=self.accelerator.device)
        self.ema = deepcopy(self.model_dit)  # EMA without .to(self.device)
        requires_grad(self.ema, False)

        self.scheduler = None
        if self.config["diffusion"]["sample_method"] == "PNDM":
            self.scheduler = PNDMScheduler.from_pretrained(
                self.config["diffusion"]["scheduler_path"],
                beta_start=self.config["diffusion"]["beta_start"],
                beta_end=self.config["diffusion"]["beta_end"],
                beta_schedule=self.config["diffusion"]["beta_schedule"],
                variance_type=self.config["diffusion"]["variance_type"],
            )
        elif self.config["diffusion"]["sample_method"]:
            self.scheduler = PNDMScheduler.from_pretrained(
                self.config["diffusion"]["scheduler_path"],
                beta_start=self.config["diffusion"]["beta_start"],
                beta_end=self.config["diffusion"]["beta_end"],
                beta_schedule=self.config["diffusion"]["beta_schedule"],
                variance_type=self.config["diffusion"]["variance_type"],
            )

    def __load_config__(self):
        # Load config
        with open(self.config_dir, "r") as file:
            self.config = yaml.safe_load(file)

    def train(self):
        self.accelerator.wait_for_everyone()
        update_ema(self.ema, self.model_dit, decay=0)
        self.model_dit.train()
        self.ema.eval()
        best_loss = float("inf")
        step = 0
        val_loss = 0

        for epoch in range(self.n_epochs):
            train_loss = self._train_one_epoch(step)
            if (
                self.val_loader is not None
                and step % self.config["trainer"]["val_num"] == 0
            ):
                val_loss = self._validate()
                if step % self.config["trainer"]["val_num_gen"] == 0:
                    self.validate_video_generation(step)
                    if val_loss < best_loss:
                        best_loss = val_loss
                        self._save_checkpoint(step)

            print(
                f"(Epoch={epoch:04d}) Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )
            step += 1

        self.model_dit.eval()
        self._save_checkpoint(step)
        print("Training finished.")

    def _train_one_epoch(self, step):
        self.model_dit.train()
        running_loss = 0.0
        for next_seq, action in tqdm(self.data_loader, desc="Training"):
            next_seq, action = (
                next_seq.to(dtype=torch.float32),
                action.to(dtype=torch.float32),
            )

            with torch.no_grad():
                b, _, _, _, _ = next_seq.shape
                x = rearrange(next_seq, "b f c h w -> (b f) c h w").contiguous()
                x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, "(b f) c h w -> b f c h w", b=b).contiguous()

            t = torch.randint(
                0,
                self.diffusion_s.num_timesteps,
                (x.shape[0],),
                device=self.accelerator.device,
            )
            model_kwargs = dict(a=action, mask_frame_num=2)
            loss_dict = self.diffusion_s.training_losses(
                self.model_dit, x, t, model_kwargs
            )
            loss = loss_dict["loss"].mean()

            if self.config["trainer"]["wandb_log"]:
                self.accelerator.log({"loss": loss})

            self.optimizer.zero_grad()
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            update_ema(self.ema, self.model_dit)
            running_loss += loss.item()

        return running_loss

    def _validate(self):
        self.model_dit.eval()
        running_loss = 0.0
        with torch.no_grad():
            for step_val, (next_seq, action) in enumerate(
                tqdm(self.val_loader, desc="Validation")
            ):
                next_seq, action = (
                    next_seq.to(device=self.accelerator.device, dtype=torch.float32),
                    action.to(device=self.accelerator.device,dtype=torch.float32),
                )
                b, _, _, _, _ = next_seq.shape
                x = rearrange(next_seq, "b f c h w -> (b f) c h w").contiguous()
                x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, "(b f) c h w -> b f c h w", b=b).contiguous()
                t = torch.randint(
                    0,
                    self.diffusion_s.num_timesteps,
                    (x.shape[0],),
                    device=self.accelerator.device,
                )
                model_kwargs = dict(a=action, mask_frame_num=2)
                loss_dict = self.diffusion_s.training_losses(
                    self.model_dit, x, t, model_kwargs
                )
                loss = loss_dict["loss"].mean()
                running_loss += loss.item()
        return running_loss / (step_val + 1)

    def validate_video_generation(self, step):
        self.model_dit.eval()
        self.vae.eval()

        # Use Accelerator device
        device = self.accelerator.device

        batch_id = list(range(0, len(self.val_dataset), int(len(self.val_dataset) / 3)))

        # Load data onto the device
        batch_list = [self.val_dataset.__getitem__(id) for id in batch_id]
        actions = torch.cat(
            [action.unsqueeze(0) for i, (video, action) in enumerate(batch_list)],
            dim=0,
        ).to(device, non_blocking=True, dtype=torch.float32)
        true_video = torch.cat(
            [video.unsqueeze(0) for i, (video, action) in enumerate(batch_list)],
            dim=0,
        ).to(device, non_blocking=True, dtype=torch.float32)

        mask_frame_num = self.config["model"]["mask_n"]
        mask_x = true_video[:, 0:mask_frame_num]

        # Inference with no gradients
        with torch.no_grad():
            b, f, _, _, _ = mask_x.shape
            mask_x = rearrange(mask_x, "b f c h w -> (b f) c h w").contiguous()
            mask_x = (
                self.vae.encode(mask_x)
                .latent_dist.sample()
                .mul_(self.vae.config.scaling_factor)
            )
            mask_x = rearrange(mask_x, "(b f) c h w -> b f c h w", b=b, f=f)

            videogen_pipeline = Trajectory2VideoGenPipeline(
                vae=self.vae,
                scheduler=self.scheduler,
                transformer=self.accelerator.unwrap_model(self.ema),
            )
            print(f"Generation total {mask_x.size(0)} videos")

            # Generate videos
            videos, latents = videogen_pipeline(
                actions,
                mask_x=mask_x,
                video_length=self.config["model"]["seq_len"],
                height=self.image_size,
                width=self.image_size,
                num_inference_steps=self.config["diffusion"][
                    "infer_num_sampling_steps"
                ],
                guidance_scale=self.config["diffusion"]["guidance_scale"],
                device=device,
                output_type="both",
            )

            videos = torch.cat(
                [
                    true_video[:, 0:mask_frame_num, :, :, :],
                    videos[:, mask_frame_num:, :, :, :],
                ],
                dim=1,
            )

            avg_psnr, avg_ssim, avg_fid = self.metrics.evaluate_video(
                videos, true_video
            )
            videos = rearrange(videos, "b f c h w -> (b f) c h w")
            true_video = rearrange(true_video, "b f c h w -> (b f) c h w")

            print(
                "\n------------------------|| Metrics Validation ||--------------------------------"
            )
            print(f" || psnr: {avg_psnr}  ssim: {avg_ssim}  fid: {avg_fid} ||")
            if self.config["trainer"]["wandb_log"]:
                wandb.log({"psnr": avg_psnr})
                wandb.log({"ssim": avg_ssim})
                wandb.log({"fid": avg_fid})

            save_image(
                videos,
                self.eval_save_gen_dir + f"_{step}.png",
                nrow=self.config["model"]["seq_len"],
                normalize=True,
                value_range=(0, 1),
            )
            save_image(
                true_video,
                self.eval_save_real_dir + f"_{step}.png",
                nrow=self.config["model"]["seq_len"],
                normalize=True,
            )

    def _save_checkpoint(self, step):
        checkpoint = {
            "model": self.accelerator.unwrap_model(self.model_dit).state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/scene_scene_epoch_{step}.pth")


if __name__ == "__main__":
    trainer = DiTTrainerScene(
        config_path="/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/config/dit.yaml"
    )
    # wandb.init(
    #     # project=trainer.config["wandb"]["project"],
    #     # entity=trainer.config["wandb"]["entity"],
    # )
    trainer.train()
