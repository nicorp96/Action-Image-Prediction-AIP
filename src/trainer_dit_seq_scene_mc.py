import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from src.dataset.data_set_seq_scene import RobotDatasetSeqSceneCanny
from src.dataset.data_set_bridge import BridgeDatasetMC
from src.models.diffusion_frame_pred import (
    DiTActionSeqISimMultiCondi,
    LinearDiTActionMultiCondi,
)
from copy import deepcopy
from src.models.difussion_utils.schedule import create_diffusion_seq
from diffusers.models import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler
from src.trainer_base import TrainerBase
from torchvision.utils import save_image
from diffusers.optimization import get_scheduler
import matplotlib.pyplot as plt
from diffusers.schedulers import PNDMScheduler
import numpy as np
from sklearn.manifold import TSNE
import wandb
import yaml
from einops import rearrange
import os
from .utils import update_ema, requires_grad, NormalizeVideo
from src.metrics.video_metrics import VideoMetrics


def model_factory(model_config):
    model_classes = {
        "DiTActionSeqISimMultiCondi": DiTActionSeqISimMultiCondi,
        "LinearDiTActionMultiCondi": LinearDiTActionMultiCondi,
    }
    type = model_config["type"]
    if type not in model_classes:
        raise ValueError(f"Unknown model type: {type}")

    model_class = model_classes[type]
    return model_class(model_config)


class DiTTrainerSceneMC(TrainerBase):
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
        self.mask_num = model_config["mask_n"]
        self.eval_save_real_dir = os.path.join(
            base, self.config["trainer"]["eval_save_real"]
        )
        self.eval_save_gen_dir = os.path.join(
            base, self.config["trainer"]["eval_save_gen"]
        )

        self.ema = deepcopy(self.model_dit).to(
            self.device
        )  # Create an EMA of the model for use after training
        requires_grad(self.ema, False)
        self.model_ddp = DDP(self.model_dit.to(self.device), device_ids=[self.rank])

        self.diffusion_s = create_diffusion_seq(
            timestep_respacing=self.config["diffusion"]["timestep_respacing"],
            learn_sigma=model_config["learn_sigma"],
        )
        self.metrics = VideoMetrics(device=self.device)
        # Load VAE
        vae_config = self.config["vae"]
        self.vae = AutoencoderKL.from_pretrained(
            vae_config["path"], subfolder=vae_config["subfolder"]
        ).to(self.device)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                NormalizeVideo(),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize((self.image_size, self.image_size)),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Load dataset
        self.dataset = BridgeDatasetMC(
            json_dir=self.data_path,
            transform=transform,
            sequence_length=model_config["seq_len"],
        )
        self.sampler = DistributedSampler(
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
            sampler=self.sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            # collate_fn=collate_fn,
        )
        self.val_dataset = BridgeDatasetMC(
            json_dir=self.data_path,
            transform=transform,
            sequence_length=model_config["seq_len"],
        )
        self.val_loader = (
            DataLoader(self.val_dataset, batch_size=32, shuffle=False)
            if self.val_dataset
            else None
        )
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model_ddp.parameters(),
            lr=self.config["trainer"]["learning_rate"],
            weight_decay=self.config["trainer"]["weight_decay"],
        )
        self.lr_scheduler = get_scheduler(
            name="constant",
            optimizer=self.optimizer,
            # num_warmup_steps=self.config["trainer"]["lr_warmup_steps"] * self.config["trainer"]["gradient_accumulation_steps"] ,
            num_training_steps=self.n_epochs
            * self.config["trainer"]["gradient_accumulation_steps"],
        )
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

    def __setup__DDP(self, distributed_config):
        assert (
            torch.cuda.is_available()
        ), "Training currently requires at least one GPU."

        dist.init_process_group(
            distributed_config["backend"], world_size=distributed_config["world_size"]
        )
        assert (
            self.batch_size % dist.get_world_size() == 0
        ), f"Batch size must be divisible by the world size."
        self.rank = dist.get_rank()
        self.device = torch.device(
            self.rank % torch.cuda.device_count()
        )  # self.cuda_num
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
            self.sampler.set_epoch(epoch)
            train_loss = self._train_one_epoch(step)
            if self.val_loader is not None:
                if step % self.config["trainer"]["val_num"] == 0:
                    val_loss = self._validate()
                    if val_loss < best_loss:
                        best_loss = val_loss
                        self._save_checkpoint(step)
                if step % self.config["trainer"]["val_num_gen"] == 0:
                    self.validate_video_generation(step)

            print(
                f"(Epoch={epoch:04d}) Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

            step += 1

        self.model_ddp.eval()
        self._save_checkpoint(step)
        dist.destroy_process_group()
        print("Training finished.")

    def _train_one_epoch(self, step):
        self.model_ddp.train()
        running_loss = 0.0
        for frames, action, canny in tqdm(self.data_loader, desc="Training"):
            frames, action, canny = (
                frames.to(self.device, dtype=torch.float32),
                action.to(self.device, dtype=torch.float32),
                canny.to(self.device, dtype=torch.float32),
            )

            with torch.no_grad():
                b, _, _, _, _ = frames.shape
                x = rearrange(frames, "b f c h w -> (b f) c h w").contiguous()
                x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, "(b f) c h w -> b f c h w", b=b).contiguous()

            t = torch.randint(
                0, self.diffusion_s.num_timesteps, (x.shape[0],), device=self.device
            )

            model_kwargs = dict(
                a=action, c_m=canny[:, : self.mask_num, :, :], mask_frame_num=2
            )
            loss_dict = self.diffusion_s.training_losses(
                self.model_ddp, x, t, model_kwargs
            )
            loss = loss_dict["loss"].mean()
            if self.config["trainer"]["wandb_log"]:
                wandb.log({"loss": loss})
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            update_ema(self.ema, self.model_ddp.module)
            running_loss += loss.item()
        return running_loss

    def _validate(self):
        self.model_ddp.eval()
        running_loss = 0.0
        with torch.no_grad():
            for step_val, (frames, action, canny) in enumerate(
                tqdm(self.data_loader, desc="Training")
            ):
                frames, action, canny = (
                    frames.to(self.device, dtype=torch.float32),
                    action.to(self.device, dtype=torch.float32),
                    canny.to(self.device, dtype=torch.float32),
                )
                b, _, _, _, _ = frames.shape
                x = rearrange(frames, "b f c h w -> (b f) c h w").contiguous()
                x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, "(b f) c h w -> b f c h w", b=b).contiguous()
                t = torch.randint(
                    0, self.diffusion_s.num_timesteps, (x.shape[0],), device=self.device
                )
                model_kwargs = dict(
                    a=action, c_m=canny[:, : self.mask_num, :, :], mask_frame_num=2
                )
                loss_dict = self.diffusion_s.training_losses(
                    self.model_ddp, x, t, model_kwargs
                )
                loss = loss_dict["loss"].mean()
                running_loss += loss
        running_loss = running_loss.detach()
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        running_loss = running_loss.item() / dist.get_world_size()
        self.model_ddp.train()
        return running_loss / step_val

    def _save_checkpoint(self, step):
        checkpoint = {
            "model": self.model_ddp.module.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/scene_mc_epoch_{step}.pth")

    def save_image_actions(self, step, x, frames, actions_real, canny):
        model_kwargs = dict(
            a=actions_real, c_m=canny[:, : self.mask_num, :, :], mask_frame_num=2
        )
        b, f, c, h, w = x.shape
        z = torch.randn_like(
            x,
            device=self.device,
        )
        z[:, : self.mask_num, :, :] = x[:, : self.mask_num, :, :]
        samples = self.diffusion_s.p_sample_loop(
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
            nrow=5,
            normalize=True,
            value_range=(-1, 1),
        )
        save_image(
            frames[batchsize, :, :, :, :],
            self.eval_save_real_dir + f"_{step}.png",
            nrow=5,
            normalize=True,
            value_range=(-1, 1),
        )
        avg_psnr, avg_ssim = self.metrics.evaluate_video(
            samples[batchsize, :, :, :, :], frames[batchsize, :, :, :, :]
        )
        if self.config["trainer"]["wandb_log"]:
            wandb.log({"psnr": avg_psnr})
            wandb.log({"ssim": avg_ssim})


if __name__ == "__main__":
    trainer = DiTTrainerSceneMC(
        config_path="/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/config/dit.yaml"
    )
    # wandb.init(
    #     # project=trainer.config["wandb"]["project"],
    #     # entity=trainer.config["wandb"]["entity"],
    # )
    trainer.train()
