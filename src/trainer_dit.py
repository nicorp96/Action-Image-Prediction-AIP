import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset.data_set import RobotDataset
from src.models.diffusion_dit_base import DiT
from copy import deepcopy
from src.models.difussion_utils.schedule import create_diffusion
from diffusers.models import AutoencoderKL
from torch.utils.data.distributed import DistributedSampler
from src.trainer_base import TrainerBase
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import wandb
import yaml
from .utils import update_ema, requires_grad


class DiTTrainer(TrainerBase):
    def __init__(self, config_dir, val_dataset=None):
        super().__init__(config_dir)
        self.__load_config__()
        if self.config["trainer"]["wandb_log"]:
            wandb.init()
        # Trainer settings
        self.batch_size = self.config["trainer"]["batch_size"]
        self.global_seed = self.config["trainer"]["global_seed"]
        self.image_size = self.config["trainer"]["image_size"]
        self.n_epochs = self.config["trainer"]["n_epochs"]
        self.data_path = self.config["trainer"]["data_path"]

        self.__setup__DDP(self.config["distributed"])

        # Model settings
        model_config = self.config["model"]
        self.model_dit = DiT(
            input_size=model_config["input_size"],
            patch_size=model_config["patch_size"],
            in_channels=model_config["in_channels"],
            hidden_size=model_config["hidden_size"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
            action_dim=model_config["action_dim"],
            learn_sigma=model_config["learn_sigma"],
        )

        self.ema = deepcopy(self.model_dit).to(
            self.device
        )  # Create an EMA of the model for use after training
        requires_grad(self.ema, False)
        self.model_ddp = DDP(self.model_dit.to(self.device), device_ids=[self.rank])

        self.diffusion_s = create_diffusion(
            timestep_respacing=self.config["diffusion"]["timestep_respacing"]
        )

        # Load VAE
        vae_config = self.config["vae"]
        self.vae = AutoencoderKL.from_pretrained(
            vae_config["path"], subfolder=vae_config["subfolder"]
        ).to(self.device)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Load dataset
        self.dataset = RobotDataset(data_path=self.data_path, transform=transform)
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
        )

        self.val_loader = (
            DataLoader(val_dataset, batch_size=32, shuffle=False)
            if val_dataset
            else None
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model_ddp.parameters(),
            lr=self.config["trainer"]["learning_rate"],
            weight_decay=self.config["trainer"]["weight_decay"],
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
        self.device = self.rank % torch.cuda.device_count()
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
        for epoch in range(self.n_epochs):
            train_loss = self._train_one_epoch(step)
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
        for current_img, next_img, action in tqdm(self.data_loader, desc="Training"):
            current_img, next_img, action = (
                current_img.to(device=self.device, dtype=torch.float32),
                next_img.to(self.device, dtype=torch.float32),
                action.to(self.device, dtype=torch.float32),
            )

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = self.vae.encode(next_img).latent_dist.sample().mul_(0.18215)

            t = torch.randint(
                0, self.diffusion_s.num_timesteps, (x.shape[0],), device=self.device
            )
            model_kwargs = dict(a=action)
            loss_dict = self.diffusion_s.training_losses(
                self.model_ddp, x, t, model_kwargs
            )
            loss = loss_dict["loss"].mean()
            if self.config["trainer"]["wandb_log"]:
                wandb.log({"loss": loss})
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            update_ema(self.ema, self.model_ddp.module)
            running_loss += loss.item()

            if step % 200 == 0:
                with torch.no_grad():
                    model_kwargs = dict(a=action[:2, :])
                    z = torch.randn(
                        2,
                        4,
                        self.image_size // 8,
                        self.image_size // 8,
                        device=self.device,
                    )
                    samples = self.diffusion_s.p_sample_loop(
                        self.ema,
                        z.shape,
                        z,
                        clip_denoised=False,
                        progress=True,
                        model_kwargs=model_kwargs,
                    )
                    # samples, _ = samples.chunk(2, dim=0)
                    samples = self.vae.decode(samples / 0.18215).sample
                    save_image(
                        samples,
                        f"results/diffusion_dit/gen/epoch_{step}.png",
                        nrow=4,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                    save_image(
                        next_img[:2, :, :],
                        f"results/diffusion_dit/real/epoch_{step}.png",
                        nrow=4,
                        normalize=True,
                        value_range=(-1, 1),
                    )

        return running_loss

    def _validate_one_epoch(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, actions, timesteps in tqdm(self.val_loader, desc="Validation"):
                images, actions, timesteps = (
                    images.to(self.device),
                    actions.to(self.device),
                    timesteps.to(self.device),
                )
                outputs = self.model(images, timesteps, actions)
                loss = self.criterion(outputs, images)
                running_loss += loss.item() * images.size(0)
        return running_loss / len(self.val_loader.dataset)

    def _save_checkpoint(self):
        torch.save(self.model_ddp.state_dict(), "best_model.pth")


if __name__ == "__main__":
    trainer = DiTTrainer(
        config_path="/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/config/dit.yaml"
    )
    # wandb.init(
    #     # project=trainer.config["wandb"]["project"],
    #     # entity=trainer.config["wandb"]["entity"],
    # )
    trainer.train()
    # visualize_latent_space(trainer.data_loader, trainer.vae, trainer.device)
