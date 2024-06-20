from models.vae_model import VAE
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.data_set import RobotDataset


def loss_function(recon_x, x, mean, log_var):
    BCE = nn.BCELoss(reduction="sum")
    x_out = BCE(recon_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return x_out + KLD


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = VAE().to(device=self.device)
        self.image_size = 64
        self.batch_size = 64
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.save_dir = "/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/data/panda_ds.npy"
        self.dataset = RobotDataset(data_path=self.save_dir, transform=transform)
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        self.criterion = nn.BCELoss(reduction="sum")
        self.optimizer_vae = Adam(self.vae.parameters(), lr=1e-3)
        self.num_epochs = 100

    def train(self):
        self.vae.train()

        for epoch in range(self.num_epochs):
            train_loss = 0
            for step, (current_image, next_image, action) in enumerate(
                self.data_loader
            ):
                current_image, action, next_image = (
                    current_image.to(self.device),
                    action.to(
                        self.device,
                        dtype=torch.float32,
                    ),
                    next_image.to(self.device),
                )
                self.optimizer_vae.zero_grad()

                recon_batch, mean, log_var = self.vae(current_image)
                recon_x = recon_batch.clamp(1e-8, 1 - 1e-8)
                x = current_image.clamp(1e-8, 1 - 1e-8)  # Cl
                loss_bce = self.criterion(recon_x, x)
                kdl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                loss = loss_bce + kdl
                loss.backward()
                train_loss += loss.item()
                self.optimizer_vae.step()
            print(
                f"Epoch: {epoch + 1}/{self.num_epochs}, Loss: {train_loss / len(self.data_loader.dataset):.6f}"
            )


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
