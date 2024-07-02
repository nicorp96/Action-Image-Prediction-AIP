import torch
import torch.nn as nn
from models.diffusion import (
    DiffusionForward,
    DiffusionReverse,
    UNet,
    UNet2,
    TransformerDiffusionModel,
)

import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.data_set import RobotDataset


class Trainer:
    def __init__(self, config_dir) -> None:
        self.config_dir = config_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = 100
        self.num_epochs = 1000
        self.image_size = 64
        self.cond_channels = 7

        # self.model = UNet2(in_channels=3, out_channels=3, cond_channels=7).to(
        #     device=self.device
        # )
        self.model = TransformerDiffusionModel(
            image_size=64, patch_size=16, num_classes=3, cond_channels=7
        ).to(device=self.device)

        betas = torch.linspace(0.0001, 0.02, steps=20).to(
            dtype=torch.float32, device=self.device
        )
        self.diffusion_forward = DiffusionForward(betas, self.device)
        self.diffusion_reverse = DiffusionReverse(betas, self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

        # Example transform
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Load dataset
        self.data_path = "/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/data/ur_ds.npy"

        self.dataset = RobotDataset(data_path=self.data_path, transform=transform)
        self.data_loader = DataLoader(self.dataset, batch_size=32, shuffle=True)

    def __load_config_file__(self):
        pass  #

    def train(self):
        for epoch in range(self.num_epochs):
            for step, (current_img, next_img, action) in enumerate(self.data_loader):
                current_img, action, next_img = (
                    current_img.to(self.device),
                    action.to(
                        self.device,
                        dtype=torch.float32,
                    ),
                    next_img.to(self.device),
                )

                self.model.zero_grad()

                # Add noise to images using forward diffusion
                noisy_images, ground_truth_noise = self.diffusion_forward.add_noise(
                    current_img,
                    torch.randint(
                        0, len(self.diffusion_forward.betas), (current_img.size(0),)
                    ),
                )

                # Predict noise using the model
                predicted_noise = self.model(noisy_images, action)

                # Calculate loss and update the model
                loss = self.criterion(predicted_noise, ground_truth_noise)
                loss.backward()
                self.optimizer.step()
                print(f"Epoch {epoch} Loss: {loss.item()}")

            if epoch % 25 == 0:
                noisy_image, _ = self.diffusion_forward.add_noise(
                    next_img,
                    torch.randint(
                        0, len(self.diffusion_forward.betas), (next_img.size(0),)
                    ),
                    save_image=True,
                    image_id=step,
                    epoch=epoch,
                )

                self.diffusion_reverse.reverse_diffusion(
                    noisy_image, action, save_image=True, image_id=2, epoch=epoch
                )


if __name__ == "__main__":
    trainer = Trainer(
        "/home/nrodriguez/Documents/research-image-pred/Robot-Movement-Prediction/config"
    )
    trainer.train()
