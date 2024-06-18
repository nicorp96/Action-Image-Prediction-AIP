import torch
import torch.nn as nn
from models.diffusion import DiffusionForward, DiffusionReverse, UNet

import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.data_set import RobotDataset


class Trainer:
    def __init__(self, config_dir) -> None:
        self.config_dir = config_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = 100
        self.model = UNet(in_channels=3, out_channels=3, cond_channels=7).to(
            device=self.device
        )
        betas = [0.1] * 10
        betas = torch.tensor(betas).to(dtype=torch.float32, device=self.device)
        self.diffusion_forward = DiffusionForward(betas, self.device)
        self.diffusion_reverse = DiffusionReverse(betas, self.model)
        self.cond_channels = 7
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.num_epochs = 100
        self.image_size = 64
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
        self.data_path = "/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/data/panda_ds.npy"

        self.dataset = RobotDataset(data_path=self.data_path, transform=transform)
        self.data_loader = DataLoader(self.dataset, batch_size=64, shuffle=True)

    def __load_config_file__(self):
        pass  #

    def train(self):
        for epoch in range(self.num_epochs):
            for step, (current_state, next_state, action) in enumerate(
                self.data_loader
            ):
                current_state, action, next_state = (
                    current_state.to(self.device),
                    action.to(
                        self.device,
                        dtype=torch.float32,
                    ),
                    next_state.to(self.device),
                )

                self.model.zero_grad()

                # Add noise to images using forward diffusion
                noisy_images, ground_truth_noise = self.diffusion_forward.add_noise(
                    current_state,
                    torch.randint(
                        0, len(self.diffusion_forward.betas), (current_state.size(0),)
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
                    current_state,
                    torch.randint(
                        0, len(self.diffusion_forward.betas), (current_state.size(0),)
                    ),
                    save_image=True,
                    image_id=step,
                    epoch=epoch,
                )

                self.diffusion_reverse.reverse_diffusion(
                    noisy_image, action, save_image=True, image_id=step, epoch=epoch
                )


if __name__ == "__main__":
    trainer = Trainer(
        "/home/nrodriguez/Documents/research-image-pred/Robot-Movement-Prediction/config"
    )
    trainer.train()
