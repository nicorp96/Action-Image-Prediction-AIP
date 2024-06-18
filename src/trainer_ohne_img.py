import torch
import torch.nn as nn
from models.generator import Generator
from models.discriminator import Discriminator
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.data_set import RobotDataset
import os
import torchvision.utils as vutils

import albumentations as A
from albumentations.pytorch import ToTensorV2


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Trainer:
    def __init__(self, config_dir) -> None:
        self.config_dir = config_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = 100
        self.generator = Generator().to(self.device)
        self.generator.apply(weights_init)
        self.discriminator = Discriminator().to(self.device)
        self.discriminator.apply(weights_init)
        self.optimizer_gen = optim.Adam(
            self.generator.parameters(), lr=0.0001, betas=(0.5, 0.9)
        )
        self.optimizer_dis = optim.Adam(
            self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9)
        )
        self.fixed_noise = torch.randn(64, self.noise_dim, 1, 1, device=self.device)
        self.criterion = nn.BCELoss()
        self.num_epochs = 300
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
        self.data_path = "/home/nrodriguez/Documents/research-image-pred/Robot-Movement-Prediction/data/panda_ds.npy"
        self.save_dir = "/home/nrodriguez/Documents/research-image-pred/Robot-Movement-Prediction/results"
        dataset = RobotDataset(data_path=self.data_path, transform=transform)
        self.data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

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

                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.discriminator.zero_grad()
                # Format batch
                b_size = next_state.size(0)
                real_labels = torch.ones(current_state.size(0), device=self.device)
                # Forward pass real batch through D
                output = self.discriminator(next_state).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, real_labels)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.noise_dim, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                fake_labels = torch.zeros(next_state.size(0), device=self.device)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, fake_labels)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizer_dis.step()
                #### Update Discriminator ####
                self.optimizer_dis.zero_grad()

                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator.zero_grad()
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, real_labels)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizer_gen.step()
                if step % 50 == 0:
                    print(
                        "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                        % (
                            epoch,
                            self.num_epochs,
                            step,
                            len(self.data_loader),
                            errD.item(),
                            errG.item(),
                            D_x,
                            D_G_z1,
                            D_G_z2,
                        )
                    )
            # Save generated images for visualization
            if not os.path.exists("results"):
                os.makedirs("results")
            if epoch % 25 == 0:
                with torch.no_grad():
                    fake = self.generator(self.fixed_noise)  # .detach().cpu()

                self.save_images(epoch, step, fake)
            # cv2.save_image(
            #     fake_images, f"results/fake_samples_epoch_{epoch}.png", normalize=True
            # )

    def save_images(self, epoch, step, fake_images):
        image_path = os.path.join(self.save_dir, f"epoch{epoch}_step{step}.png")
        vutils.save_image(fake_images, image_path, normalize=True)


if __name__ == "__main__":
    trainer = Trainer(
        "/home/nrodriguez/Documents/research-image-pred/Robot-Movement-Prediction/config"
    )
    trainer.train()
