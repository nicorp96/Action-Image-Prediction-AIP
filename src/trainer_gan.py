import torch
import torch.nn as nn
from models.generator import GeneratorActor
from models.discriminator import Discriminator
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.data_set import RobotDataset
import os
import torchvision.utils as vutils
import numpy as np
import yaml


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def normalize_action(
    actions,
    pos_range=None,
    euler_range=(-np.pi, np.pi),
    method="min-max",
    angle_unit="radian",
):
    if angle_unit == "degree":
        euler_range = (-180, 180)

    positions = actions[:, :3]
    euler_angles = actions[:, 3:]

    if method == "min-max":
        # Normalize positions using min-max scaling
        if pos_range is None:
            pos_min, pos_max = positions.min(axis=0), positions.max(axis=0)
        else:
            pos_min, pos_max = pos_range

        positions = (positions - pos_min) / (pos_max - pos_min + 1e-6)
        positions = 2 * positions - 1  # Normalize to range [-1, 1]

        # Normalize Euler angles using min-max scaling
        euler_min, euler_max = euler_range
        euler_angles = (euler_angles - euler_min) / (euler_max - euler_min)
        euler_angles = 2 * euler_angles - 1  # Normalize to range [-1, 1]

    elif method == "z-score":
        # Standardize positions
        pos_mean, pos_std = positions.mean(axis=0), positions.std(axis=0)
        positions = (positions - pos_mean) / pos_std

        # Standardize Euler angles
        euler_mean, euler_std = euler_angles.mean(axis=0), euler_angles.std(axis=0)
        euler_angles = (euler_angles - euler_mean) / euler_std
    else:
        raise ValueError("Normalization method not recognized.")

    actions_normalized = np.concatenate([positions, euler_angles], axis=1)
    return actions_normalized


def generate_random_actions(range_pos=None):
    action = np.zeros((5, 7))
    for b in range(action.shape[0]):
        action[b, 0] = np.random.uniform(-0.2, 0.4)
        action[b, 1] = np.random.uniform(0, 0.2)
        action[b, 2] = np.random.uniform(0.45, 0.5)
        action[b, 3] = -0.6775629260137974
        action[b, 4] = 1.536645118148555
        action[b, 5] = -2.2832693130805426
        action[b, 6] = np.random.uniform(0, 0.08)

    action_norm = normalize_action(action, pos_range=range_pos)
    action_norm = torch.from_numpy(action_norm)
    return action, action_norm.to(dtype=torch.float32)


class Trainer:
    def __init__(self, config_dir) -> None:
        # Load configuration from YAML file
        self.__load_config_file__(config_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = self.config["noise_dim"]
        self.generator = GeneratorActor().to(self.device)
        self.generator.apply(weights_init)
        self.discriminator = Discriminator().to(self.device)
        self.discriminator.apply(weights_init)
        self.optimizer_gen = optim.Adam(
            self.generator.parameters(),
            lr=self.config["generator_lr"],
            betas=tuple(self.config["generator_betas"]),
        )
        self.optimizer_dis = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config["discriminator_lr"],
            betas=tuple(self.config["discriminator_betas"]),
        )
        self.fixed_noise = torch.randn(
            self.config["fixed_noise_samples"], self.noise_dim, 1, 1, device=self.device
        )

        if self.config["criterion"] == "BCE":
            self.criterion = nn.BCELoss()
        # Add more criteria as needed

        self.num_epochs = self.config["num_epochs"]
        self.image_size = self.config["image_size"]

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Load dataset
        self.data_path = self.config["data_path"]
        self.save_dir = self.config["save_dir"]
        self.dataset = RobotDataset(data_path=self.data_path, transform=transform)
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.config["batch_size"], shuffle=True
        )

    def __load_config_file__(self, config_dir):
        with open(config_dir, "r") as file:
            self.config = yaml.safe_load(file)

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
                # print(output.size())
                # print(real_labels.size())
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, real_labels)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.noise_dim, 1, 1, device=self.device)

                # Generate fake image batch with G
                fake = self.generator(noise, action)
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
                        "[%d/%d][%d/%d]\tLoss_D: %.6f\tLoss_G: %.6f\tD(x): %.6f\tD(G(z)): %.6f / %.6f"
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
                action, action_nm = generate_random_actions(
                    (self.dataset.min, self.dataset.max)
                )
                action_nm = action_nm.to(device=self.device)

                with torch.no_grad():
                    fake = self.generator(self.fixed_noise, action_nm)
                self.save_images_actions(epoch, step, fake, action)

    def save_images_actions(self, epoch, step, fake_images, actions):
        image_path = os.path.join(self.save_dir, f"epoch{epoch}_step{step}.png")
        action_path = os.path.join(self.save_dir, "actions")
        action_path = os.path.join(action_path, f"epoch{epoch}_step{step}.csv")
        vutils.save_image(fake_images, image_path, normalize=True)
        np.savetxt(
            action_path,
            actions,
            delimiter=",",
        )


if __name__ == "__main__":
    trainer = Trainer(
        "/home/nrodriguez/Documents/research-image-pred/Robot-Movement-Prediction/config/gan_img.yaml"
    )
    trainer.train()
