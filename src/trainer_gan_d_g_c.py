import torch
import torch.nn as nn
from src.models.generator import (
    GeneratorActor,
    GeneratorActor2,
    GeneratorActorUN,
    GeneratorActorImg,
)
from src.models.discriminator import DiscriminatorCA2, DiscriminatorCA
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from src.dataset.data_set import RobotDataset
import os
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import yaml
from src.trainer_base import TrainerBase


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class TrainerGANCAGD(TrainerBase):
    def __init__(self, config_dir) -> None:
        super().__init__(config_dir)

        self.__load_config_file__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = self.config["noise_dim"]
        self.generator = GeneratorActorImg().to(self.device)
        self.generator.apply(weights_init)
        self.discriminator = DiscriminatorCA().to(
            self.device
        )  # Changed to DiscriminatorCA
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
        elif self.config["criterion"] == "MSE":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError
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
        self.save_dir_metric = self.config["save_dir_metric"]
        self.save_dir_mdl = self.config["save_dir_checkpoint"]
        # Initialize arrays to store metrics
        self.epoch_metrics = {
            "epoch": [],
            "step": [],
            "loss_d": [],
            "loss_g": [],
            "D_x": [],
            "D_G_z_real": [],
            "D_G_z_fake": [],
        }

    def __load_config_file__(self):
        with open(self.config_dir, "r") as file:
            self.config = yaml.safe_load(file)

    def save_metrics(self):
        metrics_path = os.path.join(self.save_dir_metric, "training_metrics.npz")
        np.savez(metrics_path, **self.epoch_metrics)

    def save_weights(self):
        model_dir = os.path.join(self.save_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(
            self.generator.state_dict(),
            os.path.join(model_dir, f"generator_epoch.pth"),
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(model_dir, f"discriminator_epoch.pth"),
        )

    def train(self):
        for epoch in range(self.num_epochs):
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

                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                self.discriminator.zero_grad()
                b_size = next_image.size(0)
                real_labels = torch.ones(current_image.size(0), device=self.device)
                output = self.discriminator(next_image, action).view(-1)
                errD_real = self.criterion(output, real_labels)
                errD_real.backward()
                D_x = output.mean().item()
                noise = torch.randn(b_size, self.noise_dim, 1, 1, device=self.device)

                # Generate fake image batch with G
                fake = self.generator(noise, action, current_image)
                fake_labels = torch.zeros(next_image.size(0), device=self.device)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach(), action).view(-1)
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

                self.generator.zero_grad()
                output = self.discriminator(fake, action).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, real_labels)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizer_gen.step()
                self.optimizer_gen.zero_grad()
                self.epoch_metrics["epoch"].append(epoch)
                self.epoch_metrics["step"].append(step)
                self.epoch_metrics["loss_d"].append(errD.item())
                self.epoch_metrics["loss_g"].append(errG.item())
                self.epoch_metrics["D_x"].append(D_x)
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
                with torch.no_grad():
                    fake = self.generator(
                        self.fixed_noise, action[:5, :], current_image[:5, :, :]
                    )
                # self.save_images_actions(epoch, step, fake, action)
                self.save_images_actions_real(
                    epoch,
                    step,
                    fake,
                    action[:5, :],
                    next_image[:5, :, :],
                )

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

    def save_images_actions_real(self, epoch, step, fake_images, actions, real):
        def convert_to_displayable_format(image):
            image = np.transpose(
                image, (1, 2, 0)
            )  # Convert from (C, H, W) to (H, W, C)
            image = (image - image.min()) / (
                image.max() - image.min()
            )  # Normalize to range [0, 1]
            return image

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 5))
        img_fake_np = fake_images.detach().cpu().numpy()
        img_real_np = real.detach().cpu().numpy()
        image_path = os.path.join(self.save_dir, f"epoch{epoch}_step{step}.png")
        action_path = os.path.join(self.save_dir, "actions")
        action_path = os.path.join(action_path, f"epoch{epoch}_step{step}.csv")
        for i in range(img_fake_np.shape[0]):
            ax_fake = axes[0, i]
            ax_real = axes[1, i]

            fake_img = convert_to_displayable_format(img_fake_np[i])
            real_img = convert_to_displayable_format(img_real_np[i])

            ax_fake.imshow(fake_img)
            ax_fake.axis("off")
            ax_fake.set_title("Fake")

            ax_real.imshow(real_img)
            ax_real.axis("off")
            ax_real.set_title("Real")

        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()
        np.savetxt(
            action_path,
            actions.detach().cpu().numpy(),
            delimiter=",",
        )


if __name__ == "__main__":
    trainer = TrainerGANCAGD(
        "/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/config/gan_condition_gd.yaml"
    )
    trainer.train()
    trainer.save_metrics()
    trainer.save_weights()
