import torch
import torch.nn as nn
from models.generator import GeneratorActor, GeneratorActor2
from models.discriminator import DiscriminatorAuxAct
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.data_set import RobotDataset
import os
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from utils import unnormilize_action_torch


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
        self.generator = GeneratorActor().to(self.device)
        self.generator.apply(weights_init)
        self.discriminator = DiscriminatorAuxAct().to(self.device)
        self.discriminator.apply(weights_init)
        self.optimizer_gen = optim.Adam(
            self.generator.parameters(), lr=0.0001, betas=(0.5, 0.9)
        )
        self.optimizer_dis = optim.Adam(
            self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9)
        )
        self.fixed_noise = torch.randn(5, self.noise_dim, 1, 1, device=self.device)
        self.criterion = nn.BCELoss()
        self.criterion_action = nn.MSELoss()
        self.num_epochs = 700
        self.image_size = 64
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
        self.save_dir = "/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/results/gan_aux_cond"
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

                # (1) Update D network
                ###########################
                self.discriminator.zero_grad()
                # Format batch
                b_size = next_state.size(0)
                real_labels = torch.ones(current_state.size(0), device=self.device)
                # Forward pass real batch through D
                real_fake_pred, action_pred = self.discriminator(next_state)
                # Calculate loss on all-real batch
                errD_real = self.criterion(real_fake_pred.view(-1), real_labels)
                errD_act = self.criterion_action(action_pred, action)
                # Calculate gradients for D in backward pass
                err_loss_real = errD_real + errD_act
                err_loss_real.backward()
                D_x = real_fake_pred.mean().item()

                ## Train with all-fake batch
                noise = torch.randn(b_size, self.noise_dim, 1, 1, device=self.device)

                fake = self.generator(noise, action)
                fake_labels = torch.zeros(next_state.size(0), device=self.device)
                # Classify all fake batch with D
                output, action_pred = self.discriminator(fake.detach())
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output.view(-1), fake_labels)
                errD_fk_act = self.criterion_action(action_pred, action)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = err_loss_real + errD_fake + errD_fk_act

                self.optimizer_dis.step()
                self.optimizer_dis.zero_grad()

                # (2) Update G network

                self.generator.zero_grad()

                output, action_pred = self.discriminator(fake)
                errG = self.criterion(output.view(-1), real_labels)
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizer_gen.step()

                if step % 50 == 0:
                    print(
                        "[%d/%d][%d/%d]\tLoss_D: %.6f\tLoss_G: %.6f\tD(x): %.6f\tD(G(z)): %.6f / %.6f \t L_ac: %.6f"
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
                            errD_fk_act.item(),
                        )
                    )
            # Save generated images for visualization
            if not os.path.exists("results"):
                os.makedirs("results")
            if epoch % 25 == 0:

                with torch.no_grad():
                    fake = self.generator(self.fixed_noise, action[:5, :])
                    pred, actions_pred = self.discriminator(fake)
                # self.save_images_actions(epoch, step, fake, action)
                self.save_images_actions_real(
                    epoch,
                    step,
                    fake,
                    next_state[:5, :, :],
                    action[:5, :],
                    actions_pred,
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

    def save_images_actions_real(
        self, epoch, step, fake_images, real, actions_real, actions_pred
    ):
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
        action_un = unnormilize_action_torch(actions_pred)
        act_diff = actions_real - action_un
        np.savetxt(
            action_path,
            act_diff.detach().cpu().numpy(),
            delimiter=",",
        )


if __name__ == "__main__":
    trainer = Trainer(
        "/home/nrodriguez/Documents/research-image-pred/Robot-Movement-Prediction/config"
    )
    trainer.train()
