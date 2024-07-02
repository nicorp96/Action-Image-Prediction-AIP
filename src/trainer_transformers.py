import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.transformers import BiDirectionalTransformer
from torchvision import transforms
from dataset.data_set import RobotDataset
from dataset.data_set_seq import RobotDatasetSeq
import yaml


class Trainer:

    def __init__(self, config_dir):
        self.config_dir = config_dir
        # Load configuration from YAML file
        self.__load_config_file__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiDirectionalTransformer(
            image_feature_dim=256,
            robot_state_dim=10,
            transformer_dim=512,
            num_heads=8,
            num_layers=6,
        )
        self.num_epochs = self.config["num_epochs"]
        self.image_size = self.config["image_size"]
        self.optimizer_gen = optim.Adam(self.model.parameters(), lr=0.001)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if self.config["criterion"] == "BCE":
            self.criterion = nn.BCELoss()
        elif self.config["criterion"] == "MSE":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError

        self.model.to(self.device)
        self.data_path = self.config["data_path"]
        self.save_dir = self.config["save_dir"]
        self.dataset = RobotDataset(data_path=self.data_path, transform=transform)
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.config["batch_size"], shuffle=True
        )

    def __load_config_file__(self):
        with open(self.config_dir, "r") as file:
            self.config = yaml.safe_load(file)

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for step, (current_image, next_image, action) in enumerate(self.data_loader):
            # Move data to the device
            current_image = current_image.to(self.device)
            robot_state = robot_state.to(self.device)
            action = action.to(self.device)
            next_image = next_image.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(current_image, robot_state, next_image_seq)
            l = self.criterion(outputs, target_next_image)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(self.train_loader)
        return average_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, (
                current_image,
                robot_state,
                next_image_seq,
                target_next_image,
            ) in enumerate(self.val_loader):
                # Move data to the device
                current_image = current_image.to(self.device)
                robot_state = robot_state.to(self.device)
                next_image_seq = next_image_seq.to(self.device)
                target_next_image = target_next_image.to(self.device)

                # Forward pass
                outputs = self.model(current_image, robot_state, next_image_seq)
                loss = self.criterion(outputs, target_next_image)

                running_loss += loss.item()

        average_loss = running_loss / len(self.val_loader)
        return average_loss

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            # val_loss = self.validate()
            val_loss = 0
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )


# Example of how to use the Trainer class
if __name__ == "__main__":
    # Initialize the trainer
    trainer = Trainer(
        "/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/config/transformers.yaml"
    )
    # Train the model
    trainer.train()
