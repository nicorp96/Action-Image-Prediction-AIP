from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import os


class RobotDatasetSeq(Dataset):
    def __init__(self, data_path, transform=None, img_size=64):
        self.data = np.load(
            data_path,
            allow_pickle=True,
        )
        self.transform = transform
        self.max = None
        self.min = None
        self.img_size = img_size
        self._compute_max_min_actions()

    def _compute_max_min_actions(self):
        action_list = []
        for d in self.data:
            ac_i = d["action"][:3]
            action_list.append(ac_i)
        action_np = np.array(action_list)
        self.max = np.max(action_np)
        self.min = np.min(action_np)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_obs_i = self.data[idx]
        s, w, h, c = data_obs_i["image_after_action"].shape
        print(s)
        data_trans = torch.zeros((s, c, self.img_size, self.img_size))
        if self.transform:
            initial_img = self.transform(data_obs_i["image_current"])
            for idx in range(s):
                data_i = data_obs_i["image_after_action"][idx, :, :, :]
                data_trans[idx, :, :, :] = self.transform(data_i)
        # Normalize action

        action_torch = torch.from_numpy(data_obs_i["action"][np.newaxis, :])
        action_nm = self.normalize_action_torch(
            action_torch,
            pos_range=(self.min, self.max),
            method="min-max",
        ).squeeze()
        return (
            initial_img,
            data_trans,
            action_nm,
        )  # data_obs_i["action"]

    @staticmethod
    def normalize_action_torch(
        actions,
        pos_range=None,
        method="min-max",
    ):
        positions = actions[:, :3]
        euler_angles = actions[:, 3:]

        if method == "min-max":
            # Normalize positions using min-max scaling
            if pos_range is None:
                pos_min, pos_max = positions.min(dim=0)[0], positions.max(dim=0)[0]
            else:
                pos_min, pos_max = pos_range

            positions = (positions - pos_min) / (pos_max - pos_min + 1e-6)
            positions = 2 * positions - 1  # Normalize to range [-1, 1]

        elif method == "z-score":
            # Standardize positions
            pos_mean, pos_std = positions.mean(dim=0), positions.std(dim=0)
            positions = (positions - pos_mean) / pos_std

            # Standardize Euler angles
            euler_mean, euler_std = euler_angles.mean(dim=0), euler_angles.std(dim=0)
            euler_angles = (euler_angles - euler_mean) / euler_std
        else:
            raise ValueError("Normalization method not recognized.")

        actions_normalized = torch.cat([positions, euler_angles], dim=1)
        return actions_normalized


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    rd_ds = RobotDatasetSeq(
        data_path="/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/data/ur_ds_seq.npy",
        transform=transform,
    )
    data_loader = DataLoader(rd_ds, batch_size=64, shuffle=True)
    for step, (current_image, next_image, action) in enumerate(data_loader):
        image_path = os.path.join(
            "/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/w",
            f"/img_{step}.png",
        )
        vutils.save_image(
            next_image,
            image_path,
            normalize=True,
        )
