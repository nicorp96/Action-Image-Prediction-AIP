from torch.utils.data import Dataset
import torch
import numpy as np


class RobotDataset(Dataset):
    def __init__(self, data_path, transform=None, transform_2=None):
        self.data = np.load(
            data_path,
            allow_pickle=True,
        )
        self.transform = transform
        self.transform2 = transform_2
        self.max = None
        self.min = None
        self._compute_max_min_actions()
        if transform_2 is None:
            self.transform2 = transform

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
        if self.transform:
            initial_state = self.transform2(data_obs_i["image_current"])
            next_state = self.transform(data_obs_i["image_after_action"])
        # Normalize action
        action_torch = torch.from_numpy(data_obs_i["action"][np.newaxis, :])
        action_nm = self.normalize_action_torch(
            action_torch,
            pos_range=(self.min, self.max),
            method="min-max",
        ).squeeze()
        return initial_state, next_state, action_nm  # data_obs_i["action"]

    @staticmethod
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
