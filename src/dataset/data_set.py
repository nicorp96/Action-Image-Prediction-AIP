from torch.utils.data import Dataset
import numpy as np


class RobotDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = np.load(
            data_path,
            allow_pickle=True,
        )
        self.transform = transform
        self.max = None
        self.min = None
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
        if self.transform:
            initial_state = self.transform(data_obs_i["image_current"])
            next_state = self.transform(data_obs_i["image_after_action"])
        # Normalize action
        action_nm = self.normalize_action(
            data_obs_i["action"][np.newaxis, :],
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

            # Normalize Euler angles using min-max scaling
            # euler_min, euler_max = euler_range
            # euler_angles = (euler_angles - euler_min) / (euler_max - euler_min)
            # euler_angles = 2 * euler_angles - 1  # Normalize to range [-1, 1]

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

    def normlize2(action):
        # Normalize actions to [-1, 1]
        action_norm = action
        action_norm[:, :3] = (
            action_norm[:, :3] - action_norm[:, :3].min(0, keepdim=True)[0]
        ) / (
            action_norm[:, :3].max(0, keepdim=True)[0]
            - action_norm[:, :3].min(0, keepdim=True)[0]
        ) * 2 - 1
        action_norm[:, 3:] = torch.stack(
            [torch.sin(action_norm[:, 3:]), torch.cos(action_norm[:, 3:])], dim=-1
        ).view(action_norm.size(0), -1)
        return action_norm
