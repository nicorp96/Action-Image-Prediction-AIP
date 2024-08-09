from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import os
from collections import OrderedDict
import cv2
import einops
import matplotlib.pyplot as plt
from .utils_ds import DataProcessor


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(
        input_image,
        (W, H),
        interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA,
    )
    return img


def canny(
    image,
    use_cuda=False,
    t_lower=2,
    t_upper=10,
    aperture_size=5,
    image_resolution=64,
):
    image = np.transpose(image, (1, 2, 0))
    img = HWC3(np.uint8(image))
    edge_map2 = cv2.Canny(img, t_lower, t_upper)
    edge_map = HWC3(edge_map2)

    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # plt.subplot(1, 2, 2)
    # plt.title("Canny Edge Detection")
    # plt.imshow(edge_map2)

    # plt.show()

    edge_map_torch = torch.from_numpy(edge_map.copy()).float() / 255.0
    edge_map_torch = einops.rearrange(edge_map_torch, "h w c -> c h w")
    return edge_map_torch


class RobotDatasetSeqScene(Dataset):
    def __init__(self, data_path, transform=None, img_size=120, seq_l=45):
        data_dir = os.path.expanduser(data_path)
        self._file_names = OrderedDict()
        self.data = []
        for file_name in next(os.walk(data_dir))[-1]:
            file_name = os.path.join(data_dir, file_name)
            print(file_name)
            self.data.extend(
                np.load(
                    file_name,
                    allow_pickle=True,
                )
            )
        self.transform = transform
        self.max = None
        self.min = None
        self.img_size = img_size
        self.sequence_l = seq_l
        self.__pre_process_data__()
        self._compute_max_min_actions()

    @staticmethod
    def extract_first_last_random_frames_tensor(frames, actions, k):
        l, w, h, c = frames.shape

        if k > l:
            raise ValueError("k cannot be greater than the sequence length (l).")

        if k < 2:
            raise ValueError(
                "k should be at least 2 to include the first and last frame."
            )

        # Always include the first and last frame
        selected_indices = [0, l - 1]

        # Randomly select the remaining (k - 2) unique indices from the range [1, l-1)
        remaining_indices = torch.randperm(l - 2)[: k - 2] + 1

        # Combine the indices
        selected_indices.extend(remaining_indices.tolist())

        selected_indices = sorted(selected_indices)
        # Index the tensor with the selected indices to get the random frames
        random_frames = frames[selected_indices, :, :, :]
        random_actions = actions[selected_indices, :]

        return random_frames, random_actions

    def _compute_max_min_actions(self):
        action_list = []
        for d in self.data:
            for l in range(len(d["action"])):
                action_list.append(d["action"][:3])
        action_np = np.array(action_list)
        self.max = np.max(action_np)
        self.min = np.min(action_np)

    def __len__(self):
        return len(self.data)

    def __pre_process_data__(self):
        indexes = []
        for idx in range(len(self.data)):
            s, w, h, c = self.data[idx]["image_after_action"].shape
            if s < 11:
                indexes.append(idx)
        self.data = np.delete(self.data, indexes)

    def __getitem__(self, idx):
        data_obs_i = self.data[idx]
        s, w, h, c = data_obs_i["image_after_action"].shape
        data_trans = torch.zeros((s, c, self.img_size, self.img_size))
        if self.transform:
            initial_img = self.transform(data_obs_i["image_current"])
            data_trans = torch.stack(
                [self.transform(img) for img in data_obs_i["image_after_action"]]
            )

        data_trans, action_torch = self.extract_first_last_random_frames_tensor(
            data_trans,
            torch.from_numpy(np.array(data_obs_i["action"])),
            self.sequence_l,
        )

        action_nm = self.normalize_action_torch(
            action_torch,
            pos_range=(self.min, self.max),
            method="min-max",
        ).squeeze()

        return (
            initial_img,
            data_trans,
            action_nm,
        )

    @staticmethod
    def normalize_action_torch(
        actions,
        pos_range=None,
        method="min-max",
    ):
        for l in range(actions.size()[0]):
            positions = actions[l, :3]
            euler_angles = actions[l, 3:]

            if method == "min-max":
                # Normalize positions using min-max scaling
                if pos_range is None:
                    pos_min, pos_max = positions.min(dim=0)[0], positions.max(dim=0)[0]
                else:
                    pos_min, pos_max = pos_range

                actions[l, :3] = (positions - pos_min) / (pos_max - pos_min + 1e-6)
                actions[l, :3] = 2 * positions - 1  # Normalize to range [-1, 1]

            elif method == "z-score":
                # Standardize positions
                pos_mean, pos_std = positions.mean(dim=0), positions.std(dim=0)
                positions = (positions - pos_mean) / pos_std

                # Standardize Euler angles
                euler_mean, euler_std = euler_angles.mean(dim=0), euler_angles.std(
                    dim=0
                )
                euler_angles = (euler_angles - euler_mean) / euler_std
            else:
                raise ValueError("Normalization method not recognized.")

            # actions_normalized = torch.cat([positions, euler_angles], dim=1)
            return actions


class RobotDatasetSeqSceneCanny(RobotDatasetSeqScene):
    def __init__(self, data_path, transform=None, img_size=120, seq_l=45):
        super().__init__(data_path, transform, img_size, seq_l)
        self.data_proc = DataProcessor()

    def __getitem__(self, idx):
        data_obs_i = self.data[idx]
        s, w, h, c = data_obs_i["image_after_action"].shape
        data_trans = torch.zeros((s, c, self.img_size, self.img_size))
        if self.transform:
            initial_img = self.transform(data_obs_i["image_current"])
            data_trans = torch.stack(
                [self.transform(img) for img in data_obs_i["image_after_action"]]
            )

        data_trans, action_torch = self.extract_first_last_random_frames_tensor(
            data_trans,
            torch.from_numpy(np.array(data_obs_i["action"])),
            self.sequence_l,
        )

        canny_v = torch.stack(
            [
                self.data_proc.calculate_normals(img.detach().cpu().numpy())
                for img in data_trans
            ]
        )
        action_nm = self.normalize_action_torch(
            action_torch,
            pos_range=(self.min, self.max),
            method="min-max",
        ).squeeze()

        return (initial_img, data_trans, action_nm, canny_v)


def collate_fn(batch):
    current_images, next_images, actions = zip(*batch)

    # Find the maximum sequence length in this batch
    max_seq_len = max([seq.shape[0] for seq in next_images])

    # Pad sequences
    padded_next_images = []
    for seq in next_images:
        padding = (0, 0, 0, 0, 0, 0, 0, max_seq_len - seq.shape[0])
        padded_seq = F.pad(seq, padding, "constant", 0)  # Pad with 0s
        padded_next_images.append(padded_seq)

    padded_next_images = torch.stack(padded_next_images)
    current_images = torch.stack(current_images)
    actions = torch.stack(actions)

    return current_images, padded_next_images, actions


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((120, 120)),  # Resize to (H, W)
            transforms.CenterCrop(120),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    rd_ds = RobotDatasetSeqSceneCanny(
        data_path="/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/data/dynamic_scene",
        transform=transform,
    )

    data_loader = DataLoader(rd_ds, batch_size=64, shuffle=True)

    for step, (current_image, next_image, action, canny) in enumerate(data_loader):
        image_path = os.path.join(
            "/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/w/",
            f"img_{step}.png",
        )
        vutils.save_image(
            canny.view(-1, canny.size(-3), next_image.size(-2), next_image.size(-1)),
            image_path,
            normalize=True,
        )
