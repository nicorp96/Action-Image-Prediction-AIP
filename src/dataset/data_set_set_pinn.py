from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import os
import random


class DatasetObjectSeqTrj(Dataset):
    def __init__(self, data_path, transform=None, img_size=64, seq_l=16):
        self.data = np.load(
            data_path,
            allow_pickle=True,
        )
        self.transform = transform
        self.seq_data = []
        self.img_size = img_size
        self.sequence_l = seq_l
        self.__pre_process_data__()

    def __len__(self):
        return len(self.seq_data)

    def __pre_process_data__(self):
        indexes = []
        data_train = self.data[:59]
        for i in range(70):
            indexes = sorted(random.sample(range(59), 4))
            extracted_data = [data_train[i] for i in indexes]
            self.seq_data.append(extracted_data)
        print(len(self.seq_data))

    def __getitem__(self, idx):
        data_obs_i = self.seq_data[idx]
        frames_t = torch.stack(
            [self.transform(data_i["image_primary"]) for data_i in data_obs_i]
        )

        positions = torch.stack(
            [torch.from_numpy(data_i["position"]) for data_i in data_obs_i]
        )
        orientations = torch.stack(
            [torch.from_numpy(data_i["orientation"]) for data_i in data_obs_i]
        )
        velocity = torch.stack(
            [torch.from_numpy(data_i["orientation"]) for data_i in data_obs_i]
        )
        time = torch.stack([torch.tensor(data_i["time"]) for data_i in data_obs_i])

        return (
            frames_t,
            positions,
            orientations,
            velocity,
            time,
        )


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((120, 120)),  # Resize to (H, W)
            # transforms.CenterCrop(64),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    rd_ds = DatasetObjectSeqTrj(
        data_path="/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/data/object_mov.npy",
        transform=transform,
    )

    data_loader = DataLoader(rd_ds, batch_size=10, shuffle=True)

    for step, (
        frames_t,
        positions,
        orientations,
        velocity,
        time,
    ) in enumerate(data_loader):

        image_path = os.path.join(
            "/home/nrodriguez/Documents/research-image-pred/Action-Image-Prediction-AIP/w/",
            f"img_{step}.png",
        )
        vutils.save_image(
            frames_t.view(-1, frames_t.size(-3), frames_t.size(-2), frames_t.size(-1)),
            image_path,
            normalize=True,
            nrow=4,
        )
        print(f"time {time.detach().numpy()}\n")
        print(f"velocity {velocity.detach().numpy()}\n")
        print(f"positions {positions.detach().numpy()}\n")
