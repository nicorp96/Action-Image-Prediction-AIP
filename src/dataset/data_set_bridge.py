import os
import json
import torch
import torch.utils.data as data
import cv2  # or use imageio for video frame extraction
import numpy as np
import torchvision.transforms as transforms
import torchvision.utils as vutils
from .utils_ds import DataProcessor
import random

class BridgeDataset(data.Dataset):

    def __init__(self, json_dir, transform=None, sequence_length=10):
        """
        Args:
            json_dir (str): Directory with all the JSON files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            frame_extraction_rate (int): Rate at which to extract frames from video.
        """
        self.json_dir = json_dir
        self.transform = transform
        self.file = []
        self.base =os.path.join(os.getcwd(), "ds_bridge/training")
        self.data = self._load_data()
        self.sequence_length = sequence_length

    def _load_data(self):
        data = []
        for file_name in os.listdir(self.json_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.json_dir, file_name)
                self.file.append(file_path)
                with open(file_path, "r") as f:
                    sample = json.load(f)
                    if os.path.exists(os.path.join(self.base, sample["videos"][0]["video_path"])):
                        data.append(sample)
        return data

    def _extract_video_frames(
        self,
        video_path,
    ):
        # Open the video file
        cap = cv2.VideoCapture(os.path.join(self.base,video_path))

        if not cap.isOpened():
            raise ValueError(f"Error: Cannot open video file at {os.path.join(self.base,video_path)}")
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames

    def __len__(self):
        return len(self.data)

    def _get_sequence_action(self, data, start_index):
        dim = data[0].shape[0]
        sequence = np.zeros((self.sequence_length, int(dim)))
        sequence[0, :] = data[0]
        sequence[(self.sequence_length - 1), :] = data[-1]
        sequence[1 : (self.sequence_length - 1), :] = data[start_index]
        return sequence

    def _get_sequence_video(self, data, start_index):
        h, w, c = data[0].shape
        sequence = np.zeros((self.sequence_length, h, w, c))
        sequence[0, :, :, :] = data[0]
        sequence[(self.sequence_length - 1), :, :, :] = data[-1]
        sequence[1 : (self.sequence_length - 1), :, :, :] = data[start_index]
        return sequence

    def __getitem__(self, idx):
        sample = self.data[idx]
        #text = sample["texts"][0]  # Assuming one text per sample
        video_path = sample["videos"][0]["video_path"]
        video_frames = self._extract_video_frames(video_path)
        video_length = len(video_frames)
        actions = np.array(sample["action"])
        # states = np.array(sample["state"])
        action_length = len(actions)

        # Ensure the sequence length is valid
        max_start_index = min(video_length, action_length) #- self.sequence_length
        if max_start_index < 0:
            raise ValueError(
                f"Sequence length {self.sequence_length} is too long for sample {self.file[idx]}"
            )
            
        random_indices = sorted(
            random.sample(range(1, (max_start_index - 1)), (self.sequence_length - 2))
        )

        # Get sequences
        video_sequence = self._get_sequence_video(video_frames, random_indices)
        # state_sequence = self._get_sequence(states, start_index)
        actions_sequence = self._get_sequence_action(actions, random_indices)

        # Apply transforms if any
        if self.transform:
            video_sequence = [self.transform(video_i) for video_i in video_sequence]

        video_sequence = torch.stack(video_sequence)
        # state_sequence = torch.tensor(state_sequence, dtype=torch.float32)
        actions_sequence = torch.tensor(actions_sequence, dtype=torch.float32)
        return video_sequence, actions_sequence


class BridgeDatasetMC(BridgeDataset):

    def __init__(self, json_dir, transform=None, sequence_length=10):
        super().__init__(json_dir, transform, sequence_length)
        self.data_proc = DataProcessor()
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        #text = sample["texts"][0]  # Assuming one text per sample
        video_path = sample["videos"][0]["video_path"]
        video_frames = self._extract_video_frames(video_path)
        video_length = len(video_frames)
        actions = np.array(sample["action"])
        # states = np.array(sample["state"])
        action_length = len(actions)

        max_start_index = min(video_length, action_length) #- self.sequence_length
        if max_start_index < 0:
            raise ValueError(
                f"Sequence length {self.sequence_length} is too long for sample {self.file[idx]}"
            )
            
        random_indices = sorted(
            random.sample(range(1, (max_start_index - 1)), (self.sequence_length - 2))
        )

        # Get sequences
        video_sequence = self._get_sequence_video(video_frames, random_indices)
        # state_sequence = self._get_sequence(states, start_index)
        actions_sequence = self._get_sequence_action(actions, random_indices)

        # Apply transforms if any
        if self.transform:
            video_sequence = [self.transform(video_i) for video_i in video_sequence]
        
        normals_v = torch.stack(
            [
                self.data_proc.calculate_normals(img.detach().cpu().numpy())
                for img in video_sequence
            ]
        )
        video_sequence = torch.stack(video_sequence)
        # state_sequence = torch.tensor(state_sequence, dtype=torch.float32)
        actions_sequence = torch.tensor(actions_sequence, dtype=torch.float32)
        return video_sequence, actions_sequence, normals_v


if __name__ == "__main__":
    custom_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert video frames to tensor
            transforms.Resize((128, 128)),  # Resize to (H, W)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # ),  # Normalization
        ]
    )
    # Usage
    dataset = BridgeDatasetMC(
        json_dir="/home/snoopy/Nicolas/Action-Image-Prediction-AIP/ds_bridge/training/annotation",
        sequence_length=10,
        transform=custom_transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        drop_last=True,
    )

    # Iterate over data
    for step, (first_frame, video_sequence, actions_sequence, normals_v) in enumerate(dataloader):
        print(video_sequence.shape)  # Should be [batch_size, sequence_length, H, W, C]
        # print(
        #     state_sequences.shape
        # )  # Should be [batch_size, sequence_length, num_features]
        image_path = os.path.join(
            "/home/snoopy/Nicolas/Action-Image-Prediction-AIP/w/",
            f"img_{step}.png",
        )
        vutils.save_image(
            video_sequence.view(
                -1,
                video_sequence.size(-3),
                video_sequence.size(-2),
                video_sequence.size(-1),
            ),
            image_path,
            normalize=True,
            nrow=10,
        )
        print(actions_sequence.size())
