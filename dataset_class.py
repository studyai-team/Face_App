import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2

# from .video_extraction_conversion import *
from networks import *
import torchvision.transforms as transforms

from torchvision.utils import save_image
import face_alignment


std = np.array([0.5, 0.5, 0.5])
mean = np.array([1, 1, 1])


class VidDataSet(Dataset):
    def __init__(self, size, data_path, device):
        self.size = size
        self.data_path = data_path
        self.device = device
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std),
             ]
        )
        self.files = glob.glob(data_path + "/*/*/*/*.jpg")
        self.face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,
                                                         device='cuda:0')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx % len(self.files)]
            source_image = cv2.imread(path)
            source_image = cv2.resize(source_image, (self.size, self.size))
            source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            source_landmark = generate_landmarks(source_image, self.face_aligner)

            target_image = cv2.imread(get_target_landmark_path(path))
            target_image = cv2.resize(target_image, (self.size, self.size))
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_landmark = generate_landmarks(target_image, self.face_aligner)

            source_image = source_image.transpose(2, 0, 1) / 255
            target_image = target_image.transpose(2, 0, 1) / 255
            source_landmark = source_landmark.transpose(2, 0, 1) / 255
            target_landmark = target_landmark.transpose(2, 0, 1) / 255

            source_image = torch.from_numpy(source_image)
            target_image = torch.from_numpy(target_image)
            source_landmark = torch.from_numpy(source_landmark)
            target_landmark = torch.from_numpy(target_landmark)

            return {"source_image": source_image, "target_image": target_image, "source_landmark": source_landmark, "target_landmark": target_landmark}

        except:
            return None


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
