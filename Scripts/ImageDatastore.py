import os
import torch
import pandas as pd

from torchvision.io import read_image
from torch.utils.data import Dataset


class ImageDatastore(Dataset):

    dataset_path = {
        "train": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/train_small.csv",
        },
        "train_retrieval": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/train_retrieval.csv",
        },
        "train_unlabeled": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/labels_experiment/train_retrieval_k25_centroid.csv",
        },
        "val_set": {
            "images_directory": "../Dataset/val_set",
            "labels_file": "../Dataset/val_info.csv",
        },
        "val_degradate": {
            "images_directory": "../Dataset/val_set_degraded",
            "labels_file": "../Dataset/val_info.csv",
        },
        "val_degradate_clear": {
            "images_directory": "../Dataset/Preprocessed",
            "labels_file": "../Dataset/val_deg_process.csv",
        },
        "train_augmented": {
            "images_directory": "../Dataset/train_augmented_2_retrieval",
            "labels_file": "../Dataset/train_augmented.csv",
        },
    }

    def __init__(self, dataset_type, transform=None, target_transform=None):
        self.images_directory = self.dataset_path[dataset_type]["images_directory"]
        self.labels = pd.read_csv(self.dataset_path[dataset_type]["labels_file"])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_directory, self.labels.iloc[idx, 0])
        image = read_image(image_path)
        image = image.type(torch.float32) / 255.0
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
