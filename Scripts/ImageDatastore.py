import os
import torch
import pandas as pd

from torchvision.io import decode_image
from torch.utils.data import Dataset


class ImageDatastore(Dataset):

    dataset_path = {
        "train": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/train_small.csv",
        },
        "train_aug": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/train_augmented_3.csv",
        },
        "train_unlabeled": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/train_unlabeled.csv",
        },
        "train_unlabeled_0": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/train_unlabeled_0.csv",
        },
        "train_unlabeled_1": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/train_unlabeled_1.csv",
        },
        "train_unlabeled_2": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/train_unlabeled_2.csv",
        },
        "train_unlabeled_3": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/train_unlabeled_3.csv",
        },
        "train_unlabeled_4": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/train_unlabeled_4.csv",
        },
        "train_unlabeled_5": {
            "images_directory": "../Dataset/train_set",
            "labels_file": "../Dataset/train_unlabeled_5.csv",
        },
        "val_set": {
            "images_directory": "../Dataset/val_set",
            "labels_file": "../Dataset/val_info.csv",
        },
        "val_degradate": {
            "images_directory": "../Dataset/val_set_degraded",
            "labels_file": "../Dataset/val_info.csv",
        },
    }

    def __init__(self, dataset_type, transform=None, target_transform=None):
        self.images_directory = self.dataset_path[dataset_type]["images_directory"]
        self.labels = pd.read_csv(
            self.dataset_path[dataset_type]["labels_file"], header=None
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_directory, self.labels.iloc[idx, 0])
        image = decode_image(image_path)
        image = image.type(torch.float32) / 255.0
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
