import os
import torch
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn


class FoodDatastore(Dataset):

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
            "images_directory": "../Dataset/train_augmented",
            "labels_file": "../Dataset/train_augmented.csv",
        },
    }

    def __init__(self, dataset_type, transform=None, target_transform=None):
        self.images_directory = self.dataset_path[dataset_type]["images_directory"]
        self.labels = pd.read_csv(self.dataset_path[dataset_type]["labels_file"])
        self.transform = transform
        self.target_transform = target_transform

        # Object Proposal Model
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights="FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1"
        )
        self.model.eval()
        self.rpn = self.model.rpn
        self.model_transform = self.model.transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_directory, self.labels.iloc[idx, 0])

        image = Image.open(image_path).convert("RGB")
        img_tensor = F.to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            transformed_image_list = self.model_transform(img_tensor)[0]
            features = self.model.backbone(transformed_image_list.tensors)

            proposals, _ = self.rpn(transformed_image_list, features)

            attention_mask = torch.zeros_like(img_tensor)
            proposals = proposals[0]

            # Use top-k proposals
            k = min(5, len(proposals))
            for box in proposals[:k]:
                x1, y1, x2, y2 = box.int()

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(attention_mask.shape[3], x2)
                y2 = min(attention_mask.shape[2], y2)
                attention_mask[:, :, y1:y2, x1:x2] += 1.0

            max_val = attention_mask.max()
            if max_val > 0:
                attention_mask = attention_mask / max_val

            # Apply attention to original image
            attended_image = img_tensor * attention_mask
        
        attended_image = attended_image.squeeze(0)
        
        if self.transform:
            attended_image = self.transform(attended_image)

        return attended_image, self.labels.iloc[idx, 1]
