import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

class ImageDatastore(Dataset):

    dataset_path = {
        'train': {
            'images_directory': '../Dataset/train_set',
            'labels_file': '../Dataset/train_small.csv'
        },
        'train_unlabeled': {
            'images_directory': '../Dataset/train_set',
            'labels_file': '../Dataset/train_unlabeled.csv'
        },
        'val_set': {
            'images_directory': '../Dataset/val_set',
            'labels_file': '../Dataset/val_info.csv'
        },
        'val_degradated': {
            'images_directory': '../Dataset/val_set_degraded',
            'labels_file': '../Dataset/val_info.csv'
        }
    }

    def __init__(self, dataset_type, transform=None, target_transform=None):
        self.images_directory = self.dataset_path[dataset_type]['images_directory']
        self.labels = pd.read_csv(self.dataset_path[dataset_type]['labels_file'], header=None)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_directory, self.labels.iloc[idx, 0])
        image = Image.open(image_path)
        image = image.resize((224, 224))
        label = self.labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label