import os
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

class ImageDatastore(Dataset):
    def __init__(self, images_directory, labels_file, transform=None, target_transform=None):
        self.images_directory = images_directory
        self.labels = pd.read_csv(labels_file)
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