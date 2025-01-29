from ImageDatastore import ImageDatastore
from Utils import create_or_clear_directory, transform
from NeuralFeatureExtractor import MobileNetFeatureExtractor
from torch.utils.data import DataLoader
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
import numpy as np
import os
import shutil
from sklearn.preprocessing import StandardScaler


train_data = ImageDatastore('train', transform=transform(256))

batch_size = 512
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

extractor = MobileNetFeatureExtractor()
features, labels = extractor.compute_features(train_dataloader)

path_outliers = '../Results/outliers2'
create_or_clear_directory(path_outliers)

valid_image_idx = list(range(len(features)))

range_labels = len(features) // 20
outliers = []

scaler = StandardScaler()

for i in range(range_labels):
    
    current_features = features[i*20:(i+1)*20, :]

    features_scaled = scaler.fit_transform(current_features)

    distanze = pdist(features_scaled, 'euclidean')
    
    epsilon = np.median(distanze)
    min_pts = 2
    
    dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
    labels = dbscan.fit_predict(features_scaled)
    
    outlier_idx = np.where(labels == -1)[0]

    if outlier_idx.size == 0:
        outlier_idx = np.array([0])
    else:
        for idx in outlier_idx:
            img_idx = i * 20 + idx
            img_name = train_data.labels.iloc[img_idx][0]
            img_path = os.path.join(train_data.images_directory, img_name)

            valid_image_idx.remove(img_idx)
            
            file_name, file_ext = os.path.splitext(img_name)
            outlier_file_name = os.path.join(path_outliers, f"{i}_class_{file_name}{file_ext}")
            
            shutil.copy2(img_path, outlier_file_name)
    outliers.extend(outlier_idx + i * 20)