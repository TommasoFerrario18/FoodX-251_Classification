import cv2
import torch

from torchvision.models import alexnet
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

class ExtractFeatures:
    def __init__(self):
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        

    def extract_featuresSIFT(image):
        pass
    
    def extract_featuresSURF(image):
        pass
    
    def extract_featuresAlexNet(self, images):
        model = alexnet(pretrained=True)
        # Remove the last layer
        features_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        # Move the model to the device
        features_extractor = features_extractor.to(self.device)
        # Set the model to evaluation mode
        features_extractor.eval()
        
        features, labels = [], []
        with torch.no_grad():
            for batch, (x, y) in enumerate(images):
                # Move the data to the device
                x, y = x.to(self.device), y.to(self.device)
                
                # Extract the features
                features_batch = features_extractor(x)
                features.append(features_batch)
                labels.append(y)

        return features, labels