import torch
import numpy as np

class FeatureExtractor:
    def __init__(self, model, layer, input_size, result_file = '../Features/default.csv'):
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.model = self.prepare_model(model, layer)
       self.result_file = result_file
       
    def prepare_model(self, model, layer):
        model_layers = model.named_children()
        new_model_layers = []
        for idx, (name, module) in enumerate(model_layers):
                
            if name == layer:
                break
            new_model_layers.append(module)
        
        new_model = torch.nn.Sequential(*new_model_layers)
        return new_model.to(self.device)
    
    def compute_features(self, imageDataloder):
        self.model.eval()
        
        features, labels = [], []
        with torch.no_grad():
            for batch, (x, y) in enumerate(imageDataloder):
                print(f'Batch {batch}')
                x = x.to(self.device)
                outputs = self.model(x)
                features.append(outputs.cpu().numpy())
                labels.append(y.cpu().numpy())


        final_feat = np.vstack([x.reshape(1, -1) for feat_group in features for x in feat_group])
        final_lab = [label for _, lab_group in zip(features, labels) for label in lab_group]
        return final_feat, final_lab