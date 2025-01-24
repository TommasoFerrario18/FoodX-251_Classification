from ImageDatastore import ImageDatastore
from Utils import create_or_clear_directory
from ExtractFeatures import ExtractFeatures
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


train_data = ImageDatastore('train', transform=ToTensor())

batch_size = 512
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

create_or_clear_directory('../Results/outliers')

extractor = ExtractFeatures()
features, labels = extractor.extract_featuresAlexNet(train_dataloader)

print(f"N elements {len(features)}, N feature {len(features[0][2])}")
