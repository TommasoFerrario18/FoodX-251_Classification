import os
import torch

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from ImageDatastore import ImageDatastore
from torchvision import models, transforms


def create_model(model, num_classes):
    # Replace the last layer with a new layer with num_classes output units
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    # Conta il numero di blocchi InvertedResidual congelati
    frozen_blocks = 0

    for layer in model.features:
        if isinstance(layer, models.mobilenetv3.InvertedResidual):
            frozen_blocks += 1
            for param in layer.parameters():
                param.requires_grad = False
            if frozen_blocks == 10:
                break

    return model


def load_data(batch_size):
    # Load the data
    transform = transforms.Compose(
        [
            transforms.Resize(
                (232, 232), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    datasets = ImageDatastore("train_aug", transform=transform)
    image_datasets = {}

    train, val = torch.utils.data.random_split(
        datasets, [int(0.8 * len(datasets)), len(datasets) - int(0.8 * len(datasets))]
    )
    image_datasets["train"] = train
    image_datasets["val"] = val

    dataloders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
        for x in ["train", "val"]
    }

    return dataloders, image_datasets


def train_model(model, dataloaders, num_epochs, image_datasets):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.classifier[3].parameters(), lr=0.001, momentum=0.9
    )

    for epoch in tqdm(range(num_epochs)):
        # print(f"Epoch {epoch}/{num_epochs - 1}")
        # print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        if not os.path.exists("../Model"):
            os.makedirs("../Model")
        
        torch.save(model.state_dict(), f"../Model/model_{epoch}.pth")

    return model


if __name__ == "__main__":
    # Load the pre-trained model
    model = models.mobilenet_v3_large(weights="IMAGENET1K_V2")

    # Create the model
    model = create_model(model, 251)

    # Load the data
    dataloaders, image_datasets = load_data(batch_size=512)

    # Train the model
    model = train_model(model, dataloaders, num_epochs=5, image_datasets=image_datasets)

    # Save the model
    if not os.path.exists("../Model"):
        os.makedirs("../Model")

    torch.save(model.state_dict(), "../Model/model.pth")
    torch.save(model, "../Model/model_full.pth")
