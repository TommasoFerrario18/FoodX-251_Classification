import os
import torch

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from ImageDatastore import ImageDatastore
from torchvision import models, transforms


# To unfreeze all layers if needed
def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True


def check_frozen_status(model):
    frozen_layers = []
    trainable_layers = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
        else:
            frozen_layers.append(name)

    print(f"Number of frozen layers: {len(frozen_layers)}")
    print(f"Number of trainable layers: {len(trainable_layers)}")

    print("\nTrainable layers:")
    for layer in trainable_layers:
        print(layer)


def freeze_until_layer(model, target_layer="features.16"):
    # Flag to track if we've reached the target layer
    reached_target = False

    # Print and freeze layers
    for name, param in model.named_parameters():
        if target_layer in name:
            reached_target = True
            param.requires_grad = False
            continue

        if not reached_target:
            param.requires_grad = False
        else:
            print(f"Keeping layer trainable: {name}")


def count_parameters_per_layer(model):
    trainable_params_per_layer = {}
    frozen_params_per_layer = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params_per_layer[name] = param.numel()
        else:
            frozen_params_per_layer[name] = param.numel()

    print("Trainable parameters per layer:")
    for name, count in trainable_params_per_layer.items():
        print(f"{name}: {count:,} parameters")

    print("\nFrozen parameters per layer:")
    for name, count in frozen_params_per_layer.items():
        print(f"{name}: {count:,} parameters")


def count_parameters(model):
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count non-trainable parameters
    non_trainable_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # Total parameters
    total_params = trainable_params + non_trainable_params

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {total_params:,}")

    return trainable_params, non_trainable_params, total_params


def create_model(model, num_classes):
    # Replace the last layer with a new layer with num_classes output units
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    # Freeze layers
    print("\nFreezing layers:")
    freeze_until_layer(model, "features.15")

    # Check status after freezing
    check_frozen_status(model)
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

    datasets = ImageDatastore("train_retrieval", transform=transform)
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

    count_parameters_per_layer(model)
    count_parameters(model)

    # Load the data
    # dataloaders, image_datasets = load_data(batch_size=512)

    # # Train the model
    # model = train_model(model, dataloaders, num_epochs=5, image_datasets=image_datasets)

    # # Save the model
    # if not os.path.exists("../Model"):
    #     os.makedirs("../Model")

    # torch.save(model.state_dict(), "../Model/model.pth")
    # torch.save(model, "../Model/model_full.pth")
