import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


def get_dataloaders(data_dir, batch_size=32):
    train_transforms, val_transforms = get_transforms()

    # Load full dataset with train transforms first
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

    # Class names (benign / malignant)
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Total images: {len(full_dataset)}")

    # 80% train, 20% validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply val transforms to validation set
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"Train samples: {train_size} | Val samples: {val_size}")
    return train_loader, val_loader, class_names
