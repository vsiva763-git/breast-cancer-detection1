import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, device='cpu'):
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all early layers — only train the last few
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    # Replace the final classifier for binary classification
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes)
    )

    model = model.to(device)
    print(f"Model loaded on: {device}")
    return model
