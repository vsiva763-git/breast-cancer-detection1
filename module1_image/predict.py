import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from module1_image.gradcam import generate_gradcam

def load_image_model(model_path, device):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = output.softmax(dim=1)[0].cpu().numpy()

    malignant_prob = float(probs[1])
    label = 'Malignant' if malignant_prob > 0.5 else 'Benign'

    # Generate Grad-CAM heatmap
    heatmap_path, _, _ = generate_gradcam(model, image_path, device, target_class=1)

    return {
        'probability': malignant_prob,
        'label': label,
        'confidence': float(max(probs)),
        'heatmap_path': heatmap_path
    }
