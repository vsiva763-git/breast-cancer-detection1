import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

def generate_gradcam(model, image_path, device, target_class=None):
    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    original_img = Image.open(image_path).convert('RGB')
    original_np = np.array(original_img.resize((224, 224))) / 255.0
    input_tensor = transform(original_img).unsqueeze(0).to(device)

    # Storage for gradients and activations
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    # Hook into last conv layer of EfficientNet
    target_layer = model.features[-1]
    hook = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    output = model(input_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()

    hook.remove()

    # Compute Grad-CAM
    grad = gradients[0].cpu().detach().numpy()[0]
    act = activations[0].cpu().detach().numpy()[0]
    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    # Overlay on original image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * original_np + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    # Save heatmap
    heatmap_path = image_path.replace('.png', '_heatmap.png').replace('.jpg', '_heatmap.jpg')
    Image.fromarray((overlay * 255).astype(np.uint8)).save(heatmap_path)

    return heatmap_path, target_class, output.softmax(dim=1)[0].detach().cpu().numpy()
