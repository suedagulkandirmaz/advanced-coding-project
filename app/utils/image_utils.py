# app/utils/image_utils.py 
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Opens the image, converts it to (224,224) size.
def bring_visual(image_path, resize_dimension=(224, 224)):
# Checks the file path.
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
# Transforms the image.
    transform = transforms.Compose([
        transforms.Resize(resize_dimension),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
# Opens the image in RGB format.
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  

# Convert to image.
def show_image(image_tensor, title=None):
#Images a tensor as an image
    image = image_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Unnormalize.
    image = image.clip(0, 1) # Limit values ​​to 0–1.

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def predict_image(model, image_tensor, class_names, device):
# Predicts which class the image belongs to and uses models.

    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = class_names[predicted_idx.item()]
    return predicted_label
