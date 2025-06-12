import torch #%75
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from app.models.cnn_model import PlantDiseaseCNN
from app.core.predictor import train_model
import os

def main():
    data_dir = 'data'
    num_classes = 13
    batch_size = 32
    num_epoch = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = 'trained_model/plant_disease_detection.pt'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PlantDiseaseCNN(total_classes=num_classes)

    train_model (model, train_loader, val_loader, num_epoch, device, model_save_path)

if __name__=="__main__":
    main()