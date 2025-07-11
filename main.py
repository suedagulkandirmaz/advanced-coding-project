import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from app.models.cnn_model import DiseaseCNN
from app.core.predictor import train_model
import os

def main():
    data_dir = 'data'
   # Class numbers.
    num_classes = 13
    batch_size = 32
    epoch_number = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Trained model save path.
    model_path = 'trained_model/plant_disease_detection.pt'

    transform = transforms.Compose([
        transforms.Resize((224, 224)), # All images are resized to 224x224.
        transforms.ToTensor(), # Converts images to tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    validation_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = DiseaseCNN(total_classes=num_classes)

    train_model (model, train_loader, validation_loader, epoch_number, device, model_path)

if __name__=="__main__":
    main()