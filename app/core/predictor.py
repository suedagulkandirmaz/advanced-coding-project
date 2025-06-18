import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train_model(model, train_loader, validation_loader, epoch_number, device, model_path):

#Carrying the model
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

#Optimizer identification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    score = 0.0

# Training loop
    for epoch in range(epoch_number):
        model.train()

#Total loss for the epoch
        total_loss = 0.0

#Number of correct guesses
        correct = 0

#Sample processed
        total = 0

        loop = tqdm(train_loader, description=f"Epoch [{epoch+1}/{epoch_number}]", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

#Update statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        validation_acc = evaluate_training_model(model, validation_loader, device)

        print(f"Epoch {epoch+1}/{epoch_number} - Train Loss: {total_loss:.4f}, validation Acc: {validation_acc:.2f}%")

#Save best model
        if validation_acc > score:
            score = validation_acc
            is_saved(model, model_path)
            print("Model is saved with accuracy.")

def evaluate_training_model(model, data_loader, device):
    model.evalidation()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

#Calculate percentage accuracy
    return 100 * correct / total

def is_saved(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
