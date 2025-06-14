import torch   
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
  


def model_training(model, train_loader, validation_loader, epoch_number, device, model_saver):
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), Ir=0.001)

    highest_acc = 0
    

    for epoch in range(epoch_number):
        model.train()
        epoch_loss = 0
        total, true = 0, 0

        for x, y in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x) 
            loss = loss_fn(pred,y)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            predicted = torch.argmax(pred, dim=1)
            total += y.size(0)
            true += (predicted == y).sum().item()

        train_acc = 100* true/total

        model.eval()
        val_total, vall_correct = 0, 0
        with torch.no_grad():
            for x, y in tqdm(validation_loader, desc=f"Validation Epoch {epoch+1}"):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                predicted = torch.argmax(pred, dim=1)
                val_total += y.size(0)
                vall_correct += (predicted == y).sum().item()
        
        acc = 100 * vall_correct / val_total
        print(f"Epoch{epoch+1} | Loss: {epoch_loss:.3f} | Val Accuracy: {acc:.2f}%")

        if validation_loader > highest_acc:
            highest_acc = acc
            torch.save = (model.state_dict(), model_saver)
            print("New model saved")


