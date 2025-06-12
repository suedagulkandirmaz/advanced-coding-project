# data_loader.py

import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision import datasets

#Class for loading and processing plant disease dataset
class DiseaseDataLoader :
    def __init__(orig,  root, dimension=224, batch=32):
       
#Constructructor method
         orig.root = root
         orig.dimension = dimension
         orig.batch = batch
      

#image_transform operations to normalize images
         orig.preprocess = Compose([
            Resize((orig.dimension, orig.dimension)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
        ])

def train_validation(orig):
    train_dir = os.path.join(orig.root, "train")
    validation_dir = os.path.join(orig.root, "validation")

#Creating ImageFolder dataset
    train_dataset = datasets.ImageFolder(train_dir, image_transform=orig.image_transform)
    validation_dataset = datasets.ImageFolder(validation_dir, image_transform=orig.image_transform)

#Creating batch data loaders with PyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch=orig.batch, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch=orig.batch, shuffle=False)
    
    return train_loader, validation_loader
