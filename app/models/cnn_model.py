import torch.nn as nn #bak tekrar %56 ai
import torch 

class DiseaseCNN(nn.Module):
    def __init__(orig, total_classes=13, rate=0.4):
        super(DiseaseCNN, orig).__init__()

        # feature extractor
        orig.feature_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2), 

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)        
              )

        with torch.no_grad(): 
            input = torch.zeros(1, 3, 224, 224) 
            output_features = orig.feature_layers(input) 
            num_features = output_features.view(output_features.size(0), -1).size(1) 

        # Fully connected layer classification title
        orig.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.4),
            nn.Linear(512, total_classes)
        )

    def forward(orig, x):
        
        x = orig.feature_layers(x)
        x = orig.fully_connected(x)
        print(x.shape) 
        return x