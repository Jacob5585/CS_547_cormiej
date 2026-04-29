import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms import v2

def get_approach_names():
    approches = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"]
    
    return approches

def get_approach_description(approach_name):
    match approach_name:
        case "a1":
            return ""
        case "a2":
            return ""
        case "a3":
            return ""
        case "a4":
            return ""
        case "a5":
            return ""
        case "a6":
            return ""
        case "a7":
            return ""
        case "a8":
            return ""
        case _:
            return "Not a valud approch"    

def get_data_transform(approach_name, training):
    # pytroch 130
    # data_transform = v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()])
    # data_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        
    # pytorch 121
    data_transform = transforms.Compose([transforms.ToTensor(),])
    
    if training:
        pass   

    return data_transform

def get_batch_size(approach_name):
    match approach_name:
        case "a1":
            return 10
        case "a2":
            return 10
        case "a3":
            return 10
        case "a4":
            return 10
        case "a5":
            return 15
        case "a6":
            return 10
        case "a7":
            return 10
        case "a8":
            return 10
        case _:
            return "Not a valud approch" 

def create_model(approach_name, class_cnt):
    match approach_name:
        case "a1":
            return nn.Sequential(
                # layer 1
                nn.Conv2d(3, 32, kernel_size=3, padding=1), # takes 3 channel 32x32 image
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 16x16

                # layer 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1), # takes 3 channel 32x32 image
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 8x8

                # layer 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1), # takes 3 channel 32x32 image
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 4x4

                nn.Flatten(),

                # layer 4 Fully Connected 
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),

                # Output
                nn.Linear(256, class_cnt)
            )

        case "a2":
            pass
        case "a3":
            pass
        case "a4":
            pass
        case "a5":
            pass
        case "a6":
            pass
        case "a7":
            pass
        case "a8":
            pass
        case _:
            return "Not a valud approch" 

def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    criterion = nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimzer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimzer.step()
    
    return model        