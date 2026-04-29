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
            return "Not a valid approach"

def get_data_transform(approach_name, training):
    # pytroch 130
    # data_transform = v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()])
    # data_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        
    # pytorch 121
    # data_transform = transforms.Compose([transforms.ToTensor(),])
    # data_transform 
    
    if training:
        match approach_name:
            case "a1":
                data_transform = transforms.Compose([transforms.ToTensor(),])
                return data_transform

            case "a2":
                data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                return data_transform

            case "a3":
                data_transform = transforms.Compose([transforms.ToTensor(),])
                return data_transform

            case "a4":
                data_transform = transforms.Compose([transforms.ToTensor(),])
                return data_transform

            case "a5":
                data_transform = data_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                return data_transform

            case "a6":
                data_transform = transforms.Compose([transforms.ToTensor(),])
                return data_transform

            case "a7":
                data_transform = transforms.Compose([transforms.ToTensor(),])
                return data_transform

            case "a8":
                data_transform = data_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

                return data_transform

            case _:
                return "Not a valid approach"
    else:
        data_transform = transforms.Compose([transforms.ToTensor(),])
        return data_transform

    # return data_transform

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
            return "Not a valid approach"

def create_model(approach_name, class_cnt):
    match approach_name:
        case "a1":
        # base
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
        # add normalize augmentation to base
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

        case "a3":
            # Removed Fully connected layer and added batch norm
            return nn.Sequential(
                # Layer 1
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),   
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.1),

                # Layer 2
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),

                # Layer 4
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),

                # Output
                nn.Linear(64, class_cnt)
            )

        case "a4":
            # Add batch norm to base
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),

                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Linear(256, class_cnt)
            )

        case "a5":
            # Add batch norm to base then data augemntation
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),

                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Linear(256, class_cnt)
            )
            
        case "a6":
            # Add batch norm and dropout to base then data augemntation (removed color jitter and resize crop from a5)
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),

                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, class_cnt)
            )


        case "a7":
            # GO LONG
            return nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Block 4
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),

                # Fully connected layers
                nn.Linear(256 * 2 * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, class_cnt)
            )

        case "a8":
            # GO LONG with augmentations
            return nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Block 4
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),

                # Fully connected layers
                nn.Linear(256 * 2 * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, class_cnt)
            )

        case _:
            return "Not a valid approach"

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