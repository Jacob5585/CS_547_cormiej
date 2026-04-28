import torch
import torchvision
from torchvision.transform import v2

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
    if training:
        data_transform = v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()])
        data_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

def get_batch_size(approach_name):
    match approach_name:
        case "a1":
            return 10
        case "a2":
            return 20
        case "a3":
            return 30
        case "a4":
            return 40
        case "a5":
            return 25
        case "a6":
            return 5
        case "a7":
            return 1
        case "a8":
            return 50
        case _:
            return "Not a valud approch" 

def create_model(approach_name, class_cnt):
    match approach_name:
        case "a1":
            pass
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
    pass