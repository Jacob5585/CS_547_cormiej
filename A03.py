import os
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from functools import partial

class CellFinder():
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.wbc_model_path = os.path.join(self.model_dir, "wbc_model_pth")
        self.rbc_model_path = os.path.join(self.model_dir, "rbc_model_pth")

        self.wbc_model = self._get_model(2)
        self.rbc_model = self._get_model(2)

    
    def train_WBC(self, train_data):
        self._train(self.wbc_model, train_data, 50, self.wbc_model_path)

    def find_WBC(self, image):
        pass

    def train_RBC(self, train_data):
        self._train(self.rbc_model, train_data, 50, self.wbc_model_path)

    def find_RBC(self, image):
        pass

    def _get_model(self, num_classes):
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights='DEFAULT')
        
        num_anchors = model.head.classification_head.num_anchors
        
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        
        return model.to(self.device)
    
    def _train(self, model, train_data, epochs, save_path):
        model.train()
        optimizer = torch.optim.Adam(model.parmameters(), lr=0.0001)

        for epoch in range(epochs):
            for image, target in train_data:
                image_tensor = F.to_tensor(image).to(self.device)

                boxes = torch.as_tensor(target['boxes'], dtype=torch.float32).to(self.device)
                labels = torch.ones((len(boxes),), dtype=torch.int64).to(self.device)

                targets = [{"boxes": boxes, "labels": labels}]

                loss_dict = model([image_tensor], targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backwards()
                optimizer.step()
            
            torch.save(model.state_dict(), save_path)