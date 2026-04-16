import os
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms import functional as F
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
        self._train(self.wbc_model, train_data, 5, self.wbc_model_path)

    def find_WBC(self, image):
        if os.path.exists(self.wbc_model_path):
            self.wbc_model.load_state_dict(torch.load(self.wbc_model_path, map_location=self.device))

        return self._find_cell(image, self.wbc_model)

    def train_RBC(self, train_data):
        self._train(self.rbc_model, train_data, 5, self.rbc_model_path)

    def find_RBC(self, image):
        if os.path.exists(self.rbc_model_path):
            self.rbc_model.load_state_dict(torch.load(self.rbc_model_path, map_location=self.device))

        return self._find_cell(image, self.rbc_model)

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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        for epoch in range(epochs):
            for image, objects in train_data:
                image_tensor = F.to_tensor(image).to(self.device)

                raw_boxes = objects['bbox'] 
                if len(raw_boxes) == 0: 
                    continue
                
                # Convert list of arrays to tensor and swap coordinates
                formatted_boxes = []
                for b in raw_boxes:
                    # formatted_boxes.append([b[1], b[0], b[3], b[2]])
                    xmin, ymin, xmax, ymax = b[1], b[0], b[3], b[2]

                    if xmax > xmin and ymax > ymin:
                        formatted_boxes.append([xmin, ymin, xmax, ymax])
                    else:
                        # Optional: Print a warning to see how much data you're losing
                        print(f"Skipping degenerate box: {[xmin, ymin, xmax, ymax]}")

                if len(formatted_boxes) == 0:
                    continue
                
                targets = [{
                    "boxes": torch.as_tensor(formatted_boxes, dtype=torch.float32).to(self.device),
                    "labels": torch.ones((len(formatted_boxes),), dtype=torch.int64).to(self.device)
                }]

                loss_dict = model([image_tensor], targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            
            torch.save(model.state_dict(), save_path)

    def _find_cell(self, image, model):
        model.eval()

        with torch.no_grad():
            image_tensor = F.to_tensor(image).to(self.device)
            prediction = model([image_tensor])[0]

            detected_boxes = []
            for box, score in zip(prediction['boxes'], prediction['scores']):
            # Filter detections by confidence to avoid 'too many cells' penalties 
                if score > 0.5:
                    # Convert back to (ymin, xmin, ymax, xmax) for evaluation [cite: 20, 143]
                    x1, y1, x2, y2 = box.tolist()
                    detected_boxes.append((int(y1), int(x1), int(y2), int(x2)))
        
        return detected_boxes