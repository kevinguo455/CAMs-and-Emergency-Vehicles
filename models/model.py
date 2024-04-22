import torch
import torchvision
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


# A default Resnet18 with no modifications.
class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')

    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def get_features(self, x):
        feature_extractor = create_feature_extractor(self, return_nodes={'backbone.layer4': 'layer4'})
        return feature_extractor(x)['layer4']    


# Resnet18 with a modified classification head to classify between num_classes options.
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def get_features(self, x):
        feature_extractor = create_feature_extractor(self, return_nodes={'backbone.layer4': 'layer4'})
        return feature_extractor(x)['layer4']