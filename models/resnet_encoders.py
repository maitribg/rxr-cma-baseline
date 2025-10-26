"""ResNet-based visual encoders"""

import torch
import torch.nn as nn
import torchvision.models as models


class VlnResnetDepthEncoder(nn.Module):
    """ResNet encoder for depth images"""
    
    def __init__(self, observation_space, output_size=128, backbone="resnet50"):
        super().__init__()
        
        # Use pretrained ResNet
        if backbone == "resnet50":
            resnet = models.resnet50(pretrained=True)
        else:
            resnet = models.resnet18(pretrained=True)
        
        # Modify first conv for single-channel depth
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        
        # Use ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Output projection
        self.fc = nn.Linear(2048 if backbone == "resnet50" else 512, output_size)
        
    def forward(self, observations):
        depth = observations["depth"]
        
        # Normalize depth
        depth = depth.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        depth = depth / 10.0  # Normalize to roughly [0, 1]
        
        # ResNet forward
        x = self.conv1(depth)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class TorchVisionResNet50(nn.Module):
    """ResNet50 encoder for RGB images"""
    
    def __init__(self, output_size=256, backbone="resnet50"):
        super().__init__()
        
        # Load pretrained ResNet
        resnet = models.resnet50(pretrained=True)
        
        # Use all layers except final FC
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Output projection
        self.fc = nn.Linear(2048, output_size)
        
    def forward(self, observations):
        rgb = observations["rgb"]
        
        # Normalize RGB
        rgb = rgb.permute(0, 3, 1, 2).float() / 255.0  # [B, H, W, C] -> [B, C, H, W]
        
        # ResNet forward
        x = self.conv1(rgb)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x