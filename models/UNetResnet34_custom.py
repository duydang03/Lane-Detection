import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class UNetResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super(UNetResNet34, self).__init__()

        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        self.input_layer = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.upconv4 = self._up_block(512, 256)
        self.upconv3 = self._up_block(256 + 256, 128)
        self.upconv2 = self._up_block(128 + 128, 64)
        self.upconv1 = self._up_block(64 + 64, 64)

        # Vì d1 concat với x1 nên in_channels = 64 + 64 = 128
        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.encoder1(self.pool(x1))
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        d4 = self.upconv4(x5)
        d4 = torch.cat([d4, x4], dim=1)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, x3], dim=1)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, x2], dim=1)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, x1], dim=1)

        out = self.final_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out
