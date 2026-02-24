import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ==========================================
# ASPP Module
# ==========================================
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()

        modules = []

        # 1x1 conv
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Atrous conv
        for rate in atrous_rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Image pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(modules), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []

        for conv in self.convs[:-1]:
            res.append(conv(x))

        # Image pooling branch
        img_pool = self.convs[-1](x)
        img_pool = F.interpolate(
            img_pool,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        res.append(img_pool)

        res = torch.cat(res, dim=1)
        return self.project(res)


# ==========================================
# DeepLabV3+
# ==========================================
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DeepLabV3Plus, self).__init__()

        backbone = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        # Extract layers
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # low-level
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4  # high-level

        # ASPP on layer4
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # Reduce low-level feature
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]

        # Backbone
        x = self.layer0(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP
        x = self.aspp(x)

        # Upsample ASPP output
        x = F.interpolate(
            x,
            size=low_level.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Process low-level
        low_level = self.low_level_conv(low_level)

        # Concatenate
        x = torch.cat([x, low_level], dim=1)

        x = self.decoder(x)
        x = self.classifier(x)

        # Upsample to input size
        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return x