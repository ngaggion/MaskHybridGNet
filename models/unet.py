import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4, reduction=16):
        super(SEResNeXtBlock, self).__init__()
        
        width = int(out_channels * (base_width / 64.)) * cardinality
        
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, track_running_stats=False)
        
        self.se = SEBlock(out_channels, reduction)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, track_running_stats=False)
            )

    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        
        if self.shortcut is not None:
            residual = self.shortcut(x)
        
        out += residual
        out = F.relu(out)
        
        return out

class UNet(nn.Module):
    def __init__(self, c = 4, n_classes = 4):
        super(UNet, self).__init__()
        
        self.c = c
        
        size = self.c * np.array([2,4,8,16,32], dtype = np.intc)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.dconv_down1 = SEResNeXtBlock(1, size[0])
        self.dconv_down2 = SEResNeXtBlock(size[0], size[1])
        self.dconv_down3 = SEResNeXtBlock(size[1], size[2])
        self.dconv_down4 = SEResNeXtBlock(size[2], size[3])
        self.dconv_down5 = SEResNeXtBlock(size[3], size[4])
        
        self.bottleneck = SEResNeXtBlock(size[4], size[4])
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up5 = SEResNeXtBlock(2 * size[4] , size[4])
        self.dconv_up4 = SEResNeXtBlock(size[4] + size[3] , size[3])
        self.dconv_up3 = SEResNeXtBlock(size[3] + size[2], size[2])
        self.dconv_up2 = SEResNeXtBlock(size[2] + size[1], size[1])
        self.dconv_up1 = SEResNeXtBlock(size[1] + size[0], size[0])
        self.conv_last = nn.Conv2d(size[0], n_classes, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        
        conv5 = self.dconv_down5(x)
        x = self.maxpool(conv5)
        
        self.bottle = self.bottleneck(x)
        
        x = self.upsample(self.bottle)
        x = torch.cat((x, conv5), dim=1)
        x = self.dconv_up5(x)
        
        x = self.upsample(x)
        x = torch.cat((x, conv4), dim=1)
        x = self.dconv_up4(x)
        
        x = self.upsample(x) 
        x = torch.cat((x, conv3), dim=1)
        x = self.dconv_up3(x)
      
        x = self.upsample(x)
        x = torch.cat((x, conv2), dim=1)
        x = self.dconv_up2(x)
        
        x = self.upsample(x)
        x = torch.cat((x, conv1), dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        
        return out