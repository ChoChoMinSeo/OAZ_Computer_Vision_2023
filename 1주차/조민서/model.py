import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class half_efficientNetb0(nn.Module):
    def __init__(self):
        super(half_efficientNetb0, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0',in_channels=1,num_classes=10)
        self.layer1 = nn.Sequential(
            self.backbone._conv_stem,
            self.backbone._bn0
        )
        self.layer2 = self.backbone._blocks[0]
        self.layer3 = nn.Sequential(
            self.backbone._blocks[1],
            self.backbone._blocks[2]
        )
        self.layer4 = nn.Sequential(
            self.backbone._blocks[3],
            self.backbone._blocks[4]
        )
        self.conv_head = nn.Conv2d(40,1280,1,1,0)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_features=1280, out_features=10, bias=True)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.conv_head(out)
        out = self.avg_pooling(out)
        out = out.flatten(start_dim=1)
        out = self.fc(out)
        return out