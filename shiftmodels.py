import torch
import torch.nn as nn
import shifts
import torch.nn.functional as functional

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, shift_type, flattened, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        central_multiplier = 1 
        if (flattened==True):
            central_multiplier = self.expansion

        self.conv1 = nn.Conv2d(inplanes, planes * central_multiplier, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * central_multiplier)
        self.conv2 = nn.Conv2d(planes * central_multiplier, planes * central_multiplier, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * central_multiplier)
        self.conv3 = nn.Conv2d(planes * central_multiplier, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if (shift_type=='4C'):
            self.shift = shifts.shift_4C()
        elif (shift_type=='8C'): 
            self.shift = shifts.shift_8C()
        else:
            print("No valid shift type input in bottleneck")
            raise KeyboardInterrupt

    def forward(self, x):
        identity = x
        out = self.shift(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        if (self.stride==2):
            out = functional.avg_pool2d(out,2)
        rl_1 = out
        out = self.shift(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out+rl_1)
        out = self.shift(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            if (self.stride==2):
                identity = functional.avg_pool2d(identity,2)
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, shift_type, flattened = False, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0], shift_type, flattened)
        self.layer2 = self._make_layer(block, 128, layers[1], shift_type, flattened, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shift_type, flattened, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shift_type, flattened, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)        

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, shift_type, flattened, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, shift_type, flattened, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, shift_type, flattened))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def shift_resnet101(shift_type):
    model = ResNet(Bottleneck, [3, 4, 23, 3], shift_type, flattened = False, num_classes=1000, zero_init_residual=False)

    return model

def shift_flattened_resnet35(shift_type):

    model = ResNet(Bottleneck, [1, 1, 8, 1], shift_type, flattened = True, num_classes=1000, zero_init_residual=False)

    return model


