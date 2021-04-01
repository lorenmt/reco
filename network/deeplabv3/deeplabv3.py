from .aspp import *
from functools import partial


class DeepLabv3Plus(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=16, num_classes=2, output_dim=256):
        super(DeepLabv3Plus, self).__init__()
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            aspp_dilate = [12, 24, 36]

        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            aspp_dilate = [6, 12, 18]

        # take pre-defined ResNet, except AvgPool and FC
        self.resnet_conv1 = orig_resnet.conv1
        self.resnet_bn1 = orig_resnet.bn1
        self.resnet_relu1 = orig_resnet.relu
        self.resnet_maxpool = orig_resnet.maxpool

        self.resnet_layer1 = orig_resnet.layer1
        self.resnet_layer2 = orig_resnet.layer2
        self.resnet_layer3 = orig_resnet.layer3
        self.resnet_layer4 = orig_resnet.layer4

        self.ASPP = ASPP(2048, aspp_dilate)

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

        self.representation = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 1)
        )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)

            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # with ResNet-50 Encoder
        x = self.resnet_relu1(self.resnet_bn1(self.resnet_conv1(x)))
        x = self.resnet_maxpool(x)

        x_low = self.resnet_layer1(x)
        x = self.resnet_layer2(x_low)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        feature = self.ASPP(x)

        # Decoder
        x_low = self.project(x_low)
        output_feature = F.interpolate(feature, size=x_low.shape[2:], mode='bilinear', align_corners=True)
        prediction = self.classifier(torch.cat([x_low, output_feature], dim=1))
        representation = self.representation(torch.cat([x_low, output_feature], dim=1))
        return prediction, representation
