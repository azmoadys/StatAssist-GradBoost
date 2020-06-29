import torch.nn as nn
import math

###V2<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
#        self.dilation = dilation

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, 
                      stride = stride, dilation = dilation, padding = dilation, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, dataset, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.dataset = dataset
        
        if self.dataset.startswith('cifar'):
            self.interverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1,  1],
                [6, 24, 2,  1],
                [6, 32, 3,  2],
                [6, 64, 4,  1],
                [6, 96, 3,  2],
                [6, 160, 3, 1],
                [6, 320, 1, 1],
            ]
            input_stride = 1
        else:
            self.interverted_residual_setting = [
                # t, c, n, s, d
                [1, 16, 1, 1, 1],
                [6, 24, 2, 2, 1],
                [6, 32, 3, 2, 1],
                [6, 64, 4, 2, 1],
                [6, 96, 3, 1, 1],
                [6, 160, 3, 1, 2],
                [6, 320, 1, 1, 2],
            ]
            input_stride = 2

        # building first layer
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, input_stride)]
        # building inverted residual blocks
        for t, c, n, s, d in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
#                if i == n-1:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, d, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, d, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
#        self.features.append(nn.AvgPool2d(input_size//32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        #self.classifier = nn.Sequential(
    #   #     nn.Dropout(0.2),
        #    nn.Linear(self.last_channel, n_class),
        #)

        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



#    def forward(self, x):
#        x = self.features(x)
#        x = x.view(-1, self.last_channel)
#        x = self.classifier(x)
#        return x

