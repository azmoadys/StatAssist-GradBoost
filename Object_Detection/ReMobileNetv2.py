import torch
import torch.nn as nn
import math
#from math import round
import torch.utils.model_zoo as model_zoo


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


class InvertedBottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
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
    
class Bottleneck(nn.Module):
    #outchannel_ratio = 6

    def __init__(self, inplanes, planes, stride=1, dilation = 1, expand=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.convstack = nn.Sequential(
            nn.Conv2d(inplanes, planes*expand, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes*expand),
            nn.ReLU6(inplace=True),
            nn.Conv2d(planes*expand, planes*expand, kernel_size=3, stride=stride, padding=dilation, bias=False, groups=planes*expand, dilation = dilation),
            nn.BatchNorm2d(planes*expand),
            nn.ReLU6(inplace=True),
            nn.Conv2d(planes*expand, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes)
        ) 
        self.downsample = downsample
        self.stride = stride
 #       self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):

        out = self.convstack(x)

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = x.size()[1]
        if self.stride == 1: 
            featuremap_size = out.size()[2:4]
            if residual_channel != shortcut_channel:
                padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
                residual = torch.cat((x, padding), 1)
            else:
                residual = x 
            
            out += residual
#        out = self.relu(out)
        return out


class PyramidMobile(nn.Module):
        
    def __init__(self, depth, alpha, num_classes, bottleneck=False):
        super(PyramidMobile, self).__init__()   	
#        blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
#        layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        blocks = {50: Bottleneck, 32: Bottleneck, 20: Bottleneck, 17:Bottleneck}
        layers  = {50: [1, 2, 3, 4, 3, 3], 32: [1, 2, 2, 2, 2, 1], 20:[1,1,1,1,1,1], 17:[1,1,1,1,0,1]}
        stride_set   = {50: [1, 2, 2, 2, 1, 1], 32:[1,2,2,2,1,2], 20:[1,2,2,2,1,2], 17:[1,2,2,2,1,2]}
        expand_ratio = {50: [1, 6, 6, 6, 6, 6], 32: [1,6,6,6,6,6], 20:[1,1,1,1,1,1], 17:[1,1,1,1,1,1]}
        dilation_set = {50: [1, 1, 1, 1, 1, 2]}        
        self.inplanes = 32
        self.last_channel = 1280 
        n = (depth-2)//3
        self.addrate = alpha / (n*1.0)

        self.input_featuremap_dim = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU6(inplace=True)
        self.featuremap_dim = self.input_featuremap_dim 
        self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[depth][0], stride=stride_set[depth][0], dilation = dilation_set[depth][0], expand=expand_ratio[depth][0])
        self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[depth][1], stride=stride_set[depth][1], dilation = dilation_set[depth][1], expand=expand_ratio[depth][1])
        self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[depth][2], stride=stride_set[depth][2], dilation = dilation_set[depth][2], expand=expand_ratio[depth][2])
        self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[depth][3], stride=stride_set[depth][3], dilation = dilation_set[depth][3], expand=expand_ratio[depth][3])
 #       self.layer5 = self.pyramidal_make_layer(blocks[depth], layers[depth][4], stride=stride_set[depth][4], dilation = dilation_set[depth][4], expand=expand_ratio[depth][4])
        #self.layer6 = self.pyramidal_make_layer(blocks[depth], layers[depth][5], stride=stride_set[depth][5], dilation = dilation_set[depth][5], expand=expand_ratio[depth][5])
        
        self.final_featuremap_dim = self.input_featuremap_dim
        self.finalconv = conv_1x1_bn(self.final_featuremap_dim, self.last_channel)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1, dilation=1, expand=6):
        downsample = None
        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, dilation, expand,downsample))
#        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), 1, dilation, expand, None))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
#           if i == block_depth-1:
#                layers.append(block(int(round(self.featuremap_dim)), int(round(temp_featuremap_dim)), stride, dilation, expand, downsample))
#            else:
            layers.append(block(int(round(self.featuremap_dim)), int(round(temp_featuremap_dim)), 1, dilation, expand, None))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) 

        return nn.Sequential(*layers)
    
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.relu(x)
#        #x = self.conv2(x)
#
#        x = self.layer1(x)
#        x = self.layer2(x)
#        x = self.layer3(x)
#        x = self.layer4(x)
#        x = self.layer5(x)
#        x = self.layer6(x)
#        x = self.finalconv(x)
#
#        #x = self.bn_final(x)
#        #x = self.relu_final(x)
#        x = self.avgpool(x)
#        x = x.view(x.size(0), -1)
#        x = self.fc(x)
#    
#        return x
