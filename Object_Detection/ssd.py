import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco, TDSOD_voc, TDSOD_coco
import os
import resnet as RN
import MobileNetv2 as Mv2
import ReMobileNetv2 as RMv2

class SSD_TDSOD(nn.Module):
    def conv_bn(self,inp, oup, stride, k_size=3, p=1, group=1):
        return nn.Sequential(
            nn.Conv2d(inp, oup, k_size, stride, padding=p, bias=False, groups=group),
            nn.BatchNorm2d(oup),
            nn.ReLU()
        )
    
    def dwd_block(self, inp, oup):
        return nn.Sequential(
            self.conv_bn(inp=inp, oup=oup, stride=1, k_size=(1, 1), p=0),
            self.conv_bn(inp=oup, oup=oup, stride=1, k_size=(3, 3), p=1, group=oup)
        )
    
    def trans_block(self, inp, oup):
        return nn.Sequential(
            self.conv_bn(inp=inp, oup=oup, stride=1, k_size=(1, 1), p=0),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode = True)
        )
    
    def downsample(self, in_channels, out_channels):
        return (
            nn.Sequential(
                #nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, ceil_mode=True),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
            ,
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                # nn.BatchNorm2d(out_channels),
                # nn.ReLU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                          stride=2, padding=1, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        )
    
    def upsample(self, in_channels): # should use F.inpterpolate
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                      stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )


    def __init__(self, phase, size, head, num_classes):
        super(SSD_TDSOD, self).__init__()
        self.phase = phase
        self.size = size
        self.num_classes = num_classes
        self.cfg = (TDSOD_coco, TDSOD_voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.get_prior()
        self.priors.requires_grad = False

        self.base = nn.Sequential(
            self.conv_bn(inp=3,  oup=64, stride=2, k_size=(3, 3), p=1),
            self.conv_bn(inp=64, oup=64, stride=1, k_size=(1, 1), p=0),
            self.conv_bn(inp=64, oup=64, stride=1, k_size=(3, 3), p=1, group=64),
            self.conv_bn(inp=64, oup=128, stride=1, k_size=(1, 1), p=0),
            self.conv_bn(inp=128, oup=128, stride=1, k_size=(3, 3), p=1, group=128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
    
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
#            self.detect = detect(cfg)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

        # list of ddb and trans layers
        self.ddb_0 = []
        inp = 128
        for it in range(4):
            if it == 0:
                self.ddb_0.append(self.dwd_block(inp=inp, oup=32))
            else:
                inp += 32
                self.ddb_0.append(self.dwd_block(inp=inp, oup=32))
        self.ddb_0 = nn.ModuleList(self.ddb_0)
        self.trans_0 = self.trans_block(inp=256, oup=128) # output: 38x38

        self.ddb_1 = []
        inp = 128
        for it in range(6):
            if it == 0:
                self.ddb_1.append(self.dwd_block(inp=inp, oup=48))
            else:
                inp += 48
                self.ddb_1.append(self.dwd_block(inp=inp, oup=48))
        self.ddb_1 = nn.ModuleList(self.ddb_1)
        self.trans_1 = self.trans_block(inp=416, oup=128) # output: 19x19

        self.ddb_2 = []
        inp = 128
        for it in range(6):
            if it == 0:
                self.ddb_2.append(self.dwd_block(inp=inp, oup=64))
            else:
                inp += 64
                self.ddb_2.append(self.dwd_block(inp=inp, oup=64))
        self.ddb_2 = nn.ModuleList(self.ddb_2)
        self.trans_2 = self.conv_bn(inp=512, oup=256, stride=1, k_size=1, p=0) # output: 19x19

        self.ddb_3 = []
        inp = 256
        for it in range(6):
            if it == 0:
                self.ddb_3.append(self.dwd_block(inp=inp, oup=80))
            else:
                inp += 80
                self.ddb_3.append(self.dwd_block(inp=inp, oup=80))
        self.ddb_3 = nn.ModuleList(self.ddb_3)
        self.trans_3 = self.conv_bn(inp=736, oup=64, stride=1, k_size=1, p=0) # output: 19x19

        #list of upsample and downsample layers
        self.downfeat_0 = []
        self.downfeat_1 = []
        for it in range(5):
            if it == 1:
                self.downfeat_0.append(self.downsample(in_channels=128 + 64, out_channels=64)[0])
                self.downfeat_1.append(self.downsample(in_channels=128 + 64, out_channels=64)[1])
            else:
                self.downfeat_0.append(self.downsample(in_channels=128, out_channels=64)[0])
                self.downfeat_1.append(self.downsample(in_channels=128, out_channels=64)[1])

        self.upfeat = []
        for it in range(5):
            self.upfeat.append(self.upsample(in_channels=128))

        self.downfeat_0 = nn.ModuleList(self.downfeat_0)
        self.downfeat_1 = nn.ModuleList(self.downfeat_1)
        self.upfeat = nn.ModuleList(self.upfeat)


    def forward(self, x):
        """applies network layers and ops on input image(s) x.

        args:
            x: input image or batch of images. shape: [batch,3,300,300].

        return:
            depending on phase:
            test:
                variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, shape: [batch*num_priors,num_classes]
                    2: localization layers, shape: [batch,num_priors*4]
                    3: priorbox layers, shape: [2,num_priors*4]
        """
        size = x.size()[2:]
        sources = list()
        loc = list()
        conf = list()

        # apply base networks
        x = self.base(x) # output: 75x75

        infeat_0 = x

        # apply dwd_block 0
        for blc in self.ddb_0:
            x = torch.cat((x, blc(x)), 1)

        x = self.trans_0(x) # output: 38x38

        infeat_1 = x
        # apply dwd_block 1
        for blc in self.ddb_1:
            x = torch.cat((x, blc(x)), 1)

        x = self.trans_1(x) # output: 19x19
        
        #infeat_1 = x

        # apply dwd_block 2
        for blc in self.ddb_2:
            x = torch.cat((x, blc(x)), 1)

        x = self.trans_2(x)  # output: 19x19

        # apply dwd_block 3
        for blc in self.ddb_3:
            x = torch.cat((x, blc(x)), 1)

        x = self.trans_3(x)  # output: 19x19
        infeat_2 = x
        infeat_3 = torch.cat((self.downfeat_0[0](infeat_1), self.downfeat_1[0](infeat_1)), 1)
#        print('infeat_2',infeat_2.size())
#        print('infeat_3',infeat_3.size())

#        sz_x = min(infeat_3.size()[2], infeat_2.size()[2])
#        sz_y = min(infeat_3.size()[3], infeat_2.size()[3])
        sz_x = infeat_3.size()[2]
        sz_y = infeat_3.size()[3]

        # why should we use sz_x, sz_y?
        s0 = torch.cat((infeat_3[:, :, :sz_x, :sz_y], infeat_2[:, :, :sz_x, :sz_y]), 1)
        s1 = torch.cat((self.downfeat_0[1](s0), self.downfeat_1[1](s0)), 1) 
        s2 = torch.cat((self.downfeat_0[2](s1), self.downfeat_1[2](s1)), 1) 
        s3 = torch.cat((self.downfeat_0[3](s2), self.downfeat_1[3](s2)), 1) 
        s4 = torch.cat((self.downfeat_0[4](s3), self.downfeat_1[4](s3)), 1) 

        sources.append(s4)

        u1 = self.upfeat[0](F.interpolate(s4, size=(s3.size()[2],s3.size()[3]), mode='bilinear')) + s3
        sources.append(u1)
        u2 = self.upfeat[1](F.interpolate(u1, size=(s2.size()[2],s2.size()[3]), mode='bilinear')) + s2
        sources.append(u2)
        u3 = self.upfeat[2](F.interpolate(u2, size=(s1.size()[2],s1.size()[3]), mode='bilinear')) + s1
        sources.append(u3)
        u4 = self.upfeat[3](F.interpolate(u3, size=(infeat_3.size()[2],infeat_3.size()[3]), mode='bilinear')) + infeat_3
        sources.append(u4)
        u5 = self.upfeat[4](F.interpolate(u4, size=(infeat_1.size()[2],infeat_1.size()[3]), mode='bilinear')) + infeat_1
        sources.append(u5)

####### Features should be reversed order
        sources = sources[::-1] ####### #reverse order

        # apply multibox head to source layers

        '''
        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])

        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)
        #print(loc_x.size(), conf_x.size(), max_conf.size())

        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())
        print(sources[0].size())

        for i in range(0, len(sources)):
            x = sources[i]
            print(x.size())
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())

        '''
        
        for (x, l, c) in zip(sources, self.loc, self.conf):
        #    print(x.size())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        '''
        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]
        '''
        #self.priorbox = PriorBox(size, features_maps, cfg)
        #self.priors = Variable(self.priorbox.forward(), volatile=True)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = self.detect.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )

        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            #m.bias.data.zero_()


class SSD_ReMobileNetv2(nn.Module):
    def __init__(self, phase, size, base, extras, extras_head_pos, head, num_classes, use_final_conv):
        super(SSD_ReMobileNetv2, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = base
        if use_final_conv == False:
            self.vgg.finalconv = None
        # Layer learns to scale the l2 normalized features from conv4_3
        #self.L2Norm = L2Norm(77, 20)
        self.extras = nn.ModuleList(extras)
        self.extras_head_pos = extras_head_pos

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
    
    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        x = self.vgg.conv1(x)
        x = self.vgg.bn1(x)
        x = self.vgg.relu(x)
        x = self.vgg.layer1(x)
        x = self.vgg.layer2(x)
        x = self.vgg.layer3(x)
        #s = self.L2Norm(x)
        s = x
        sources.append(s)
        #print(s.size())

        x = self.vgg.layer4(x)
        #x = self.vgg.layer5(x)
        #x = self.vgg.layer6(x)
       
        sources.append(x)
        #print(x.size())
#        x = self.vgg.finalconv(x)
#        sources.append(x)
        
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k in self.extras_head_pos:
                sources.append(x)
                #print(x.size())
        
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
#            print(l(x).size())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class SSD_MobileNetV2(nn.Module):
    def __init__(self, phase, size, base, extras, extras_head_pos, head, num_classes, use_final_conv):
        super(SSD_MobileNetV2, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = base
        if use_final_conv == False:
            self.vgg.finalconv = None
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(32, 20)
        self.extras = nn.ModuleList(extras)
        self.extras_head_pos = extras_head_pos

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(7):
            x = self.vgg.features[k](x)

        s = self.L2Norm(x)
#        s = x  ## when L2 norm is not used
        sources.append(s)

        # apply vgg up to fc7
        for k in range(7, len(self.vgg.features)):
            x = self.vgg.features[k](x)
        
        sources.append(x)
        
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k in self.extras_head_pos:
                sources.append(x)
        
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class SSD_resnet(nn.Module):
    def __init__(self, phase, size, base, extras, extras_head_pos, head, num_classes):
        super(SSD_resnet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = base
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.extras_head_pos = extras_head_pos

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        x = self.vgg.conv1(x)
        x = self.vgg.bn1(x)
        x = self.vgg.relu(x)
        x = self.vgg.maxpool(x)
        x = self.vgg.layer1(x)
        x = self.vgg.layer2(x)
        s = self.L2Norm(x)
        sources.append(s)
        #print(x.size())
        # apply vgg up to fc7
        x = self.vgg.layer3(x)
        x = self.vgg.layer4(x)
        sources.append(x)
        #print(x.size())
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k in self.extras_head_pos:
                sources.append(x)
        #        print(x.size())
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

class SSD_VGG(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, extras_head_pos, head, num_classes):
        super(SSD_VGG, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
#        print(self.vgg)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.extras_head_pos = extras_head_pos

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)
#        print(x.size())

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
#        print(x.size())

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k in self.extras_head_pos:
                sources.append(x)
#                print(x.size())
#
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, strides, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if v == 'P':
            layers += [nn.AvgPool2d(3,3)]
        else:
            if batch_norm == True:
                convbn = nn.Sequential(nn.Conv2d(in_channels, v, stride=strides[k], padding=(0, 1)[strides[k] == 2],
                        kernel_size=(1, 3)[flag]),
                        nn.BatchNorm2d(v))
            else:
                convbn = nn.Sequential(nn.Conv2d(in_channels, v, stride=strides[k], padding=(0, 1)[strides[k] == 2],
                        kernel_size=(1, 3)[flag]),
                        )
            layers += convbn
        flag = not flag
        in_channels = v
    return layers

def add_extras_bn_group(cfg, strides, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if v == 'P':
            layers += [nn.AvgPool2d(3,3)]
        else:
            if batch_norm == True:
                convbn = nn.Sequential(nn.Conv2d(in_channels, v, stride=strides[k], padding=(0, 1)[strides[k] == 2],
                        kernel_size=(1, 3)[flag], groups=(1,in_channels)[flag]),
                        nn.BatchNorm2d(v))
            else:
                convbn = nn.Sequential(nn.Conv2d(in_channels, v, stride=strides[k], padding=(0, 1)[strides[k] == 2],
                        kernel_size=(1, 3)[flag], groups=(1,in_channels)[flag]),
                        )
            layers += convbn
        flag = not flag
        in_channels = v
    return layers

def multibox_TDSOD(num_classes, cfg):
    loc_layers = []
    conf_layers= []

    loc_layers += [nn.Conv2d(128, cfg[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(128, cfg[0] * num_classes, kernel_size=3, padding=1)]

    for k in range(1, 6):
        loc_layers += [nn.Conv2d(128, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(128, cfg[k] * num_classes, kernel_size=3, padding=1)]

    return (loc_layers, conf_layers)

def multibox_vgg(vgg, extra_layers, cfg, extras_head_pos, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
#    for k, v in enumerate(extra_layers[1::2], 2):
#        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
#                                 * 4, kernel_size=3, padding=1)]
#        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
#                                  * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extras_head_pos, 2):
        if isinstance(extra_layers[v],  nn.Conv2d): 
            loc_layers += [nn.Conv2d(extra_layers[v].out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(extra_layers[v].out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(extra_layers[v-1].out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(extra_layers[v-1].out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

def multibox_rmv2(vgg, extra_layers, cfg, extras_head_pos, num_classes, batch_norm):
    loc_layers = []
    conf_layers = []
    source_1st = vgg.layer3[-1].convstack[6]
    # source_2nd = vgg.layer6[-1].convstack[6]
    #source_2nd = vgg.layer5[-1].convstack[6]
    source_2nd = vgg.layer4[-1].convstack[6]

    if batch_norm == False:
        loc_layers += [nn.Sequential(nn.Conv2d(source_1st.out_channels,
                                     cfg[0] * 4, kernel_size=3, padding=1))]
        conf_layers += [nn.Sequential(nn.Conv2d(source_1st.out_channels,
                            cfg[0] * num_classes, kernel_size=3, padding=1))]
        loc_layers += [nn.Sequential(nn.Conv2d(source_2nd.out_channels,
                                     cfg[1] * 4, kernel_size=3, padding=1))]
        conf_layers += [nn.Sequential(nn.Conv2d(source_2nd.out_channels,
                            cfg[1] * num_classes, kernel_size=3, padding=1))] 
    else:
        loc_layers += [nn.Sequential(nn.Conv2d(source_1st.out_channels,
                                     cfg[0] * 4, kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(cfg[0]*4))]
        conf_layers += [nn.Sequential(nn.Conv2d(source_1st.out_channels,
                            cfg[0] * num_classes, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(cfg[0] * num_classes))]
        loc_layers += [nn.Sequential(nn.Conv2d(source_2nd.out_channels,
                                     cfg[1] * 4, kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(cfg[1]*4))]
        conf_layers += [nn.Sequential(nn.Conv2d(source_2nd.out_channels,
                            cfg[1] * num_classes, kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(cfg[1] * num_classes))]
#    loc_layers += [nn.Conv2d(vgg.finalconv[0].out_channels,
#                                 cfg[1] * 4, kernel_size=3, padding=1)]
#    conf_layers += [nn.Conv2d(vgg.finalconv[0].out_channels,
#                        cfg[1] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extras_head_pos, 2):
        if isinstance(extra_layers[v],  nn.Conv2d): 
            idx = v
        else:
            if isinstance(extra_layers[v-1], nn.Conv2d):
                idx = v-1
            else:
                idx = v-2
        if batch_norm == False:
            loc_layers += [nn.Sequential(nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1))]
            conf_layers += [nn.Sequential(nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1))]
        else:
            loc_layers += [nn.Sequential(nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1), 
                                    nn.BatchNorm2d(cfg[k] * 4))]
            conf_layers += [nn.Sequential(nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1), 
                                    nn.BatchNorm2d(cfg[k] * num_classes))]
    return vgg, extra_layers, (loc_layers, conf_layers)

def multibox_mobilenetv2(vgg, extra_layers, cfg, extras_head_pos, num_classes, batch_norm):
    loc_layers = []
    conf_layers = []
    if batch_norm == False:
        loc_layers += [nn.Conv2d(vgg.features[6].conv[6].out_channels,
                                cfg[0] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg.features[6].conv[6].out_channels,
                                cfg[0] * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(vgg.features[-1][0].out_channels,
                                cfg[1] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg.features[-1][0].out_channels,
                                cfg[1] * num_classes, kernel_size=3, padding=1)]
    else:
        loc_layers += [nn.Sequential(nn.Conv2d(vgg.features[6].conv[6].out_channels,
                                     cfg[0] * 4, kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(cfg[0]*4))]
        conf_layers += [nn.Sequential(nn.Conv2d(vgg.features[6].conv[6].out_channels,
                                     cfg[0] * num_classes, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(cfg[0] * num_classes))]
        loc_layers += [nn.Sequential(nn.Conv2d(vgg.features[-1][0].out_channels,
                                     cfg[1] * 4, kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(cfg[1]*4))]
        conf_layers += [nn.Sequential(nn.Conv2d(vgg.features[-1][0].out_channels,
                            cfg[1] * num_classes, kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(cfg[1] * num_classes))]
    
    for k, v in enumerate(extras_head_pos, 2):
        if isinstance(extra_layers[v],  nn.Conv2d): 
            idx = v
        else:
            if isinstance(extra_layers[v-1], nn.Conv2d):
                idx = v-1
            else:
                idx = v-2
        if batch_norm == False:
            loc_layers += [nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Sequential(nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1), 
                                    nn.BatchNorm2d(cfg[k] * 4))]
            conf_layers += [nn.Sequential(nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1), 
                                    nn.BatchNorm2d(cfg[k] * num_classes))]
    return vgg, extra_layers, (loc_layers, conf_layers)

def multibox_resnet(vgg, extra_layers, cfg, extras_head_pos, num_classes, batch_norm):
    loc_layers = []
    conf_layers = []
    if batch_norm == False:
        loc_layers += [nn.Conv2d(vgg.layer2[-1].conv3.out_channels,
                                     cfg[0] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg.layer2[-1].conv3.out_channels,
                            cfg[0] * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(vgg.layer4[-1].conv3.out_channels,
                                     cfg[1] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg.layer4[-1].conv3.out_channels,
                            cfg[1] * num_classes, kernel_size=3, padding=1)]
    else:
        loc_layers += [nn.Sequential(nn.Conv2d(vgg.layer2[-1].conv3.out_channels,
                                     cfg[0] * 4, kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(cfg[0]*4))]
        conf_layers += [nn.Sequential(nn.Conv2d(vgg.layer2[-1].conv3.out_channels,
                                     cfg[0] * num_classes, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(cfg[0] * num_classes))]
        loc_layers += [nn.Sequential(nn.Conv2d(vgg.layer4[-1].conv3.out_channels,
                                     cfg[1] * 4, kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(cfg[1]*4))]
        conf_layers += [nn.Sequential(nn.Conv2d(vgg.layer4[-1].conv3.out_channels,
                            cfg[1] * num_classes, kernel_size=3, padding=1), 
                                     nn.BatchNorm2d(cfg[1] * num_classes))]

    print(extra_layers)
    for k, v in enumerate(extras_head_pos, 2):
        if isinstance(extra_layers[v],  nn.Conv2d): 
            idx = v
        else:
            if isinstance(extra_layers[v-1], nn.Conv2d):
                idx = v-1
            else:
                idx = v-2
        if batch_norm == False:
            loc_layers += [nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Sequential(nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1), 
                                    nn.BatchNorm2d(cfg[k] * 4))]
            conf_layers += [nn.Sequential(nn.Conv2d(extra_layers[idx].out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1), 
                                    nn.BatchNorm2d(cfg[k] * num_classes))]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    'VGG300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    'VGG300': [256, 512, 128, 256, 128, 256, 128, 256],
#    'resnet300': [128, 512, 256, 512, 256, 512, 'P'],
    'resnet300': [128, 256, 128, 256, 128, 256, 'P'],
#    'mobilenetv2300': [256, 512, 256, 512, 256, 512, 'P'],
#    'rmobilenetv2300': [256, 512, 256, 512, 256, 512, 'P'],
#    'mobilenetv2300': [128, 256, 128, 256, 128, 256, 'P'],
#    'rmobilenetv2300': [128, 256, 128, 256, 128, 256, 'P'],
    'mobilenetv2300': [32, 128, 32, 128, 32, 128, 'P'],
#    'rmobilenetv2300': [32, 128, 32, 128, 32, 128, 'P'],
#    'mobilenetv2300': [32, 32, 32, 32, 32, 32, 'P'],
    'rmobilenetv2300': [32, 32, 32, 32, 32, 32, 'P'],
    '512': [],
}
strides={
    'VGG300': [1,2,1,2,1,1,1,1],
    'resnet300': [1,2,1,2,1,2,1],
    'mobilenetv2300': [1,2,1,2,1,2,1],
    'rmobilenetv2300': [1,2,1,2,1,2,1],
}
extras_head_pos = {
    'VGG300': [1, 3, 5, 7],
    'resnet300': {'None' : [1, 3, 5, 6], 'BN': [3, 7, 11, 12]},
    'mobilenetv2300': {'None' : [1, 3, 5, 6], 'BN': [3, 7, 11, 12]},
    'rmobilenetv2300': {'None' : [1, 3, 5, 6], 'BN': [3, 7, 11, 12]},
    #'rmobilenetv2300': [1, 3, 5, 6],
#    'mobilenetv2300': [3, 7, 11, 12],  # when bn exists
    #'rmobilenetv2300': [3, 7, 11, 12] # when bn exists,
    
}
mbox = {
    'VGG300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    'resnet300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    'mobilenetv2300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    'rmobilenetv2300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    'tdsod300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
#    'tdsod300': [4, 4, 6, 6, 6, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(nettype, phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    if nettype =='VGG':
        batch_norm_on_head = True
        batch_norm_on_extra_layers = True
        if batch_norm_on_extra_layers == False:
            extras_head_pos_on_model = extras_head_pos[nettype+str(size)]['None']
        else:
            extras_head_pos_on_model = extras_head_pos[nettype+str(size)]['BN']
        base_, extras_, head_ = multibox_vgg(vgg(base[nettype+str(size)], 3),
                                         add_extras(extras[nettype+str(size)], 
                                            strides[nettype+str(size)], 1024),
                                         mbox[nettype+str(size)], 
                                         extras_head_pos[nettype+str(size)],  num_classes)
        return SSD_VGG(phase, size, base_, extras_, extras_head_pos[nettype+str(size)],  head_, num_classes)
    elif nettype == 'resnet':
        model = RN.ResNet(50,1000,True)
        batch_norm_on_head = True
        batch_norm_on_extra_layers = True
        if batch_norm_on_extra_layers == False:
            extras_head_pos_on_model = extras_head_pos[nettype+str(size)]['None']
        else:
            extras_head_pos_on_model = extras_head_pos[nettype+str(size)]['BN']
        base_, extras_, head_ = multibox_resnet(model,
                                         add_extras(extras[nettype+str(size)], 
                                            strides[nettype+str(size)], 2048,
                                            batch_norm=batch_norm_on_extra_layers),mbox[nettype+str(size)], 
                                        extras_head_pos_on_model, num_classes,
                                            batch_norm=batch_norm_on_head)
        return SSD_resnet(phase, size, base_, extras_, extras_head_pos_on_model,  head_, num_classes)
    elif nettype == 'mobilenetv2':
        model = Mv2.MobileNetV2('ImageNet',1000, 224, 1)
        use_final_conv = True
        batch_norm_on_head = False
        batch_norm_on_extra_layers = False#True
        if use_final_conv == True:
            final_feat_dim = 1280
        else:
            final_feat_dim = 152
        if batch_norm_on_extra_layers == False:
            extras_head_pos_on_model = extras_head_pos[nettype+str(size)]['None']
        else:
            extras_head_pos_on_model = extras_head_pos[nettype+str(size)]['BN']
        base_, extras_, head_ = multibox_mobilenetv2(model,
                                         add_extras_bn_group(extras[nettype+str(size)], 
                                            strides[nettype+str(size)], final_feat_dim, 
                                            batch_norm=batch_norm_on_extra_layers), mbox[nettype+str(size)], 
                                        extras_head_pos_on_model, num_classes, 
                                            batch_norm=batch_norm_on_head)
        return SSD_MobileNetV2(phase, size, base_, extras_, extras_head_pos_on_model,  head_, num_classes, use_final_conv)
    elif nettype == 'rmobilenetv2':
        model = RMv2.PyramidMobile(50,120, 1000, True)
        use_final_conv = False
        batch_norm_on_head = False
        batch_norm_on_extra_layers = True
        if use_final_conv == True:
            final_feat_dim = 1280
        else:
            final_feat_dim = 107 # 130 # 152

        if batch_norm_on_extra_layers == False:
            extras_head_pos_on_model = extras_head_pos[nettype+str(size)]['None']
        else:
            extras_head_pos_on_model = extras_head_pos[nettype+str(size)]['BN']
        base_, extras_, head_ = multibox_rmv2(model,
                                         add_extras_bn_group(extras[nettype+str(size)], 
                                            strides[nettype+str(size)], final_feat_dim, 
                                            batch_norm=batch_norm_on_extra_layers), mbox[nettype+str(size)], 
                                        extras_head_pos_on_model, num_classes, 
                                            batch_norm=batch_norm_on_head)
        return SSD_ReMobileNetv2(phase, size, base_, extras_, extras_head_pos_on_model,  head_, num_classes, use_final_conv)
    elif nettype == 'tdsod':
        head_ = multibox_TDSOD(num_classes, mbox[nettype+str(size)]) 

        return SSD_TDSOD(phase, size, head_, num_classes)

