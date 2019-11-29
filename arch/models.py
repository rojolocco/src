import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from arch import inceptionv4A
from arch import nasnet
from arch import pnasnet
from arch import senet
from arch import xception
from arch import xception_dropout
from arch import xceptionFPN_multiscale


class Xception(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_labels):
        super(Xception, self).__init__()
        self.xception = xception.xception(pretrained=False) # input size >= 299x299
        num_features = self.xception.num_classes # 1000
        self.classifier = nn.Linear(num_features, num_labels)#*3)
        #self.num_labels = num_labels
        #self.num_classes = 3 # [p0, p1, p2] for each label

    def forward(self, x):
        x = self.xception(x)
        x = self.classifier(x)
        # we don't include sigmoid layer here
        return x #.reshape([len(x), self.num_labels, self.num_classes])


class NasNet(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_labels):
        super(NasNet, self).__init__()
        self.nasnet = nasnet.nasnetalarge(pretrained=False) # input size >= 299x299

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout()
        self.last_linear = nn.Linear(24 * 4032//24, num_labels)

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x
    def forward(self, x):
        x = self.nasnet.features(x)
        x= self.logits(x)
        # x = self.classifier(x)
        # we don't include sigmoid layer here
        return x#.reshape([len(x), self.num_labels, self.num_classes])


class pNasNet(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_labels):
        super(pNasNet, self).__init__()
        self.pnasnet = pnasnet.pnasnet5large(pretrained=False) # input size >= 299x299

        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(4320, num_labels)

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x
    def forward(self, x):
        x = self.pnasnet.features(x)
        x= self.logits(x)
        # x = self.classifier(x)
        # we don't include sigmoid layer here
        return x#.reshape([len(x), self.num_labels, self.num_classes])

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SAModule(nn.Module):
    """
    Re-implementation of spatial attention module (SAM) described in:
    *Liu et al., Dual Attention Network for Scene Segmentation, cvpr2019
    code reference:
    https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py
    """

    def __init__(self, num_channels):
        super(SAModule, self).__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels//8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels//8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_map):
        batch_size, num_channels, height, width = feat_map.size()

        conv1_proj = self.conv1(feat_map).view(batch_size, -1,
                                               width*height).permute(0, 2, 1)

        conv2_proj = self.conv2(feat_map).view(batch_size, -1, width*height)

        relation_map = torch.bmm(conv1_proj, conv2_proj)
        attention = F.softmax(relation_map, dim=-1)

        conv3_proj = self.conv3(feat_map).view(batch_size, -1, width*height)

        feat_refine = torch.bmm(conv3_proj, attention.permute(0, 2, 1))
        feat_refine = feat_refine.view(batch_size, num_channels, height, width)

        feat_map = self.gamma * feat_refine + feat_map

        return feat_map


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Xception_FPN(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_labels):
        super(Xception_FPN, self).__init__()
        #pretrained='imagenet'
        self.xception = xceptionFPN_multiscale.xception_fpn(pretrained=False) # input size >= 299x299

        self.conv1 = SeparableConv2d(2048, 2048, 3, 2,2)
        self.bn1 = nn.BatchNorm2d(2048)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = SeparableConv2d(2048, 2048, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(2048)
        self.relu2 = nn.ReLU(inplace=True)

        self.up2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2_2 = nn.Upsample(size=22, mode='bilinear', align_corners=True)
        self.merge1 = conv3x3(512 * 4, 512 * 4)
        self.merge2 = conv3x3(512 * 4, 512 * 4)

        self.attention = SAModule(2048)
        self.last_linear = nn.Linear(2048, num_labels)
        #self.num_labels = num_labels
        #self.num_classes = 3 # [p0, p1, p2] for each label
    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x
    def forward(self, x):
        x = self.xception.features(x)

        down64 = self.conv1(x)
        down64 = self.bn1(down64)
        down64 = self.relu1(down64)
        down128 = self.conv2(down64)
        down128 = self.bn2(down128)
        down128 = self.relu2(down128)

        up64 = self.up2_1(down128)
        merge64 = F.relu(self.merge1(up64 + down64), inplace=True)#up11 down12 bug
        up32 = self.up2_2(merge64)
        x = F.relu(self.merge2(up32 + x), inplace=True)

        x = self.attention(x)
        x = self.logits(x)
        # we don't include sigmoid layer here
        return x#.reshape([len(x), self.n


class SEnet(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_labels):
        super(SEnet, self).__init__()
        self.senet = senet.se_resnext101_32x4d(pretrained=None) # input size >= 299x299
        # self.attention = CAModule(512 * 4)#加了SA
        self.avg_pool = nn.AvgPool2d(22, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.last_linear = nn.Linear(512 * 4, num_labels)

    def logits(self, input):
        x = self.avg_pool(input)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.senet.features(x)
        # x=self.attention(x)
        x= self.logits(x)
        # x = self.classifier(x)
        # we don't include sigmoid layer here
        return x#.reshape([len(x), self.num_labels, self.num_classes])


class InceptionV4(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_labels):
        super(InceptionV4, self).__init__()
        self.inceptionv4 = inceptionv4A.inceptionvF(pretrained=False) # input size >= 299x299
        self.last_linear = nn.Linear(1536, num_labels)
        #self.num_labels = num_labels
        #self.num_classes = 3 # [p0, p1, p2] for each label
    def logits(self, features):
        #Allows image of any size to be processed
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.inceptionv4.features(x)
        x = self.logits(x)
        # we don't include sigmoid layer here
        return x#.reshape([len(x), self.num_labels, self.num_classes])


class Xception_SA(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_labels):
        super(Xception_SA, self).__init__()
        #pretrained='imagenet'
        self.xception = xception_dropout.xception_drop(pretrained=False) # input size >= 299x299
        self.attention = SAModule(2048)
        self.dropout = nn.Dropout(0.3)
        self.last_linear = nn.Linear(2048, num_labels)
        #self.num_labels = num_labels
        #self.num_classes = 3 # [p0, p1, p2] for each label
    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.xception.features(x)
        x = self.attention(x)
        x = self.logits(x)
        # we don't include sigmoid layer here
        return x#.reshape([len(x), self.n


class Pnasnet_SA(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_labels):
        super(Pnasnet_SA, self).__init__()
        self.pnasnet = pnasnet.pnasnet5large(pretrained=False) # input size >= 299x299
        self.attention = SAModule(4320)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(4320, num_labels)

    def logits(self, features):
        x = self.relu(features)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x
    def forward(self, x):
        x = self.pnasnet.features(x)
        x = self.attention(x)
        x= self.logits(x)
        # x = self.classifier(x)
        # we don't include sigmoid layer here
        return x#.reshape([len(x), self.num_labels, self.num_classes])


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, state_dim,  name='', out_state_dim=None):
        super(GraphConvolution, self).__init__()
        self.state_dim = state_dim

        if out_state_dim == None:
            self.out_state_dim = state_dim
        else:
            self.out_state_dim = out_state_dim
        self.fc1 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.out_state_dim,
        )

        self.fc2 = nn.Linear(
            in_features=self.state_dim,
            out_features=self.out_state_dim,
        )

        self.name = name
    def forward(self, input, adj):

        state_in = self.fc1(input)


        forward_input = self.fc2(torch.bmm(adj, input))

        return state_in + forward_input



    def __repr__(self):
        return self.__class__.__name__ + ' (' +  self.name + ')'


class GraphResConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, state_dim, name=''):
        super(GraphResConvolution, self).__init__()
        self.state_dim = state_dim

        self.gcn_1 = GraphConvolution(state_dim, '%s_1' % name)
        self.gcn_2 = GraphConvolution(state_dim, '%s_2' % name)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.name = name

    def forward(self, input, adj):

        output_1 = self.gcn_1(input, adj)
        output_1_relu = self.relu1(output_1)

        output_2 =  self.gcn_2(output_1_relu, adj)

        output_2_res = output_2 + input

        output = self.relu2(output_2_res)

        return output



    def __repr__(self):
        return self.__class__.__name__ + ' (' +  self.name + ')'


class Xception_GCN(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_labels):
        super(Xception_GCN, self).__init__()
        #pretrained='imagenet'
        self.xception = xception_dropout.xception_drop(pretrained=False) # input size >= 299x299
        self.attention = SAModule(2048)
        self.dropout = nn.Dropout(0.2)
        self.conv = nn.Conv2d(2048, 14, kernel_size=1, stride=1, padding=0)
        self.state_dim = 256
        self.gcn_0 = GraphConvolution(256, 'gcn_0', out_state_dim=self.state_dim)
        self.gcn_res_1 = GraphResConvolution(self.state_dim, 'gcn_res_1')
        self.gcn_res_2 = GraphResConvolution(self.state_dim, 'gcn_res_2')
        self.gcn_3 = GraphConvolution(self.state_dim, 'gcn_7', out_state_dim=32)

        self.fc = nn.Linear(
            in_features=32,
            out_features=1,
        )

        self.relu = nn.ReLU()

    def _normalize(self, A):
        # A = A+I
        A = A + torch.eye(A.size(0))
        d = A.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)

    def _generate_A(self):
        A = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ])

        return torch.from_numpy(A).float()

    def forward(self, x):
        A = self._generate_A()
        A_norm = self._normalize(A).cuda().unsqueeze(0)
        batch_size = x.size(0)
        adj = A_norm
        for i in range(batch_size-1):
            adj = torch.cat((adj, A_norm), 0)
        x = self.xception.features(x)
        x = self.attention(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.dropout(x)
        #
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (16, 16))
        x = x.view(x.size(0), x.size(1), -1)
        input = self.gcn_0(x, adj)
        input = self.gcn_res_1(input, adj)
        input = self.gcn_res_2(input, adj)
        output = self.gcn_3(input, adj)
        output = self.fc(output)
        return output.squeeze(2)


if __name__ == "__main__":
    x = torch.FloatTensor([0,1,2,5,2,3,0,1,2,3,4,5,0,1,2,3,4,5])
    print(x.reshape([6,3]))

    x = x.reshape([2,3,3])
    print(x.detach())
    print(x.detach().max(0))
    print(x.detach().max(1))
    print(x.detach().max(2))
    print(x.detach().max(-1))

    x1 = [1,2,3]
    x2 = [4,3,5]
    data = {'Train Loss':x1,'Valid Loss':x2}
    df = pd.DataFrame(data = data)
    df.index.name = 'epoch'
    print(df)