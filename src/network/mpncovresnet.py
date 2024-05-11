import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import torch.nn.init as init
from ..representation import MPNCOV, CovpoolLayer, SqrtmLayer, TriuvecLayer
from .resnet import BasicBlock, Bottleneck

from ..attention.SEAttention import SEAttention
from ..attention.ECAAttention import ECAAttention
from ..attention.CBAM import CBAMBlock
from ..attention.CoordAttention import CoordAtt
from ..attention.A2Atttention import DoubleAttention

__all__ = ['mpncovresnet18', 'mpncovresnet34', 'mpncovresnet50', 'mpncovresnet101',
           'mpncovresnet18tiny', 'mpncovresnet34tiny', 'mpncovresnet50tiny',
           'mpncovresnet18atten', 'mpncovresnet18attentiny',
           'mpncovresnet34atten', 'mpncovresnet34attentiny',
           'mpncovresnet50atten', 'mpncovresnet50attentiny',
           'mpncovresnet101atten', 'mpncovresnet101attentiny']

model_urls = {
    'mpncovresnet18': 'http://jtxie.com/models/mpncovresnet18.pth',
    'mpncovresnet34': 'http://jtxie.com/models/mpncovresnet34.pth',
    'mpncovresnet50': 'http://jtxie.com/models/mpncovresnet50.pth',
    'mpncovresnet101': 'http://jtxie.com/models/mpncovresnet101-ade9737a.pth',
    'mpncovresnet18-tiny': 'http://jtxie.com/models/mpncovresnet18-tiny.pth',
    'mpncovresnet34-tiny': 'http://jtxie.com/models/mpncovresnet34-tiny.pth',
    'mpncovresnet50-tiny': 'http://jtxie.com/models/mpncovresnet50-tiny.pth',
    'mpncovresnet101-tiny': 'http://jtxie.com/models/mpncovresnet101-tiny.pth',
}


def cov_feature(x):
    batchsize = x.data.shape[0]
    dim = x.data.shape[1]
    h = x.data.shape[2]
    w = x.data.shape[3]
    M = h * w
    x = x.reshape(batchsize, dim, M)
    I_hat = (-1. / M / M) * torch.ones(dim, dim, device=x.device) + (1. / M) * torch.eye(dim, dim, device=x.device)
    I_hat = I_hat.view(1, dim, dim).repeat(batchsize, 1, 1).type(x.dtype)
    y = (x.transpose(1, 2)).bmm(I_hat).bmm(x)
    return y


class AttentionBlock(nn.Module):
    def __init__(self, inplanes, planes, att_dim=64):
        super(AttentionBlock, self).__init__()

        self.ch_dim = att_dim

        self.conv_for_DR = nn.Conv2d(inplanes, self.ch_dim, kernel_size=1, stride=1, bias=False)
        self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(int(self.ch_dim * (self.ch_dim + 1) / 2), planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        # NxCxHxW
        out = self.conv_for_DR(x)
        out = self.bn_for_DR(out)
        out = self.relu(out)

        out = CovpoolLayer(out)
        out = SqrtmLayer(out, 5)
        out = TriuvecLayer(out)
        out = out.view(out.size(0), -1)

        out = self.fc(out)  # NxC
        out = out.view(out.size(0), out.size(1), 1, 1).contiguous()  # NxCx1x1
        out = self.sigmoid(out)  # NxCx1x1

        out = residual * out

        return out


class BottleneckAtten(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention='0', att_dim=128):
        super(BottleneckAtten, self).__init__()
        self.dimDR = att_dim
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.relu_normal = nn.ReLU(inplace=False)
        if attention in {'1', '+', 'M', '&'}:
            if planes > 64:
                DR_stride = 1
            else:
                DR_stride = 2

            self.ch_dim = att_dim
            self.conv_for_DR = nn.Conv2d(
                planes * self.expansion, self.ch_dim,
                kernel_size=1, stride=DR_stride, bias=True)
            self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
            self.row_bn = nn.BatchNorm2d(self.ch_dim)
            # row-wise conv is realized by group conv
            self.row_conv_group = nn.Conv2d(
                self.ch_dim, 4 * self.ch_dim,
                kernel_size=(self.ch_dim, 1),
                groups=self.ch_dim, bias=True)
            self.fc_adapt_channels = nn.Conv2d(
                4 * self.ch_dim, planes * self.expansion,
                kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()

        if attention in {'2', '+', 'M', '&'}:
            self.sp_d = att_dim
            self.sp_h = 8
            self.sp_w = 8
            self.sp_reso = self.sp_h * self.sp_w
            self.conv_for_DR_spatial = nn.Conv2d(
                planes * self.expansion, self.sp_d,
                kernel_size=1, stride=1, bias=True)
            self.bn_for_DR_spatial = nn.BatchNorm2d(self.sp_d)

            self.adppool = nn.AdaptiveAvgPool2d((self.sp_h, self.sp_w))
            self.row_bn_for_spatial = nn.BatchNorm2d(self.sp_reso)
            # row-wise conv is realized by group conv
            self.row_conv_group_for_spatial = nn.Conv2d(
                self.sp_reso, self.sp_reso * 4, kernel_size=(self.sp_reso, 1),
                groups=self.sp_reso, bias=True)
            self.fc_adapt_channels_for_spatial = nn.Conv2d(
                self.sp_reso * 4, self.sp_reso, kernel_size=1, groups=1, bias=True)
            self.sigmoid = nn.Sigmoid()
            self.adpunpool = F.adaptive_avg_pool2d

        if attention == '&':  # we employ a weighted spatial concat to keep dim
            self.groups_base = 32
            self.groups = int(planes * self.expansion / 64)
            self.factor = int(math.log(self.groups_base / self.groups, 2))
            self.padding_num = self.factor + 2
            self.conv_kernel_size = self.factor * 2 + 5
            self.dilate_conv_for_concat1 = nn.Conv2d(planes * self.expansion,
                                                     planes * self.expansion,
                                                     kernel_size=(self.conv_kernel_size, 1),
                                                     stride=1, padding=(self.padding_num, 0),
                                                     groups=self.groups, bias=True)
            self.dilate_conv_for_concat2 = nn.Conv2d(planes * self.expansion,
                                                     planes * self.expansion,
                                                     kernel_size=(self.conv_kernel_size, 1),
                                                     stride=1, padding=(self.padding_num, 0),
                                                     groups=self.groups, bias=True)
            self.bn_for_concat = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride
        self.attention = attention

    def chan_att(self, out):
        # NxCxHxW
        out = self.relu_normal(out)
        out = self.conv_for_DR(out)
        out = self.bn_for_DR(out)
        out = self.relu(out)

        out = CovpoolLayer(out)  # Nxdxd
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous()  # Nxdxdx1

        out = self.row_bn(out)
        out = self.row_conv_group(out)  # Nx512x1x1

        out = self.fc_adapt_channels(out)  # NxCx1x1
        out = self.sigmoid(out)  # NxCx1x1

        return out

    def pos_att(self, out):
        pre_att = out  # NxCxHxW
        out = self.relu_normal(out)
        out = self.conv_for_DR_spatial(out)
        out = self.bn_for_DR_spatial(out)

        out = self.adppool(out)  # keep the feature map size to 8x8

        out = cov_feature(out)  # Nx64x64
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous()  # Nx64x64x1
        out = self.row_bn_for_spatial(out)

        out = self.row_conv_group_for_spatial(out)  # Nx256x1x1
        out = self.relu(out)

        out = self.fc_adapt_channels_for_spatial(out)  # Nx64x1x1
        out = self.sigmoid(out)
        out = out.view(out.size(0), 1, self.sp_h, self.sp_w).contiguous()  # Nx1x8x8

        out = self.adpunpool(out, (pre_att.size(2), pre_att.size(3)))  # unpool Nx1xHxW

        return out

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.attention == '1':  # channel attention,GSoP default mode
            pre_att = out
            att = self.chan_att(out)
            out = pre_att * att

        elif self.attention == '2':  # position attention
            pre_att = out
            att = self.pos_att(out)
            out = self.relu_normal(pre_att * att)

        elif self.attention == '+':  # fusion manner: average
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out = pre_att * chan_att + self.relu(pre_att.clone() * pos_att)

        elif self.attention == 'M':  # fusion manner: MAX
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out = torch.max(pre_att * chan_att, self.relu(pre_att.clone() * pos_att))

        elif self.attention == '&':  # fusion manner: concat
            pre_att = out
            chan_att = self.chan_att(out)
            pos_att = self.pos_att(out)
            out1 = self.dilate_conv_for_concat1(pre_att * chan_att)
            out2 = self.dilate_conv_for_concat2(self.relu(pre_att * pos_att))
            out = out1 + out2
            out = self.bn_for_concat(out)

        out += residual
        out = self.relu(out)

        return out


# class MPNCOVResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000):
#         self.inplanes = 64
#         super(MPNCOVResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
#         self.layer_reduce = nn.Conv2d(512 * block.expansion, 256, kernel_size=1, stride=1, padding=0,
#                                       bias=False)
#         self.layer_reduce_bn = nn.BatchNorm2d(256)
#         self.layer_reduce_relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(int(256 * (256 + 1) / 2), num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # 1x1 Conv. for dimension reduction
#         x = self.layer_reduce(x)
#         x = self.layer_reduce_bn(x)
#         x = self.layer_reduce_relu(x)
#
#         x = MPNCOV.CovpoolLayer(x)
#         x = MPNCOV.SqrtmLayer(x, 5)
#         x = MPNCOV.TriuvecLayer(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x


class MPNCOVResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, attention='Cov', input_size=None):
        self.inplanes = 64
        super(MPNCOVResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.layer_reduce = nn.Conv2d(512 * block.expansion, 256, kernel_size=1, stride=1, padding=0,
                                      bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(256)
        self.layer_reduce_relu = nn.ReLU(inplace=True)

        if attention == 'Cov':
            self.attention = AttentionBlock(256, 256, att_dim=64)
        elif attention == 'ECA':
            self.attention = ECAAttention(kernel_size=3)
        elif attention == 'SE':
            self.attention = SEAttention(channel=256, reduction=8)
        elif attention == 'CBAM':
            kernel_size = 8 if input_size == 128 else 14
            self.attention = CBAMBlock(channel=256, reduction=16, kernel_size=kernel_size - 1)
        elif attention == 'CA':
            self.attention = CoordAtt(256, 256)
        elif attention == 'A2':
            self.attention = DoubleAttention(256, 128, 128)
        else:
            self.attention = None

        self.fc = nn.Linear(int(256 * (256 + 1) / 2), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

        # 1x1 Conv. for dimension reduction
        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)

        x = self.attention(x)

        x = MPNCOV.CovpoolLayer(x)
        x = MPNCOV.SqrtmLayer(x, 5)
        x = MPNCOV.TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


"""
Tiny ResNet
"""


# class TinyMPNCOVResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000):
#         self.inplanes = 16
#         super(TinyMPNCOVResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self._make_layer(block, 16, layers[0])
#         self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
#         self.layer_reduce = nn.Conv2d(128 * block.expansion, 128, kernel_size=1, stride=1, padding=0,
#                                       bias=False)
#         self.layer_reduce_bn = nn.BatchNorm2d(128)
#         self.layer_reduce_relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(int(128 * (128 + 1) / 2), num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # 1x1 Conv. for dimension reduction
#         x = self.layer_reduce(x)
#         x = self.layer_reduce_bn(x)
#         x = self.layer_reduce_relu(x)
#
#         x = MPNCOV.CovpoolLayer(x)
#         x = MPNCOV.SqrtmLayer(x, 5)
#         x = MPNCOV.TriuvecLayer(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x


class TinyMPNCOVResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, attention='Cov', input_size=None):
        self.inplanes = 16
        super(TinyMPNCOVResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer_reduce = nn.Conv2d(128 * block.expansion, 128, kernel_size=1, stride=1, padding=0,
                                      bias=False)
        self.layer_reduce_bn = nn.BatchNorm2d(128)
        self.layer_reduce_relu = nn.ReLU(inplace=True)

        if attention == 'Cov':
            self.attention = AttentionBlock(128, 128, att_dim=64)
        elif attention == 'ECA':
            self.attention = ECAAttention(kernel_size=3)
        elif attention == 'SE':
            self.attention = SEAttention(channel=128, reduction=8)
        elif attention == 'CBAM':
            kernel_size = 4 if input_size == 32 else 8
            self.attention = CBAMBlock(channel=128, reduction=16, kernel_size=kernel_size - 1)
        elif attention == 'CA':
            self.attention = CoordAtt(128, 128)
        elif attention == 'A2':
            self.attention = DoubleAttention(128, 128, 128)
        else:
            self.attention = None

        self.fc = nn.Linear(int(128 * (128 + 1) / 2), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, attention=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        # 1x1 Conv. for dimension reduction
        x = self.layer_reduce(x)
        x = self.layer_reduce_bn(x)
        x = self.layer_reduce_relu(x)

        x = self.attention(x)

        x = MPNCOV.CovpoolLayer(x)
        x = MPNCOV.SqrtmLayer(x, 5)
        x = MPNCOV.TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def mpncovresnet18(pretrained=False, progress=True, **kwargs):
    model = MPNCOVResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mpncovresnet18'])['state_dict']
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model


def mpncovresnet18atten(pretrained=False, progress=True, **kwargs):
    model = MPNCOVResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['mpncovresnet18'])['state_dict']
        model_dict.update({k.replace('module.', ''): v for k, v in state_dict.items()})
        model.load_state_dict(model_dict)
    return model


def mpncovresnet18tiny(pretrained=False, progress=True, **kwargs):
    print("Using tiny size model!")
    model = TinyMPNCOVResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mpncovresnet18-tiny'])
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model


def mpncovresnet18attentiny(pretrained=False, progress=True, **kwargs):
    print("Using tiny size model!")
    model = TinyMPNCOVResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['mpncovresnet18-tiny'])
        model_dict.update({k.replace('module.', ''): v for k, v in state_dict.items()})
        model.load_state_dict(model_dict)
    return model


def mpncovresnet34(pretrained=False, progress=True, **kwargs):
    model = MPNCOVResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mpncovresnet34'])['state_dict']
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model


def mpncovresnet34atten(pretrained=False, progress=True, **kwargs):
    model = MPNCOVResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['mpncovresnet34'])['state_dict']
        model_dict.update({k.replace('module.', ''): v for k, v in state_dict.items()})
        model.load_state_dict(model_dict)
    return model


def mpncovresnet34tiny(pretrained=False, progress=True, **kwargs):
    print("Using tiny size model!")
    model = TinyMPNCOVResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mpncovresnet34-tiny'])
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model


def mpncovresnet34attentiny(pretrained=False, progress=True, **kwargs):
    print("Using tiny size model!")
    model = TinyMPNCOVResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['mpncovresnet34-tiny'])
        model_dict.update({k.replace('module.', ''): v for k, v in state_dict.items()})
        model.load_state_dict(model_dict)
    return model


def mpncovresnet50(pretrained=False, progress=True, **kwargs):
    model = MPNCOVResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mpncovresnet50'])['state_dict']
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model


def mpncovresnet50atten(pretrained=False, progress=True, **kwargs):
    model = MPNCOVResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['mpncovresnet50'])['state_dict']
        model_dict.update({k.replace('module.', ''): v for k, v in state_dict.items()})
        model.load_state_dict(model_dict)
    return model


def mpncovresnet50tiny(pretrained=False, progress=True, **kwargs):
    print("Using tiny size model!")
    model = TinyMPNCOVResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mpncovresnet50-tiny'])
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model


def mpncovresnet50attentiny(pretrained=False, progress=True, **kwargs):
    print("Using tiny size model!")
    model = TinyMPNCOVResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['mpncovresnet50-tiny'])
        model_dict.update({k.replace('module.', ''): v for k, v in state_dict.items() if
                           'fc' not in k or ' layer_reduce_bn' not in k})
        model.load_state_dict(model_dict)
    return model


def mpncovresnet101tiny(pretrained=False, progress=True, **kwargs):
    print("Using tiny size model!")
    model = TinyMPNCOVResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mpncovresnet101-tiny'])
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    return model


def mpncovresnet101attentiny(pretrained=False, progress=True, **kwargs):
    print("Using tiny size model!")
    model = TinyMPNCOVResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['mpncovresnet101-tiny'])
        model_dict.update({k.replace('module.', ''): v for k, v in state_dict.items() if
                           'fc' not in k or ' layer_reduce_bn' not in k})
        model.load_state_dict(model_dict)
    return model


def mpncovresnet101(pretrained=False, progress=True, **kwargs):
    model = _resnet_mpncov('resnet101_mpncov', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                           **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['mpncovresnet101']))
    return model


def mpncovresnet101atten(pretrained=False, progress=True, **kwargs):
    model = MPNCOVResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        state_dict = model_zoo.load_url(model_urls['mpncovresnet101'])
        model_dict.update({k.replace('module.', ''): v for k, v in state_dict.items() if
                           'fc' not in k or ' layer_reduce_bn' not in k})
        model.load_state_dict(model_dict)
    return model
