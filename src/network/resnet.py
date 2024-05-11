import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.nn.init as init

from ..representation import MPNCOV, CovpoolLayer, SqrtmLayer, TriuvecLayer

__all__ = ['ResNet', 'resnet18', 'resnet18atten', 'resnet18attentiny', 'resnet34', 'resnet34atten', 'resnet34attentiny',
           'resnet50', 'resnet50atten', 'resnet50attentiny', 'resnet101', 'resnet152',
           'resnet18tiny', 'resnet34tiny', 'resnet50tiny']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet18-tiny': 'https://download.pytorch.org/models/resnet18-tiny.pth',
    'resnet34-tiny': 'https://download.pytorch.org/models/resnet34-tiny.pth',
    'resnet50-tiny': 'https://download.pytorch.org/models/resnet50-tiny.pth',
}


# Attention
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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, attention=False, input_size=None):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # self.attention = AttentionBlock(512 * block.expansion, 512 * block.expansion, att_dim=64) if attention else None
        self.attention = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
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

        x = self.attention(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


"""
Tiny ResNet

https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
"""


class TinyResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False, attention=False, input_size=None):
        super(TinyResNet, self).__init__()
        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], 2)

        # self.attention = AttentionBlock(128 * block.expansion, 128 * block.expansion, att_dim=64) if attention else None
        self.attention = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.attention(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet18atten(pretrained=False, progress=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], attention=True, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = model_zoo.load_url(model_urls['resnet18'])
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


def resnet18tiny(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Using tiny size model!")
    model = TinyResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = model_zoo.load_url(model_urls['resnet18-tiny'])
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


def resnet18attentiny(pretrained=False, progress=True, **kwargs):
    print("Using tiny size model!")
    model = TinyResNet(BasicBlock, [2, 2, 2, 2], attention=True, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = model_zoo.load_url(model_urls['resnet18-tiny'])
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet34atten(pretrained=False, progress=True, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], attention=True, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = model_zoo.load_url(model_urls['resnet34'])
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


def resnet34tiny(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Using tiny size model!")
    model = TinyResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = model_zoo.load_url(model_urls['resnet34-tiny'])
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


def resnet34attentiny(pretrained=False, progress=True, **kwargs):
    print("Using tiny size model!")
    model = TinyResNet(BasicBlock, [3, 4, 6, 3], attention=True, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = model_zoo.load_url(model_urls['resnet34-tiny'])
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet50atten(pretrained=False, progress=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], attention=True, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = model_zoo.load_url(model_urls['resnet50'])
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


def resnet50tiny(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Using tiny size model!")
    model = TinyResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = model_zoo.load_url(model_urls['resnet50-tiny'])
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


def resnet50attentiny(pretrained=False, progress=True, **kwargs):
    print("Using tiny size model!")
    model = TinyResNet(Bottleneck, [3, 4, 6, 3], attention=True, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = model_zoo.load_url(model_urls['resnet50-tiny'])
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
