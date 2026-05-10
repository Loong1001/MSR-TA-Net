import math
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale


def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out


def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))


def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff


def make_laplace_pyramid(img, level, channels):
    """
    Build Laplacian pyramid

    Args:
        img: Input image
        level: Number of pyramid levels
        channels: Number of channels

    Returns:
        List containing level difference maps + 1 residual, total level+1 elements
    """
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class DepthWiseConvBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            apply_norm: bool = True,
            conv_bn_act_pattern: bool = True,
            norm_cfg: dict = {'type': 'BN', 'momentum': 1e-2, 'eps': 1e-3}
    ) -> None:
        super(DepthWiseConvBlock, self).__init__()

        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False)

        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1)

        self.apply_norm = apply_norm
        if self.apply_norm:
            norm_type = norm_cfg.pop('type', 'BN')
            if norm_type == 'BN':
                self.bn = nn.BatchNorm2d(out_channels, **norm_cfg)

        self.apply_activation = conv_bn_act_pattern
        if self.apply_activation:
            self.swish = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.apply_norm:
            x = self.bn(x)
        if self.apply_activation:
            x = self.swish(x)

        return x


class ScaleAttention(nn.Module):
    def __init__(self, num_scales, in_channels, reduction=16):
        super(ScaleAttention, self).__init__()
        self.num_scales = num_scales
        self.in_channels = in_channels

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, num_scales, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mlp.to(self.device)

    def forward(self, features):
        pooled_features = [F.adaptive_avg_pool2d(feat, (1, 1)) for feat in features]
        concatenated = torch.cat(pooled_features, dim=1).to(self.device)
        flattened = concatenated.view(concatenated.size(0), -1)
        attention_weights = self.mlp(flattened)
        attention_weights = self.softmax(attention_weights)
        weighted_features = [feat * attention_weights[:, i].view(-1, 1, 1, 1) for i, feat in enumerate(features)]

        return weighted_features


class SEGA(nn.Module):
    """
    Spatial-Spectral Edge-Guided Attention Module

    Supports edge feature fusion with different Laplacian pyramid levels
    """

    def __init__(self, channel, laplace_levels=5):
        super(SEGA, self).__init__()
        self.laplace_levels = laplace_levels

        self.attention1 = nn.Sequential(
            nn.Conv2d(channel, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.attention2 = nn.Sequential(
            nn.Conv2d(channel, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.attention3 = nn.Sequential(
            nn.Conv2d(channel, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.attention4 = nn.Sequential(
            nn.Conv2d(channel, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

    def forward(self, fpn_output, edge_feature):
        """
        Args:
            fpn_output: [p2, p3, p4, p5] FPN features at four scales
            edge_feature: Laplacian pyramid output with length laplace_levels + 1

        Returns:
            Fused features at four scales
        """
        l2, l3, l4, l5 = fpn_output[0], fpn_output[1], fpn_output[2], fpn_output[3]

        # Select edge feature indices based on pyramid levels
        # Use the last 4 elements, corresponding to coarse-to-fine edge information
        edge_f1 = edge_feature[-4]
        edge_f2 = edge_feature[-3]
        edge_f3 = edge_feature[-2]
        edge_f4 = edge_feature[-1]

        # Resize edge features to match FPN feature dimensions
        edge_feature1 = F.interpolate(edge_f1, size=l2.shape[2:], mode='nearest')
        edge_feature2 = F.interpolate(edge_f2, size=l3.shape[2:], mode='nearest')
        edge_feature3 = F.interpolate(edge_f3, size=l4.shape[2:], mode='nearest')
        edge_feature4 = F.interpolate(edge_f4, size=l5.shape[2:], mode='nearest')

        # Edge feature fusion
        l2_2 = (self.attention1(l2 * edge_feature1)) * (l2 * edge_feature1) + l2
        l3_2 = (self.attention2(l3 * edge_feature2)) * (l3 * edge_feature2) + l3
        l4_2 = (self.attention3(l4 * edge_feature3)) * (l4 * edge_feature3) + l4
        l5_2 = (self.attention4(l5 * edge_feature4)) * (l5 * edge_feature4) + l5

        return l2_2, l3_2, l4_2, l5_2


class OneConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations,
                      bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class BiFPN(nn.Module):
    """
    Bidirectional Feature Pyramid Network with Laplacian Edge Enhancement

    Args:
        block: ResNet block type
        blocks_num: number of blocks in each stage
        num_classes: number of output classes
        include_top: whether to include classification head
        groups: number of groups for grouped convolution
        width_per_group: width per group
        laplace_levels: number of Laplacian pyramid levels (default: 5)
        sa_reduction: reduction ratio for ScaleAttention module (default: 16)
    """

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=8,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 laplace_levels=5,
                 sa_reduction=16):
        super(BiFPN, self).__init__()
        self.epsilon = 1e-4
        self.include_top = include_top
        self.in_channel = 64
        self.laplace_levels = laplace_levels
        self.sa_reduction = sa_reduction

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # Classification head
        if self.include_top:
            self.classifier = nn.Sequential(
                nn.Linear(256 * 8, 1024),
                nn.Dropout(p=0.5),
                nn.Linear(1024, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Bottom to up: feature map downsample module
        self.p3_down_sample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p4_down_sample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p5_down_sample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Fuse Conv Layers
        self.conv4_up = DepthWiseConvBlock(256, 256)
        self.conv3_up = DepthWiseConvBlock(256, 256)
        self.conv2_up = DepthWiseConvBlock(256, 256)

        self.conv3_down = DepthWiseConvBlock(256, 256)
        self.conv4_down = DepthWiseConvBlock(256, 256)
        self.conv5_down = DepthWiseConvBlock(256, 256)

        # weights
        self.p4_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()

        self.p3_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(
            torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(
            torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()

        self.swish = lambda x: x * torch.sigmoid(x)

        # Use configurable parameters
        self.attention_module = ScaleAttention(num_scales=4, in_channels=256*4, reduction=sa_reduction)
        self.SE1 = OneConv(512, 512, 1, 0, 1)
        self.SE2 = OneConv(512, 512, 1, 0, 1)
        self.SE3 = OneConv(512, 512, 1, 0, 1)
        self.SE4 = OneConv(512, 512, 1, 0, 1)

        self.softmax = nn.Softmax(dim=2)
        self.softmax_1 = nn.Sigmoid()

        # Use configurable Laplacian levels
        self.sega = SEGA(channel=256, laplace_levels=laplace_levels)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Generate edge features
        grayscale_img = rgb_to_grayscale(x)
        edge_feature = make_laplace_pyramid(grayscale_img, self.laplace_levels, 1)

        # Bottom-up
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down
        p5_in = self.toplayer(c5)
        p4_in = self.latlayer1(c4)
        p3_in = self.latlayer2(c3)
        p2_in = self.latlayer3(c2)

        weight_features = self.attention_module([p2_in, p3_in, p4_in, p5_in])
        p2_in_next, p3_in_next, p4_in_next, p5_in_next = weight_features[0], weight_features[1], weight_features[2], weight_features[3]

        res1, res2, res3, res4 = self.sega([p2_in_next, p3_in_next, p4_in_next, p5_in_next], edge_feature)

        # BiFPN fusion
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_up = self.conv4_up(
            weight[0] * p4_in_next +
            weight[1] * self.p4_upsample(p5_in_next))

        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_up = self.conv3_up(
            weight[0] * p3_in_next +
            weight[1] * self.p3_upsample(p4_up))

        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        p2_out = self.conv2_up(
            weight[0] * p2_in_next +
            weight[1] * self.p2_upsample(p3_up))

        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        p3_out = self.conv3_down(
            weight[0] * p3_in_next + weight[1] * p3_up +
            weight[2] * self.p3_down_sample(p2_out))

        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(
            weight[0] * p4_in_next + weight[1] * p4_up +
            weight[2] * self.p4_down_sample(p3_out))

        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(
            weight[0] * p5_in_next +
            weight[1] * self.p5_down_sample(p4_out))

        # Concatenate with SEGA features
        p2_out = torch.cat((p2_out, res1), dim=1)
        p3_out = torch.cat((p3_out, res2), dim=1)
        p4_out = torch.cat((p4_out, res3), dim=1)
        p5_out = torch.cat((p5_out, res4), dim=1)

        # Global average pooling
        p2_avg_gap = F.adaptive_avg_pool2d(p2_out, 1)
        p3_avg_gap = F.adaptive_avg_pool2d(p3_out, 1)
        p4_avg_gap = F.adaptive_avg_pool2d(p4_out, 1)
        p5_avg_gap = F.adaptive_avg_pool2d(p5_out, 1)

        p2_y_weight = self.softmax_1(self.SE1(p2_avg_gap))
        p3_y_weight = self.softmax_1(self.SE2(p3_avg_gap))
        p4_y_weight = self.softmax_1(self.SE3(p4_avg_gap))
        p5_y_weight = self.softmax_1(self.SE4(p5_avg_gap))

        weight = torch.cat([p2_y_weight, p3_y_weight, p4_y_weight, p5_y_weight], 2)
        weight = self.softmax(weight)
        p2_y_weight = torch.unsqueeze(weight[:, :, 0], 2)
        p3_y_weight = torch.unsqueeze(weight[:, :, 1], 2)
        p4_y_weight = torch.unsqueeze(weight[:, :, 2], 2)
        p5_y_weight = torch.unsqueeze(weight[:, :, 3], 2)

        p2_weighted = p2_y_weight * p2_out
        p3_weighted = p3_y_weight * p3_out
        p4_weighted = p4_y_weight * p4_out
        p5_weighted = p5_y_weight * p5_out

        p2_weighted_gap = F.adaptive_avg_pool2d(p2_weighted, 1)
        p3_weighted_gap = F.adaptive_avg_pool2d(p3_weighted, 1)
        p4_weighted_gap = F.adaptive_avg_pool2d(p4_weighted, 1)
        p5_weighted_gap = F.adaptive_avg_pool2d(p5_weighted, 1)

        concat_features = torch.cat((p2_weighted_gap, p3_weighted_gap, p4_weighted_gap, p5_weighted_gap), dim=1)
        flatten_features = concat_features.view(concat_features.size(0), -1)

        if self.include_top:
            x = self.classifier(flatten_features)
        else:
            x = flatten_features

        return x


def MSR_TA_Net(num_classes=8, include_top=True, laplace_levels=5, sa_reduction=16):
    """
    Create BiFPN model with ResNet50 backbone

    Args:
        num_classes: number of output classes
        include_top: whether to include classification head
        laplace_levels: number of Laplacian pyramid levels (recommended: 3, 4, 5)
        sa_reduction: reduction ratio for ScaleAttention module (recommended: 4, 8, 16, 32)

    Returns:
        BiFPN model instance
    """
    return BiFPN(Bottleneck, [3, 4, 6, 3],
                 num_classes=num_classes,
                 include_top=include_top,
                 laplace_levels=laplace_levels,
                 sa_reduction=sa_reduction)
