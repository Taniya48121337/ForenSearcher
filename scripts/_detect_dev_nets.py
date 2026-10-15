"""开发探针用的两个截断 ResNet-50（只到 layer2）：NPR（Tan et al., CVPR 2024, github.com/chuangchuangtan/NPR-DeepfakeDetection）
与 SAFE（Li et al., KDD 2025, github.com/Ouxiang-Li/SAFE），结构照抄官方 networks/resnet.py，只保留推理需要的部分。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _Trunk(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False), nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def trunk(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer2(self.layer1(x))
        return self.fc1(self.avgpool(x).flatten(1))


class NPRNet(_Trunk):
    def __init__(self):
        super().__init__(num_classes=1)

    @staticmethod
    def _interp(img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode="nearest", recompute_scale_factor=True),
                             scale_factor=1 / factor, mode="nearest", recompute_scale_factor=True)

    def forward(self, x):
        npr = x - self._interp(x, 0.5)
        return self.trunk(npr * 2.0 / 3.0)


class SAFENet(_Trunk):
    def __init__(self):
        super().__init__(num_classes=2)
        self._dwt = None

    def forward(self, x):
        _shim_pkg_resources()
        from pytorch_wavelets import DWTForward

        if self._dwt is None:
            self._dwt = DWTForward(J=1, mode="symmetric", wave="bior1.3").to(x.device)
        _, yh = self._dwt(x)
        hh = F.interpolate(yh[0][:, :, 2, :, :], size=(x.shape[-2], x.shape[-1]), mode="bilinear", antialias=True)
        return self.trunk(hh)


def _shim_pkg_resources() -> None:
    """pytorch_wavelets 1.3.0 在导入时用 pkg_resources.resource_stream 读系数表；新版 setuptools 去掉了 pkg_resources，补一个最小替身。"""

    import sys

    try:
        import pkg_resources  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass
    import importlib
    import os
    import types

    def resource_stream(package, resource):
        module = importlib.import_module(package)
        return open(os.path.join(os.path.dirname(module.__file__), resource), "rb")

    sys.modules["pkg_resources"] = types.SimpleNamespace(resource_stream=resource_stream)


def npr_resnet50() -> NPRNet:
    return NPRNet()


def safe_resnet50() -> SAFENet:
    return SAFENet()
