from typing import Union, List, Dict, Any, cast
from collections import OrderedDict
import logging

import torch
import torch.nn as nn
import numpy as np


from utils.util import getNetImageSizeAndNumFeats

logger = logging.getLogger(__name__)

__all__ = [
    "VGG",
    "vgg19_bn",
]


model_urls = {
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
        get_perceptual_feats: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = self._make_classifier(num_classes, init_weights, dropout)

        if init_weights:
            self._initialize_weights()
        self.get_perceptual_feats = get_perceptual_feats
        self.Out = OrderedDict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)

        if self.get_perceptual_feats:
            Out = []
            for k, v in self.Out.items():
                Out.append(v)
            Out.append(out)
            return out, Out
        else:
            return out

    def _make_classifier(self, num_classes, init_weights, dropout):
        classify = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 512, 7, 7, 4096
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        return classify

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _get_hook(self, layer_num):
        Out = self.Out

        def myhook(module, _input, _out):
            Out[layer_num] = _out

        self.features[layer_num].register_forward_hook(myhook)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ]
}


def _vgg(
    arch: str,
    cfg: str,
    batch_norm: bool,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict)
    return model


def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)


def VGG19(
    net_type="vgg19", get_perceptual_feats=True, num_classes=10, image_size=32, **kwargs
):
    logger.info(
        "buildVGGNet: {} {} {} {}".format(
            net_type, get_perceptual_feats, num_classes, image_size
        )
    )

    net = vgg19_bn(
        pretrained=True,
        progress=True,
        get_perceptual_feats=get_perceptual_feats,
        num_classes=num_classes,
        **kwargs
    )

    logger.info("# net : {} {}".format(len(net.features), net))
    if get_perceptual_feats:
        for idx in range(len(net.features)):
            if str(net.features[idx])[0:4] == "ReLU":
                logger.info(
                    "# registering hook module ({}), {}".format(
                        idx, str(net.features[idx])
                    )
                )
                net._get_hook(idx)
        ImgSizeL, numFeatsL = getNetImageSizeAndNumFeats(net, image_size=image_size)

        net.ImageSizePerLayer = np.array(ImgSizeL)
        net.numberOfFeaturesPerLayer = np.array(numFeatsL)
    return net
