import torch as T
import torchvision.models as tvm
from efficientnet_pytorch import EfficientNet
from .effnetv2 import (effnetv2_s)  # , effnetv2_m, effnetv2_l, effnetv2_xl, EffNetV2)


def get_effnetv2(pretraining, in_channels, out_channels):
    assert pretraining == 'untrained', 'error: no pre-trained weights currently available for EfficientNetV2'
    mdl = effnetv2_s(num_classes=out_channels)
    _modify_conv2d(conv2d=mdl.features[0][0], in_channels=in_channels)
    return mdl


def get_resnet(name, pretraining, in_channels, out_channels):
    assert pretraining in {'untrained', 'imagenet'}
    mdl = getattr(tvm, name)(
        pretrained=True if pretraining == 'imagenet' else False)
    _modify_conv2d(mdl.conv1, in_channels)
    mdl.fc = T.nn.Linear(
        in_features=mdl.fc.in_features, out_features=out_channels, bias=True)
    return mdl


def get_efficientnetv1(name, pretraining, in_channels, out_channels):
    assert pretraining in {'untrained', 'imagenet', 'imagenetadv'}
    if pretraining == 'imagenetadv':
        mdl = EfficientNet.from_pretrained(
            name, advprop=True, in_channels=in_channels, num_classes=out_channels)
    elif pretraining == 'imagenet':
        mdl = EfficientNet.from_pretrained(
            name, in_channels=in_channels, num_classes=out_channels)
    else:
        mdl = EfficientNet.from_name(
            name, in_channels=in_channels, num_classes=out_channels)
    return mdl


def _modify_conv2d(conv2d:T.nn.Module, in_channels:int, ):
    """Inplace modify conv2d layer to ensure has in_channels"""
    if in_channels != conv2d.in_channels:
        conv2d.in_channels = in_channels
        if in_channels < conv2d.in_channels:
            conv2d.weight = T.nn.Parameter(conv2d.weight.data[:,[1],:,:])
        else:
            raise NotImplementedError('code for this written but not tested')
            O,_,H,W = conv2d.weight.shape
            tmp = T.empty(
                (O,in_channels,H,W),
                dtype=conv2d.weight.dtype, device=conv2d.weight.device)
            T.nn.init.kaiming_uniform_(tmp)
            conv2d.weight = T.nn.Parameter(tmp)
        assert conv2d.bias is None, 'bias not implemented yet'
