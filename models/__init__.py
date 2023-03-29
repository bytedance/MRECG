import torch
from collections import OrderedDict
from .regnet import (  # noqa: F401
    regnetx_200m, regnetx_400m, regnetx_600m, regnetx_800m,
    regnetx_1600m, regnetx_3200m, regnetx_4000m, regnetx_6400m,
    regnety_200m, regnety_400m, regnety_600m, regnety_800m,
    regnety_1600m, regnety_3200m, regnety_4000m, regnety_6400m,
)
from .resnet import (  # noqa: F401
    resnet18, resnet26, resnet34, resnet50,
    resnet101, resnet152, resnet_custom
)
from .mobilenet_v2 import mobilenet_v2


def load_model(config):
    model = globals()[config['type']](**config['kwargs'])
    pretrained_dict = torch.load(config.path, map_location='cpu')

    if "mobilenetv2.pth.tar" in config['path']:
        model.load_state_dict(pretrained_dict['model'])
    elif "model_best.pth.tar" in config['path']:
        model.load_state_dict({k.replace('module.', ''): v for k, v in
                       pretrained_dict['state_dict'].items()})
    elif config['type'] == "resnet18":
        model.load_state_dict(pretrained_dict)
    elif config['type'] != "automl_mobile":
        new_state_dict = OrderedDict()
        model_dict = model.state_dict()
        keys = []
        for k, v in model_dict.items():
            keys.append(k)
        for k1,k2 in zip(keys, pretrained_dict):
            new_state_dict[k1] = pretrained_dict[k2]
        print(f'load pretrained checkpoint from: {config.path}')
        model.load_state_dict(new_state_dict)
    return model
