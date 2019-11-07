import torch.nn as nn
import pretrainedmodels
from torchvision.models import resnet18, resnet34
from efficientnet_pytorch import EfficientNet
from efficientnet_gem.model import EfficientNet as EfficientNetGem
from segmentation_models_pytorch import Unet, Linknet, FPN, PSPNet


def get_model(config):
    if config.TASK == 'seg':
        model_architecture = config.MODEL.ARCHITECTURE
        model_encoder = config.MODEL.ENCODER

        # activation은 eval 모드일 때 적용해 주는 거라 train 때에는 직접 sigmoid 쳐야한다.
        model = globals().get(model_architecture)(model_encoder, encoder_weights='imagenet', classes=4, activation='sigmoid')

        print('architecture:', model_architecture, 'encoder:', model_encoder)

    elif config.TASK == 'cls':
        model_name = config.MODEL.NAME

        if model_name.startswith('resnet'):
            model = globals().get(model_name)(pretrained=True)
            model.avgpool = nn.AdaptiveAvgPool2d(1)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 4)

        elif model_name.startswith('efficient'):
            model = EfficientNet.from_pretrained(model_name, num_classes=4)

            if config.MODEL.ADD_FC:
                in_features = model._fc.in_features
                new_fc = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.BatchNorm1d(256, eps=0.001, momentum=0.010000000000000009, affine=True,
                                   track_running_stats=True),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 4))

                model._fc = new_fc
                print('new fc added')

        else:
            model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
            model.avg_pool = nn.AdaptiveAvgPool2d(1)
            in_features = model.last_linear.in_features
            model.last_linear = nn.Linear(in_features, 4)

            print('model name:', model_name)

    if config.PARALLEL:
        model = nn.DataParallel(model)

    return model
