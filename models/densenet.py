"""Contains DENSENET Model."""

import torchvision
from torchvision import transforms as T
from model import Model
from imagenet_classes import IMAGENET_CLASSES

DENSENET = Model(torchvision.models.densenet161(pretrained=True),
                 'DenseNet is a convolutional neural network introduced ' +
                 'in the paper "Densely Connected Convolutional Networks".' +
                 ' Accepts 224x224 RGB images.',
                 IMAGENET_CLASSES,
                 lambda img: T.Compose([T.Resize(256), T.CenterCrop(224)])(img[:3]).unsqueeze(0))
