# pylint: disable=import-error

"""Contains ALEXNET Model."""

import torchvision
from torchvision import transforms as T
from model import Model
from imagenet_classes import IMAGENET_CLASSES

ALEXNET = Model(torchvision.models.alexnet(pretrained=True),
                'AlexNet is a convolutional neural network introduced ' +
                'in the paper "One weird trick for parallelizing convolutional neural networks".' +
                ' Accepts 256x256 RGB images.',
                IMAGENET_CLASSES,
                lambda img: T.Compose([T.Resize(276), T.CenterCrop(256)])(img[:3]))
