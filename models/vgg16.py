# pylint: disable=import-error

"""Contains VGG16 Model."""

import torchvision
from torchvision import transforms as T
from model import Model
from imagenet_classes import IMAGENET_CLASSES

VGG16 = Model(torchvision.models.vgg16(pretrained=True),
              'VGG16 is a convolutional neural network introduced ' +
              'in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".' +
              ' Accepts 224x224 RGB images.',
              IMAGENET_CLASSES,
              lambda img: T.Compose([T.Resize(256), T.CenterCrop(224)])(img[:3]))
