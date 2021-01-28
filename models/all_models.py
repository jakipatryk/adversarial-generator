# pylint: disable=import-error

""" Contains ALL_MODELS dictionary."""

from vgg16 import VGG16
from alexnet import ALEXNET
from densenet import DENSENET

ALL_MODELS = {
    "VGG16": VGG16,
    "AlexNet": ALEXNET,
    "DenseNet": DENSENET
}
