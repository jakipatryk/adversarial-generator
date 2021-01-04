"""Contains dictionary and list of all classes (1000) from ImageNet dataset."""

import json

IMAGENET_CLASSES_DICT = json.load(open("data/imagenet_class_index.json"))
IMAGENET_CLASSES = list(map(lambda x: x[1], IMAGENET_CLASSES_DICT.values()))
