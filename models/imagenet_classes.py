"""Contains dictionary and list of all classes (1000) from ImageNet dataset."""

import json
import os

IMAGENET_CLASSES_DICT = json.load(
    open(os.path.join(os.path.dirname(__file__), "data/imagenet_class_index.json")))
IMAGENET_CLASSES = list(map(lambda x: x[1], IMAGENET_CLASSES_DICT.values()))
