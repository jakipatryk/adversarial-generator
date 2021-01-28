# pylint: disable=import-error

""" Contains ALL_GENERATORS dictionary."""

from fast_gradient_sign_attack import FastGradientSignAttack
from deep_fool import DeepFool

ALL_GENERATORS = {
    "FastGradientSignAttack": FastGradientSignAttack,
    "DeepFool": DeepFool
}
