# pylint: disable=no-member,invalid-name

"""
Contains adversarial generator that uses DeepFool method.

(https://arxiv.org/abs/1511.04599)
"""

import torch
import torch.autograd.functional as agf
import torch.linalg as la
from adversarial_generator import AdversarialGenerator
from model import Model
from utils import clipped_renormalize


class DeepFool(AdversarialGenerator):
    """
    Adversarial examples generator that estimates the smallest perturbation
    needed to change the predicted class, in a greedy fashion.

    Parameters:
    - model (Model): a model that will be used to generate adversarial images for
    - max_iter (int) [optional]: maximal number of iterations when generating
        adversarial example, small numbers do not guarantee that generated
        'adversarial example' will be predicted into different class
        than the original image
    """

    def __init__(self, model: Model, max_iter=10):
        super(DeepFool, self).__init__(
            model,
            "Approximates in a greedy fashion the smallest perturbation needed to change " +
            "the predicted class, and then adds that perturbation to the image. ")
        self.max_iter = max_iter

    def generate_change_tensor(self, preprocessed_image: torch.Tensor) -> torch.Tensor:
        """
        Generates change tensor by iteratively going towards linearized minimal distance
        to hyperplane that is approximation for the decision boundary.

        Arguments:
        - preprocessed_image (torch.Tensor): normalized and preprocessed
            image with shape [channels, height, width]

        Returns:
        torch.Tensor: tensor to be added to the image to change prediction
        """
        self.model.classifier.eval()
        with torch.no_grad():
            original_prediction = self.model.classifier(
                preprocessed_image.unsqueeze(0))[0]
            original_prediction_class = torch.argmax(original_prediction)
            perturbated_img = preprocessed_image.clone().detach()
            perturbation = torch.zeros_like(perturbated_img)
        for _ in range(self.max_iter):
            with torch.no_grad():
                perturbated_img = clipped_renormalize(perturbated_img)
                predicted = self.model.classifier(
                    perturbated_img.unsqueeze(0))[0]
                predicted_class = torch.argmax(predicted)
                if predicted_class != original_prediction_class:
                    return perturbation
            jacobian = agf.jacobian(
                lambda x: self.model.classifier(x.unsqueeze(0))[0], perturbated_img)
            with torch.no_grad():
                w = torch.cat(
                    [
                        jacobian[:predicted_class],
                        jacobian[(predicted_class + 1):]
                    ]
                ) - jacobian[predicted_class]
                f = torch.cat(
                    [
                        predicted[:predicted_class],
                        predicted[(predicted_class + 1):]
                    ]
                ) - predicted[predicted_class]
                l = torch.argmin(
                    torch.abs(f) /
                    la.norm(torch.flatten(w, start_dim=1), dim=1)
                )
                r = (torch.abs(f[l]) / la.norm(torch.flatten(w[l]))**2) * w[l]
                perturbation = perturbation + 1.1 * r
                perturbated_img = perturbated_img + 1.1 * r
        return perturbation
