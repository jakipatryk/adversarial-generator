# pylint: disable=no-member,not-callable

"""Contains adversarial generator that uses Fast Gradient Sign Attack method."""

import torch
from adversarial_generator import AdversarialGenerator
from model import Model


class FastGradientSignAttack(AdversarialGenerator):
    """
    Adversarial examples generator that uses gradient of cost w.r.t. image.

    Parameters:
    - model (Model): a model that will be used to generate adversarial images for
    """

    def __init__(self, model: Model):
        super(FastGradientSignAttack, self).__init__(
            model,
            "Calculates gradient of loss function w.r.t. image " +
            "and then perturbates the image by +epsilon*grad_sign, " +
            "where epsilon is as small as possible but still changes predicted class.")

    def generate_change_tensor(self, preprocessed_image: torch.Tensor) -> torch.Tensor:
        """
        Generates change tensor by calculating a gradient of loss function w.r.t.
        preprocessed_image, and then using signs of this gradient finds small
        epsilon>0 such that image + (epsilon * grad_sign) is classified
        by the model into different class than preprocessed_image.

        Arguments:
        - preprocessed_image (torch.Tensor): normalized and preprocessed
            image with shape [channels, height, width]

        Returns:
        torch.Tensor: tensor to be added to the image to change prediction
        """
        self.model.classifier.eval()
        preprocessed_image.requires_grad = True
        predicted = self.model.classifier(preprocessed_image.unsqueeze(0))
        predicted_class_index = torch.argmax(predicted[0])
        cost = torch.nn.CrossEntropyLoss()(
            predicted, torch.tensor([predicted_class_index]))
        grad = torch.autograd.grad(
            cost, preprocessed_image, create_graph=False)[0]
        grad_sign = grad.sign()
        epsilon = self._find_epsilon(
            preprocessed_image, grad_sign, predicted_class_index)
        preprocessed_image.requires_grad = False
        return epsilon * grad_sign

    def _find_epsilon(
            self,
            preprocessed_image: torch.Tensor,
            grad_sign: torch.Tensor,
            predicted_class_index,
            max_iter=8) -> float:
        """
        Uses a variant of binary search to find small epsilon for generate_change_tensor.

        Arguments:
        - preprocessed_image (torch.Tensor): normalized and preprocessed
            image with shape [channels, height, width]
        - grad_sign (torch.Tensor): contains signs of gradients of each pixel,
            has the same shape as preprocessed_image
        - predicted_class_index: index of class that the model has predicted for
            original preprocessed_image
        - max_iter (int) [optional]: how many iterations of binary search to perform

        Returns:
        float: small epsilon that changes prediction
        """
        left = 0.0
        right = 1.0
        with torch.no_grad():
            for _ in range(max_iter):
                mid = (left + right)/2
                perturbated_image = preprocessed_image + mid * grad_sign
                perturbated_pred = self.model.classifier(
                    perturbated_image.unsqueeze(0))
                perturbated_pred_class_idx = torch.argmax(perturbated_pred[0])
                if perturbated_pred_class_idx == predicted_class_index:
                    left = mid
                else:
                    right = mid
        return right
