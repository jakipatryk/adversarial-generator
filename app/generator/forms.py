from django import forms
from all_models import ALL_MODELS
from all_generators import ALL_GENERATORS


class ImageUploadForm(forms.Form):
    image_field = forms.ImageField(label="Image")
    model_choice_field = forms.ChoiceField(
        choices=map(lambda x: (x, x), ALL_MODELS.keys()),
        label="Model"
    )
    method_choice_field = forms.ChoiceField(
        choices=map(lambda x: (x, x), ALL_GENERATORS.keys()),
        label="Method"
    )
