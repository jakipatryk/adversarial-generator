from django import forms
from all_models import ALL_MODELS


class ImageUploadForm(forms.Form):
    image_field = forms.ImageField()
    model_choice_field = forms.ChoiceField(
        choices=map(lambda x: (x, x), ALL_MODELS.keys()))
