# pylint: disable=missing-docstring,no-name-in-module,import-error,invalid-name

from django.shortcuts import render
from torchvision import transforms as T
from PIL import Image
from generator.forms import ImageUploadForm
from generator.utils import img_to_base64
from all_models import ALL_MODELS
from all_generators import ALL_GENERATORS


def home_view(request):
    to_PIL = T.ToPILImage()
    to_tensor = T.ToTensor()
    context = {}
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            model_chosen = ALL_MODELS[form.cleaned_data.get(
                "model_choice_field")]
            context["model_info"] = model_chosen.description
            generator = ALL_GENERATORS[form.cleaned_data.get(
                "method_choice_field")](model_chosen)
            context["generator_info"] = generator.description
            img = Image.open(form.cleaned_data.get("image_field"))
            original_prediction = model_chosen.predict(to_tensor(img))
            context["original_prediction_class"] = original_prediction[0]
            context["original_prediction_confidence"] = 100 * \
                original_prediction[1]
            img_preprocessed = to_PIL(model_chosen.preprocessing_function(
                to_tensor(img)))
            context["original_img_str"] = img_to_base64(img_preprocessed)
            perturbated = generator.generate(to_tensor(img))
            perturbated_prediction = model_chosen.predict(
                perturbated, preprocessed=True)
            context["perturbated_prediction_class"] = perturbated_prediction[0]
            context["perturbated_prediction_confidence"] = 100 * \
                perturbated_prediction[1]
            perturbated_img = to_PIL(perturbated)
            context["perturbated_img_str"] = img_to_base64(perturbated_img)
    else:
        form = ImageUploadForm()
    context['form'] = form
    return render(request, "generator/home.html", context)
