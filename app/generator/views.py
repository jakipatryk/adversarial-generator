from django.shortcuts import render
from .forms import ImageUploadForm
from torchvision import transforms as T
from PIL import Image
from all_models import ALL_MODELS
from fast_gradient_sign_attack import FastGradientSignAttack
from generator.utils import img_to_base64


def home_view(request):
    to_PIL = T.ToPILImage()
    to_tensor = T.ToTensor()
    context = {}
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            model_chosen = form.cleaned_data.get("model_choice_field")
            model_chosen = ALL_MODELS[model_chosen]
            fgsa = FastGradientSignAttack(model_chosen)
            img_form = form.cleaned_data.get("image_field")
            img = Image.open(img_form)
            original_prediction = model_chosen.predict(to_tensor(img))
            context["original_prediction_class"] = original_prediction[0]
            context["original_prediction_confidence"] = original_prediction[1]
            img_preprocessed = model_chosen.preprocessing_function(
                to_tensor(img))
            img_preprocessed = to_PIL(img_preprocessed)
            context["original_img_str"] = img_to_base64(img_preprocessed)
            perturbated = fgsa.generate(to_tensor(img))
            perturbated_prediction = model_chosen.predict(
                perturbated, preprocessed=True)
            context["perturbated_prediction_class"] = perturbated_prediction[0]
            context["perturbated_prediction_confidence"] = perturbated_prediction[1]
            perturbated_img = to_PIL(perturbated)
            context["perturbated_img_str"] = img_to_base64(perturbated_img)
    else:
        form = ImageUploadForm()
    context['form'] = form
    return render(request, "generator/home.html", context)
