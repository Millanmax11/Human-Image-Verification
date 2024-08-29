import os
import numpy as np
from django.shortcuts import render, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from PIL import Image
import random
import base64
from io import BytesIO

# Load CIFAR-10 dataset
(x_train, y_train), (_, _) = cifar10.load_data()

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'captcha_model.h5')
model = load_model(model_path)

def create_captcha_image():
    num_images = 8
    captcha_images = []
    captcha_labels = []
    for _ in range(num_images):
        index = random.randint(0, len(x_train) - 1)
        captcha_images.append(x_train[index])
        captcha_labels.append(y_train[index][0])

    captcha_image = Image.new('RGB', (32 * num_images, 32))
    for i, img in enumerate(captcha_images):
        img_pil = Image.fromarray(img.astype('uint8'))
        captcha_image.paste(img_pil, (i * 32, 0))

    target_index = random.randint(0, num_images - 1)
    captcha_label = class_names[captcha_labels[target_index]]

    # Convert the image to base64
    buffered = BytesIO()
    captcha_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str, captcha_label, captcha_labels

def captcha_view(request):
    if request.method == 'POST':
        selected_index = int(request.POST.get('captcha'))
        correct_label = request.session.get('correct_label')
        selected_label = request.session.get('labels')[selected_index]

        if class_names[selected_label] == correct_label:
            return render(request, 'captcha_app/success.html')
        else:
            return render(request, 'captcha_app/failure.html')

    captcha_image_base64, captcha_label, captcha_labels = create_captcha_image()

    request.session['correct_label'] = captcha_label
    request.session['labels'] = [int(label) for label in captcha_labels]  # Convert labels to int

    return render(request, 'captcha_app/captcha.html', {
        'captcha_label': captcha_label,
        'num_images': 8,
        'captcha_image_base64': captcha_image_base64
    })
