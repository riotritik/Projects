from django.shortcuts import render, redirect
from .forms import UploadForm
from .models import BrainTumorDetector
from keras.models import load_model
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import cv2


def perform_brain_tumor_detection(image_path):
    # load the pre-trained model
    model = load_model(
        'C:\\Users\\ritikkumar\\Desktop\\final_cancer_V1\\brain_tumor_detection\\tumor_detection\\Brain_tumor_model.h5')

    # load the image and preprocess it
    img = Image.open(image_path)
    img_resized = img.resize((64, 64), resample=Image.BILINEAR)
    x = np.array(img_resized)
    x = np.expand_dims(x, axis=0)
    input_img = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # make predictions using the model
    result = model.predict(input_img)
    result = np.argmax(result, axis=1)

    # highlight the detected tumor on the image
    if result == 0:
        label = '😊Yay your MRI shows no signs of tumor😊'
    else:
        label = '😢 Sorry you have tumor.😢 Contact us for further treatment plans'
        draw = ImageDraw.Draw(img)
        width, height = img.size
        mask = np.zeros((height, width))
        mask[x[0][:, :, 0] > 0] = 1
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        draw.contour(contours[0].reshape(-1, 2), outline='red', width=3)
    return label, img


def upload_image(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            brain_tumor_detector = BrainTumorDetector(
                image=form.cleaned_data['image'])
            brain_tumor_detector.save()
            result, img = perform_brain_tumor_detection(
                brain_tumor_detector.image.path)
            brain_tumor_detector.result = result
            brain_tumor_detector.image_processed = img
            brain_tumor_detector.save()
            return redirect('view_result')
    else:
        form = UploadForm()
    return render(request, '../templates/upload.html', {'form': form})


def view_result(request):
    detectors = BrainTumorDetector.objects.all()    
    last_detector = detectors.last()
    context={'detectors':detectors, 'last_detector':last_detector}
    return render(request,'../templates/result.html',context)


def view_home(request):
    return render(request,'../templates/upload.html')

def view_dev(request):
    return render(request,'../templates/dev.html')
