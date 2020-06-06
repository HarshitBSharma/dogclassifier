from flask import render_template, url_for, request, redirect
from dogclassifier.forms import UploadDogImage
import os
from dogclassifier import app
from dogclassifier.prediction_logic import DogPredict
import secrets
from PIL import Image


def clean_directory():
    path = os.path.join(app.root_path, 'static', 'images')
    for filename in os.listdir(path):
        if "logo" in filename:
            continue
        else:
            os.remove(os.path.join(path, filename))


def img_resize(img_path):
    max_width = 800
    max_height = 500
    img = Image.open(img_path)
    rescaled_path = ""
    img.thumbnail((max_width, max_height), Image.ANTIALIAS)
    for filename in os.listdir(os.path.join(app.root_path, 'static', 'images')):
        if "logo" in filename:
            continue
        else:
            f_name, f_ext = os.path.splitext(filename)
            rescaled_path += f_name+'rescaled'+f_ext
            img.save(os.path.join(app.root_path, 'static', 'images', rescaled_path))
    return rescaled_path


def save_img(form_picture):
    _, f_ext = os.path.splitext(form_picture.filename)
    rand_num = secrets.token_hex(4)
    picture_fn = rand_num + f_ext
    picture_path = os.path.join(app.root_path, 'static', 'images', picture_fn)
    form_picture.save(picture_path)
    return picture_path


@app.route("/", methods=['GET', 'POST'])
def home():
    clean_directory()
    form = UploadDogImage()
    breed = ""
    f_name = ""
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_img(form.picture.data)
            picture_path = picture_file
            dog = DogPredict(picture_path)
            breed += dog.predict_dog_breed()
            f_name += img_resize(picture_path)
    return render_template('index.html', form=form, breed=breed, f_name=f_name)

