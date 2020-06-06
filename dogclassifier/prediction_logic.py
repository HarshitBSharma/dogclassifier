from fastai.vision import cnn_learner, ImageDataBunch, open_image, load_learner, Learner, get_transforms, imagenet_stats, \
    models
from dogclassifier import app
from PIL import Image
import os

class DogPredict:
    def __init__(self, picture_file):
        self.classes = ['Afghan_hound', 'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier',
                        'Appenzeller', 'Australian_terrier', 'Bedlington_terrier', 'Bernese_mountain_dog',
                        'Blenheim_spaniel', 'Border_collie', 'Border_terrier', 'Boston_bull', 'Bouvier_des_Flandres',
                        'Brabancon_griffon', 'Brittany_spaniel', 'Cardigan', 'Chesapeake_Bay_retriever', 'Chihuahua',
                        'Dandie_Dinmont', 'Doberman', 'English_foxhound', 'English_setter', 'English_springer', 'EntleBucher',
                        'Eskimo_dog', 'French_bulldog', 'German_shepherd', 'German_short-haired_pointer', 'Gordon_setter',
                        'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog', 'Ibizan_hound', 'Irish_setter',
                        'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel',
                        'Kerry_blue_terrier', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'Maltese_dog',
                        'Mexican_hairless', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound', 'Norwich_terrier',
                        'Old_English_sheepdog', 'Pekinese', 'Pembroke', 'Pomeranian', 'Rhodesian_ridgeback', 'Rottweiler',
                        'Saint_Bernard', 'Saluki', 'Samoyed', 'Scotch_terrier', 'Scottish_deerhound', 'Sealyham_terrier',
                        'Shetland_sheepdog', 'Shih-Tzu', 'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel',
                        'Tibetan_mastiff', 'Tibetan_terrier', 'Walker_hound', 'Weimaraner', 'Welsh_springer_spaniel',
                        'West_Highland_white_terrier', 'Yorkshire_terrier', 'affenpinscher', 'basenji', 'basset', 'beagle',
                        'black-and', 'bloodhound', 'bluetick', 'borzoi', 'boxer', 'briard', 'bull_mastiff', 'cairn', 'chow',
                        'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dhole', 'dingo', 'flat-coated_retriever',
                        'giant_schnauzer', 'golden_retriever', 'groenendael', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'malamute',
                        'malinois', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'otterhound', 'papillon', 'pug',
                        'redbone', 'schipperke', 'silky_terrier', 'soft-coated_wheaten_terrier', 'standard_poodle', 'standard_schnauzer',
                        'toy_poodle', 'toy_terrier', 'vizsla', 'whippet', 'wire-haired_fox_terrier']
        f_name, f_ext = os.path.splitext(picture_file)
        img_path = os.path.join(app.root_path, 'static', 'images', f_name+f_ext)
        self.img = open_image(img_path)
        self.data = ImageDataBunch.single_from_classes("./", self.classes, ds_tfms=get_transforms(),
                                                       size=224).normalize(imagenet_stats)
        self.learner = cnn_learner(self.data, models.resnet50)
        self.learner.load('stage-2-rerun')

    def predict_dog_breed(self):
        pred_class, pred_idx, outputs = self.learner.predict(self.img)
        words = (self.data.classes[pred_idx]).split("_")
        breed = " ".join(words).capitalize()
        return f"We think it's a {breed} !"


