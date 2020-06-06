from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField


class UploadDogImage(FlaskForm):
    picture = FileField('Upload Dog Picture', validators=[FileRequired(), FileAllowed(['png', 'jpg'])])
    submit = SubmitField('Upload')
