import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from flask_cors import CORS
import base64
import io
from PIL import Image
import datetime

# load model
model = model_from_json(open("fer_t3.json", "r").read())
# load weights
model.load_weights('fer_t3.h5')

face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = "JackJamesJohnManual"

CORS(app)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        base64_string = request.json["file"]

        x = str(datetime.date.today()) + ".jpg"
        image_64_decode = base64.b64decode(base64_string)
        # create a writable image and write the decoding result
        image_result = open(x, 'wb')
        image_result.write(image_64_decode)

        # check if the post request has the file part
        # if 'file' not in request.files:
        #     flash('No file part')
        #     return redirect(request.url)
        # file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        # if file.filename == '':
        #     flash('No selected file')
        #     return redirect(request.url)
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image2 = cv2.imread(x)
# im_arr is one-dim Numpy array

        gray_img = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(
            gray_img, 1.32, 5)
        print(faces_detected)
        for (x, y, w, h) in faces_detected:
            print(x)
            print(y)
            roi_gray = gray_img[
                y:y + w,
                x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('None', 'Fatigue', 'Glare', 'Normal', 'Squint')
            predicted_emotion = emotions[max_index]
            print(predicted_emotion)
            return {"emotion": predicted_emotion}
        return {"emotion": "No face detected"}
