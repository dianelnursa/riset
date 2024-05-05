
from flask import Flask, render_template, request, jsonify, flash
from keras.models import load_model

from keras.utils import load_img, img_to_array
from keras_applications.mobilenet_v2 import preprocess_input
from keras_applications.mobilenet_v2 import decode_predictions
from keras_applications.mobilenet_v2 import MobileNetV2

import tensorflow as tf
from tensorflow import keras
from skimage import transform, io
import numpy as np
import os
from PIL import Image
from datetime import datetime
from keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)
app.secret_key="qwerty098765421"

# load model for prediction
modelxception = load_model("Xception-fructus-98.57.h5")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("beranda.html")

@app.route("/belajar", methods=['GET', 'POST'])
def belajar():
	return render_template("belajar.html")

	
#@app.route("/", methods=['GET', 'POST'])
#def main():v 
	#return render_template("cnn.html")
	
@app.route("/classification", methods = ['GET', 'POST'])
def classification():
	return render_template("classifications.html")

@app.route("/tentang", methods = ['GET', 'POST'])
def tentang():
	return render_template("developer.html")
@app.route('/submit', methods=['POST'])

def predict():
    files = request.files.getlist('file')
    filename = "temp_image.png"
    # errors = {}
    success = False
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            flash('Anda Belum Mengunggah File atau Ekstensi File Salah, \
                  Silahkan Ulangi Unggah File dan Pastikan Ekstensi File Sudah Sesuai Panduan di Atas!')
            return render_template("classifications.html")
        
    img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # convert image to RGB
    img = Image.open(img_url).convert('RGB')
    now = datetime.now()
    predict_image_path = 'static/uploads/' + now.strftime("%d%m%y-%H%M%S") + ".png"
    image_predict = predict_image_path
    img.convert('RGB').save(image_predict, format="png")
    img.close()

    # prepare image for prediction
    img = load_img(predict_image_path, target_size=(128, 128, 3))
    x = img_to_array(img)
    x = x/127.5-1 
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # predict
   
    prediction_array_xception = modelxception.predict(images)

    # prepare api response
    class_names = ['Amomi Fructus (Kapulaga)', 'Capsici Fructescentis Fructus (Cabai Rawit)', 'Cumini Fructus (Jinten Putih)', 'Piper Retrofractum Fructus (Cabai Jawa)', 'Piperis Albi Fructus (Lada Putih)', 'Piperis Nigri Fructus (Lada Hitam)', 'Tamarindus Indicia Fructus (Asam Jawa)', 'Tidak ada fructus yang terdeteksi \n gambar tersebut bukan fructus']	
    return render_template("classifications.html", img_path = predict_image_path, 
                        predictionxception = class_names[np.argmax(prediction_array_xception)],
                        confidenceexception = '{:2.0f}%'.format(100 * np.max(prediction_array_xception)),
                        )

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
