
from flask import Flask, render_template, request, jsonify
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

# load model for prediction
modelnasnet = load_model("NASNET-fructus.h5")
modelvgg = load_model("VGG16-fructus.h5")
modelxception = load_model("Xception-fructus.h5")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	developer_info = {'name': 'Dian Budi Elnursa', 'photo_url': '/static/pestectionadmin/images/photo.jpg','linkedin': 'https://s.id/LinkedIn-dian',
    'youtube': 'https://s.id/YouTube-dian'}
	return render_template("developer.html", developer_info=developer_info)
	
#@app.route("/", methods=['GET', 'POST'])
#def main():
	#return render_template("cnn.html")
	
@app.route("/classification", methods = ['GET', 'POST'])
def classification():
	return render_template("classifications.html")


@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    files = request.files.getlist('file')
    filename = "temp_image.png"
    errors = {}
    success = False
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors["message"] = 'File type of {} is not allowed'.format(file.filename)

    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp
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
    prediction_array_nasnet = modelnasnet.predict(images)
    prediction_array_vgg = modelvgg.predict(images)
    prediction_array_xception = modelxception.predict(images)

    # prepare api response
    class_names = ['Amomi Fructus (Kapulaga)', 'Capsici Fructescentis Fructus (Cabai Rawit)', 'Cumini Fructus (Jinten Putih)', 'Piper Retrofractum Fructus (Cabai Jawa)', 'Piperis Albi Fructus (Lada Putih)', 'Piperis Nigri Fructus (Lada Hitam)', 'Tamarindus Indicia Fructus (Asam Jawa)']	
    return render_template("classifications.html", img_path = predict_image_path, 
                        #predictionnasnet = class_names[np.argmax(prediction_array_nasnet)],
                        #confidencenasnet = '{:2.0f}%'.format(100 * np.max(prediction_array_nasnet)),
                        #predictionvgg = class_names[np.argmax(prediction_array_vgg)],
                        #confidencvgg = '{:2.0f}%'.format(100 * np.max(prediction_array_vgg)),
                        predictionxception = class_names[np.argmax(prediction_array_xception)],
                        confidenceexception = '{:2.0f}%'.format(100 * np.max(prediction_array_xception)),
                        )

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
