from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from PIL import Image as pil_image
import random

# Keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model , load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


Model= load_model('models/model_inception.h5')     

lesion_classes_dict = {
    
    0 : 'Bacterial_spot',
    1 : 'Early_blight ',
    2 : 'Late_blight',
    3 : 'Leaf_Mold',
    4:'Septoria_leaf_spot',
    5:'Spider_mites Two-spotted_spider_mite',
    6:'Target_Spot',
    7:'Tomato_Yellow_Leaf_Curl_Virus',
    8:'Tomato_mosaic_virus',
    9:'Healthy'
    

}



def model_predict(img_path, Model):
    img = image.load_img(img_path, target_size=(224, 224,3))
  
    #img = np.asarray(pil_image.open('img').resize((120,90)))
    #x = np.asarray(img.tolist())

    x = image.img_to_array(img)
    x=x/255.
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = Model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    sub = ""
    sub += lesion_classes_dict[random.randint(0, 9)]
    sub += ","
    sub += lesion_classes_dict[random.randint(0, 9)]
    sub += ","
    sub += lesion_classes_dict[random.randint(0, 9)]
    return render_template('index.html', sub = sub)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path , Model)

        # Process your result for human
        

        pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   
        pr = lesion_classes_dict[pred_class[0]]
        result =str(pr)  
        print(result)       
        return result
    print("Error")
    return None


if __name__ == '__main__':
    app.run(debug=True)

