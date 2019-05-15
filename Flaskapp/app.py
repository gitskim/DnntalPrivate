#Usage: python app.py
import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64

import predict_classifier_1 as pred

img_width, img_height = 224, 224

model_path_classifier = './models/model_classifier.h5'
model = load_model(model_path_classifier)

graph = tf.get_default_graph()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

# def predict(file):
#     with graph.as_default():
#         result = pred.predict(file)
#         result = result[0][1]
#         if result < 0.5:
#             print("Label: no cavity")
#         elif result > 0.5:
#             print("Label: cavity")
#         return result

def predict(file):
    with graph.as_default():
        image = pred.post_processing(file)
        image.save('uploads/prediction23.jpg')

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # result = predict(file_path)
            # if result < 0.5:
            #     label = 'No cavity'
            # elif result > 0.5:
            #     label = 'Cavity'			
            # print(result)
            # print(file_path)
            predict(file_path)
            label='Go to treat these cavities please.'
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            # return render_template('template.html', label=label, imagesource='../uploads/' + filename)
            return render_template('template.html', label=label, imagesource1='../uploads/prediction23.jpg')

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run()