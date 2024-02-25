from __future__ import division, print_function
import os
# Keras
from keras.models import load_model
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import numpy as np
import pandas as pd
import tensorflow as tf

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# Load your trained model
model = load_model('./models/save_model.h5')
model.make_predict_function()          # Necessary
print('Model loaded. Start serving...')
output = []


def create_model():
  model = tf.keras.Sequential()

  model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=(3,), padding='same', activation='relu', input_shape = (187,1))) # notice that this 187 is the size of columns in dataset
  model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same', activation='relu'))
  model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=(5,), padding='same', activation='relu'))

  model.add(tf.keras.layers.MaxPool1D(pool_size=(3,), strides=2, padding='same'))
  model.add(tf.keras.layers.Dropout(0.5))

  model.add(tf.keras.layers.Flatten())

  model.add(tf.keras.layers.Dense(units = 256, activation='relu'))
  model.add(tf.keras.layers.Dense(units = 512, activation='relu'))

  model.add(tf.keras.layers.Dense(units = 5, activation='softmax'))
  return model

def model_predict(uploaded_files="./uploads/PVC2.csv",weights_dir="./models/save_model.h5"):
  model = create_model()
  model.load_weights(weights_dir)

  test_data = pd.read_csv(uploaded_files, header=None)
  test_df = pd.DataFrame(test_data)

  test_df.head()
  classes = []
  index = 0

  test_X = test_df.drop([187], axis=1)
  test_X = np.array(test_X).reshape(test_X.shape[0], test_X.shape[1], 1)

  test_pred_y = model.predict(test_X)

  inference_result = [np.where(i == np.max(i))[0][0] for i in test_pred_y]
  print(f'Batch inference results:{inference_result}')

  for i in range(len(inference_result)):
    if inference_result[i] ==0:
        inference_result[i]= 'Normal'
    if inference_result[i] ==1:
            inference_result[i]= 'APC'
    if inference_result[i] ==4:
                inference_result[i]='Paced Tachycardia'
    if inference_result[i] ==2:
        inference_result[i]='VPC'
    if inference_result[i] ==3:
        inference_result[i]='VF'

  return inference_result




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from  postrequest
        uploaded_files = []

        # Save the file to ./uploads
        print(uploaded_files)
        for f in request.files.getlist('file'):
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
            print(file_path)
            if file_path[-4:] == '.csv':
                uploaded_files.append(file_path)
                f.save(file_path)
        print("--------------------------------------------------------------------------------")
        print(uploaded_files)
        # Make prediction
        pred = model_predict(uploaded_files=uploaded_files[0])
        # Process your result for human
                    # Simple argmax
        #pred_class = decode_predictions(pred, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        result = str(pred)
        

        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 8088), app)
    http_server.serve_forever()
