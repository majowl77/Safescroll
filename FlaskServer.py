from flask import Flask, request, jsonify
from yt_dlp import YoutubeDL
import os
from flask_cors import CORS
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math   # for mathematical operations
from tensorflow.keras.preprocessing import image   # for preprocessing the images
from glob import glob
from tqdm import tqdm
import random
import pathlib
import tensorflow as tf
import imageio
# from tensorflow_docs.vis import embed
import keras

results = dict()

app = Flask(__name__)
CORS(app)

@app.route('/extract_url', methods=['POST'])
def extract_url():
    data = request.json
    url = data['url']
    print(url)

    if 'youtube.com/shorts' in url or 'youtu.be' in url:
            
            output = download_video(url)
            if output:
                return jsonify({'output': output})
            else:
                return jsonify({'message': 'Video download failed'})


def download_video(url):

    download_options = {
    'format': 'best',  # Choose the best available format
    'outtmpl': 'c:\\Users\\hadee\\Downloads\\googel chrom\\%(id)s.%(ext)s',  # Specify download path
    'quiet': True,     # Suppress console output
}
    try:
        with YoutubeDL(download_options) as ydl:
          info_dict = ydl.extract_info(url, download=True)  # Download and extract video information
          video_id = info_dict.get("id")  # Get video ID
          video_title = info_dict.get("title")  # Get video title
          path  = str(video_id) + '.mp4'
          if results.get(url):
            print('exists')
            return results[url]
          else:
            # If 'video_id' doesn't exist, create it
            output = get_output(path, video_title, video_id)
            results[url] = output
            print(results)
            return output
        
    except Exception as e:
            print('Error:', e)
            return False


#defining the 'inceptionV3' feature extraction model
#takes the video frame and return a 2048 representaion of the frame
def CNN_MODEL():

    #setting the image size
    img_size = (244, 126, 3)

    #defining the keras model
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=img_size,
    )


    #defining the input and the neccassery preprocessing to be done on it
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input(img_size)
    preprocessed = preprocess_input(inputs)

    #finalizing defining the model
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

#run the model and extract frame features
feature_extractor = CNN_MODEL()


#load video funcation takees the video and return a numpy array of resized frames from it.
def get_frames(path, max_frames=0, resize=(126, 244)):
    print('get_frames')
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            #ret is true if the video's frames are read successfully
            ret, frame = cap.read()
            if not ret:
                break
            #resize and append the frames to the frame array
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break

    finally:
        cap.release()
    return np.array(frames)

#loop over all videos, load frames then extract cnn features
#you don't have to run this function, there's pre-extracted save features you can load in the cell bellow
def get_cnn_features(path):

    MAX_SEQ_LENGTH = 20
    NUM_FEATURES = 2048

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # For each video.
    if(os.path.isfile(path)):

        # Gather all its frames and add a batch dimension.
        frames = get_frames(path)
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :], verbose=0
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features = np.array([temp_frame_features.squeeze()])
        frame_masks = np.array([temp_frame_mask.squeeze()])
        print(frame_features.shape)
    else:
        print('this file do not exist')
    return (frame_features, frame_masks)

rnn_model = keras.models.load_model('rnn_model_inception.h5')

def get_rnn_features(cnn_features):
  # Step 1: Load the saved model
  # Step 2: Create a new model that outputs the features from the second last layer
  feature_extractor_model = keras.Model(inputs=rnn_model.input, outputs=rnn_model.layers[-2].output)

  # Step 3: Use the feature extractor model to predict and get features
  rnn_features = feature_extractor_model.predict(cnn_features)

  return rnn_features

from transformers import TFBertModel, BertTokenizer

#load bert model and its tokenizer
bert_model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def get_bert_embeddings(title):
  input_ids = tokenizer(title, return_tensors="tf", padding=True, truncation=True, max_length=128)
  outputs = bert_model(input_ids)
  embeddings = outputs[1]

  return embeddings.numpy()

from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K

def F1_Score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall) / (precision + recall + K.epsilon())

    return f1_val

custom_objects = {'F1_Score': F1_Score}
fused_model = keras.models.load_model('fused_model_83.h5', custom_objects=custom_objects)

def get_precentage(rnn_features, text_embeddings):
  classifier = keras.Model(inputs=fused_model.input, outputs=fused_model.output)
  # Step 3: Use the feature extractor model to predict and get features
  output = classifier.predict([rnn_features, text_embeddings])
  return output.item()

def get_output(path, title, id):
  #load cnn
  if(os.path.isfile('cnn_features_' + str(id) + '.npy')):
    cnn_features = np.load('cnn_features_' + str(id) + '.npy')
  else:
    cnn_features = get_cnn_features(path)
    # np.save('cnn_features_' + str(id) + '.npy', np.array(cnn_features))

  #load rnn
  if(os.path.isfile('rnn_features_' + str(id) + '.npy')):
    rnn_features = np.load('rnn_features_' + str(id) + '.npy')
  else:
    rnn_features = get_rnn_features(cnn_features)
    np.save('rnn_features_' + str(id) + '.npy', rnn_features)

  #bert embeddibgs
  if(os.path.isfile(('text_embeddings_' + str(id) + '.npy'))):
    text_embeddings = np.load('text_embeddings_' + str(id) + '.npy')
  else:
    text_embeddings = get_bert_embeddings(title)
    np.save('text_embeddings_' + str(id) + '.npy', text_embeddings)

  #fused and get output
  output = get_precentage(rnn_features, text_embeddings)
  #print output
  print(f"{output:.8f}")

  return output


if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)