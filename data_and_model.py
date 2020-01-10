# first load all important imports for transforming the data
import os
import keras
import keras.applications.vgg16 as vgg16
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.vgg16 import preprocess_input
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import json
import shap

# Get the VGG16 model, with weights pre-trained on ImageNet.
# The default input size for this model is 224x224.
import lime_nh

model, preprocess = vgg16.VGG16(weights='imagenet', include_top=True), vgg16.preprocess_input

# Load the data from ImageNet and store it in X. Load the images you want to explain into "to_explain"
X,y = shap.datasets.imagenet50()
to_explain = X[[39,41]]
imagenumber = 0

# Get an overview of how the image you want to explain looks like
plt.imshow(to_explain[imagenumber].astype("int64"))

# Show the top 5 predicted classes for the image
preds = model.predict(to_explain)
for x in decode_predictions(preds)[imagenumber]:
    print(x)

# Print the indexes of the top 5 predicted classes
print(model.predict(to_explain).argsort()[0, -5:][::-1])


# Create Lime Explainer by calling lime_nh.CreateLimeImageExpl
LimeExpl = lime_nh.CreateLimeImageExpl()

# Define arguments for your explanation
# int num_samples = Number of samples = amount of how many samples you want the explainer to be trained (suggested: 100)
# int top_labels = Number of top_labels you want to explain (suggested: 5)
num_samples = 100
top_labels = 5

# Create Lime explanation by calling the function lime_nh.CreateExplanation
explanation = lime_nh.CreateExplanation(LimeExpl, model , to_explain, imagenumber, num_samples, top_labels)

# Define arguments for your plot
# boolean positive_only = Define whether you only want to see image parts that positiveley influence the prediciton
# int num_features = Define how many explaining features you want to show
# boolean hide_rest = Define wheter you want to black the rest of the image
positive_only = True
num_features = 5
hide_rest = True
min_weight= 0.0

# Create plot
lime_nh.plotting(explanation, positive_only, num_features, hide_rest, imagenumber, min_weight)

