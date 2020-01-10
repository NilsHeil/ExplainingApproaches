import innvestigate_nh
import innvestigate_nh.utils
import keras.applications.vgg16 as vgg16
import shap_nh
import os
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import innvestigate_nh.utils as iutils


# Strip softmax layer
model = innvestigate_nh.utils.model_wo_softmax(model)

# Creating an analyzer
analyzer = innvestigate_nh.create_analyzer("deep_taylor", model)

# Applying the analyzer
x = preprocess(to_explain[1])
analysis = analyzer.analyze(x[None])

# Displaying the gradient
# Aggregate along color channels and normalize to [-1, 1]
analysis = analysis.sum(axis=np.argmax(np.asarray(analysis.shape) == 3))
analysis /= np.max(np.abs(analysis))
# Plot
plt.imshow(analysis[0], cmap="seismic", clim=(-1, 1))
plt.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest')
plt.show()
