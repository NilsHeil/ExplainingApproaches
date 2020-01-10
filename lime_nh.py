import os
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import shap_nh
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image

# Create the ImageExplainer
def CreateLimeImageExpl():
    explainer = lime_image.LimeImageExplainer()
    return explainer

# Hide color is the color for a superpixel turned OFF. Alternatively,
# if it is NONE, the superpixel will be replaced by the average of its pixels

def CreateExplanation (explainer, model,  explain, imagenumber,  num_samples, top_labels):

    explanation = explainer.explain_instance(explain[imagenumber], model.predict, top_labels=top_labels, num_samples= num_samples)
    return explanation

def plotting (explanation, positive_only,  num_features,  hide_rest,  imagenumber):

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[imagenumber], positive_only=positive_only, num_features=num_features, hide_rest=hide_rest)
    plt.imshow(mark_boundaries(temp, mask).astype("int64"))


