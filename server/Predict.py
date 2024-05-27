#import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import img_to_array, load_img
# from skimage.io import imread, imshow
# from skimage.transform import resize
from keras.models import load_model
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m ", "--model", required=True)
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

image = load_img(args["image"], target_size=(128,128,3))
#image = resize(image, (224,224))
imgcopy = image.copy()

imgcopy = img_to_array(imgcopy)
imgcopy = imgcopy/255
plt.axis('off')
img = np.expand_dims(imgcopy,axis=0)

model = load_model(args["model"])

classes = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]


prob = model.predict(img)
print(prob)
top = np.argmax(prob[0])
print(classes[top])
print(prob[0][top])
plt.imshow(image)