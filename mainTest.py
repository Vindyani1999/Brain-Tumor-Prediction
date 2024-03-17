import cv2

from keras.saving import load_model

from PIL import Image
import numpy as np

model = load_model('BrainTumor10Epochs.h5')

image = cv2.imread('C:\\Users\\kachv\\OneDrive\\Documents\\5th Semester\\Mobile_Application_Development\\Cloningggggggggggggg\\Brain-Tumor-Prediction\\Testing\\image(1).jpg')

img = Image.fromarray(image)

img= img.resize((64,64))

img = np.array(img)

input_img = np.expand_dims(img, axis=0)

result = np.argmax(model.predict(input_img), axis=-1)

print(result)