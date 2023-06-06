# make a prediction for a new image.
from numpy import argmax
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO

# load and prepare the image
def load_image(img):
	# load the image
	img = img.resize((28,28))
	img.save('img_new.png')
	img = img.convert('L')
	img_array = img_to_array(img)
	# img = to_categorical(img)
	img_array = img_array.reshape(1, 28, 28, 1)
	img_array = img_array.astype('float32')
	img_array = img_array / 255.0
	return img_array


def predict_image(imageData):
	# load the image
	img = load_image(imageData)
	# img = load_img('sample_image.png')
	# img = load_image(img)
	# load model
	model = load_model('final_model.h5')
	print('MODEL LOCK AND LOADED \n\n')

	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print("prediction is:",predict_value)
	print('Digit is:',digit)
	return digit	

# predict_image()
# entry point, run the example