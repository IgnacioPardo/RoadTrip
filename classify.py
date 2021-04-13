import os
import sys

where_am_i = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, where_am_i+"/python_modules")

try:
	from tensorflow_core.keras.preprocessing.image import img_to_array
	from tensorflow_core.keras.models import load_model
except:
	from tensorflow.keras.preprocessing.image import img_to_array
	from tensorflow.keras.models import load_model

import numpy as np
import argparse
#import imutils

import pickle
import cv2

def predictRoads(full = False):
	# cargar las imagenes
	#images = [cv2.imread(f"images/{x}") for x in os.listdir("images")]

	images = []

	cant = len(list(filter(lambda s: s.endswith(".jpg"), os.listdir("images/"))))

	for i in range(cant):
		images.append(cv2.imread(f"images/image-{i}.jpg"))

	if full:
		images = []
		images.append(cv2.imread(f"files/stitched-images.png"))

	input_data = []

	for image in images:
		width, height, channels = image.shape
		for x in range (int(width / 32)):
			for y in range (int(height / 32)):
				img = np.full((32, 32, 3), 0)
				for X in range(32):
					for Y in range(32):
						img[X, Y, 0] = image[x * 32 + X, y * 32 + Y, 0]
						img[X, Y, 1] = image[x * 32 + X, y * 32 + Y, 1]
						img[X, Y, 2] = image[x * 32 + X, y * 32 + Y, 2]
				input_data.append(img)

	# pre-procesar las imagenes para la clasificacion
	input_data = [input_data[x].astype("float") / 255 for x in range(len(input_data))]

	input_data = np.array(input_data)

	# cargar la red neuronal entrenada
	print("[INFO] loading network...")
	model = load_model("models/street.model")

	# clasifical las imagenes de entrada
	print("[INFO] classifying images...")
	proba = model.predict(input_data)
	idx = [np.argmax(proba[x]) for x in range(len(input_data))]


	# crear nuevas imagenes de salida
	count = 0
	img_count = 0
	for image in images:
		width, height, channels = image.shape
		out_img = np.full((int(width / 32), int(height / 32)), 0)
		for x in range (int(width / 32)):
			for y in range (int(height / 32)):
				out_img[x, y] = idx[count] * 255
				count += 1
		width, height = out_img.shape
		for x in range(1, width - 1):
			for y in range(1, height - 1):
				next = out_img[x + 1, y] / 255 + out_img[x, y + 1] / 255 + out_img[x - 1, y] / 255 + out_img[x, y - 1] / 255
				if next <= 1:
					out_img[x, y] = 0
				elif next >= 3:
					out_img[x, y] = 255
		cv2.imwrite("predictions/prediction-{}.bmp".format(img_count), out_img)
		img_count += 1