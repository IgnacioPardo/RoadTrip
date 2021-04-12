import tensorflow_core.keras
from tensorflow_core.keras.layers.normalization import BatchNormalization
from tensorflow_core.keras.layers.convolutional import Conv2D
from tensorflow_core.keras.layers.convolutional import MaxPooling2D
from tensorflow_core.keras.layers.core import Activation
from tensorflow_core.keras.layers.core import Flatten
from tensorflow_core.keras.layers.core import Dropout
from tensorflow_core.keras.layers.core import Dense
from tensorflow_core.keras.optimizers import Adam
from tensorflow_core.keras.utils import Sequence
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
from segmentation import img_width, img_height, img_depth
import random
from argparse import ArgumentParser
import os
import sys


input_shape = (img_width, img_height, img_depth)

"""

	Modelo de la red

"""

model = keras.models.Sequential([

		Conv2D(8, (3, 3), padding = "same", input_shape = input_shape),
		Activation("relu"),
		BatchNormalization(),
		MaxPooling2D(pool_size = (2, 2)),
		Dropout(0.25),

		Conv2D(16, (3, 3), padding = "same"),
		Activation("relu"),
		BatchNormalization(),

		Conv2D(16, (3, 3), padding = "same"),
		Activation("relu"),
		BatchNormalization(),
		MaxPooling2D(pool_size = (2, 2)),
		Dropout(0.25),

		Flatten(),
		Dense(128),
		Activation("relu"),
		BatchNormalization(),
		Dropout(0.5),
		Dense(64),
		Activation("relu"),
		BatchNormalization(),
		Dropout(0.5),
		Dense(2),
		Activation("softmax")

	])

"""

	Training

"""


parser = ArgumentParser()
parser.add_argument("dataset_path", default=os.path.join("datasets", "1"))
parser.add_argument("--epochs", type=int)
args = parser.parse_args()
dataset_path = args.dataset_path

epochs = args.epochs
learning_rate = 1e-3
batch_size = 32

class DataGenerator(Sequence):
	def __init__(self, path_names):
		self.path_names = path_names
		self.indexes = np.arange(len(self.path_names))
		np.random.shuffle(self.indexes)

	def __len__(self):
		return int(np.floor(len(self.path_names) / batch_size))

	def on_epoch_end(self):
		self.indexes = np.arange(len(self.path_names))
		np.random.shuffle(self.indexes)

	def __getitem__(self, index):
		batch_indexes = self.indexes[index * batch_size: (index + 1) * batch_size]
		X = np.empty((batch_size, *input_shape), dtype="float32")
		y = []

		for i, batch_i in enumerate(batch_indexes):
			X[i,] = np.load(os.path.join(dataset_path, self.path_names[batch_i]))
			y.append(np.array([1, 0]) if self.path_names[batch_i][0] == 'c' else np.array([0, 1]))

		return X, np.array(y)



paths = os.listdir(args.dataset_path)

first_car = paths.index('c0')
car_paths = paths[first_car:]
non_car_paths = paths[:first_car]

np.random.shuffle(car_paths)
np.random.shuffle(non_car_paths)
split_point_non_car = int(len(non_car_paths) * 0.8)
split_point_car = int(len(car_paths) * 0.8)

training_generator = DataGenerator(non_car_paths[:split_point_non_car] + car_paths[:split_point_car])
testing_generator = DataGenerator(non_car_paths[split_point_non_car:] + car_paths[split_point_car:])

optimizer = Adam(lr=learning_rate, decay=learning_rate/epochs)

# Init model
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["binary_accuracy"])

# Train

def train():
	H = model.fit_generator(
		generator=training_generator,
		validation_data=testing_generator,
		steps_per_epoch=(split_point_non_car + split_point_car) / batch_size,
		epochs=epochs, verbose=1)

	model_index = len(os.listdir("models"))
	model.save(os.path.join("models", f"new_model_{model_index}"))

if __name__ == "__main__":
	train()