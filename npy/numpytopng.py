import numpy
import cv2
import os
from os import listdir

road = os.listdir("road")
non_road = os.listdir("non_road")

for x in non_road:
	print(x)
	im = numpy.load(f"non_road/{x}")
	cv2.imwrite(f"non_road_png/{x[0:-4]}.png", im)