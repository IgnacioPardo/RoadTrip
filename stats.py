from segmentation import segment, select, padded_segments
from count import default_threshold
from skimage import io
import numpy as np

default_selection = {106, 108, 239, 340, 368, 329, 405, 507, 636, 671,
			 		 749, 969, 1151, 667, 511, 432, 943, 864, 792, 931,
			 		 1060, 1094, 1160}

def array_not(a):
	return (not elem for elem in a)

def array_or(a, b):
	return (elem_a or elem_b for elem_a, elem_b in zip(a, b))

def array_and(a, b):
	return (elem_a and elem_b for elem_a, elem_b in zip(a, b))

def confusion_matrix(image, model, **kwargs):
	threshold = kwargs.get("threshold", default_threshold)

	segments = segment(image)
	segment_values = range(segments.max() + 1)

	selection = default_selection

	ground_truth = [True if i in selection else False for i in segment_values]

	padded, padded_segment_values = padded_segments(image, segments, segment_values)
	partial_predictions = model.predict(padded)[:,0] > threshold
	predictions = []

	index = 0
	for i in segment_values:
		if i in padded_segment_values:
			predictions.append(partial_predictions[index])
			index += 1
		else:
			predictions.append(False)

	return [[sum(array_and(ground_truth, predictions)), sum(array_and(ground_truth, array_not(predictions)))],
			[sum(array_and(array_not(ground_truth), predictions)), sum(array_not(array_or(ground_truth, predictions)))]]