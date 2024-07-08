# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
######################


def evaluate_lite_model(interpreter, test_dataset,  print_step = 1, class_names = None, show_confusion_matrix=False, show_normed = False):
	input_index = interpreter.get_input_details()[0]['index']
	output_index = interpreter.get_output_details()[0]['index']

	input_details = interpreter.get_input_details()[0]

	# Run predictions on every image in the "test" dataset.
	prediction_digits = []
	labels_list = []

	count = 1
	print('Processing test dataset. Batch to process: ', len(test_dataset) , '\n')
	for images, labels in test_dataset:
		if count % print_step == 0:
			print(f'Processing batch {count}...' , end = '\r')

		if input_details['dtype'] == np.uint8:
			input_scale, input_zero_point = input_details["quantization"]
			images = images / input_scale + input_zero_point

		for image in images:

			# Pre-processing: add batch dimension and convert to float32 to match with
			# the model's input data format.
			image = np.expand_dims(image, axis=0).astype(input_details["dtype"])
			
			# Run inference.
			interpreter.set_tensor(input_index, image)
			interpreter.invoke()

			# Post-processing: remove batch dimension and find the digit with highest
			# probability.
			output = interpreter.tensor(output_index)
			digit = np.argmax(output()[0])
			prediction_digits.append(digit)

		for label in labels:
			label = label.numpy()
			labels_list.append(label)

		count = count + 1

	# Compare prediction results with ground truth labels to calculate accuracy.
	prediction_digits = np.array(prediction_digits)
	labels_list = np.array(labels_list)
	
	accuracy = (prediction_digits == labels_list).mean()

	if show_confusion_matrix:
		conf_matrix = confusion_matrix(labels_list, prediction_digits)
		plot_confusion_matrix(conf_matrix, figsize=(12,12), class_names=class_names, show_normed=show_normed);
	return accuracy
