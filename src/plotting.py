# Imports
import matplotlib.pyplot as plt
import numpy as np
from src.preprocessing import tensorflow_to_numpy_dataset
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from src.inference import inference,inference_lite_model
######################

def plot_training_history(training_history, figure_size = (20,8)):

	training_accuracy = training_history.history['accuracy']
	validation_accuracy = training_history.history['val_accuracy']
	training_loss = training_history.history['loss']
	validation_loss = training_history.history['val_loss']

	epochs_range = range(len(training_accuracy))

	plt.figure(figsize=figure_size)
	plt.subplot(1,2,1)
	plt.plot(epochs_range, training_accuracy,   label = 'Trainin Accuracy')
	plt.plot(epochs_range, validation_accuracy, label = 'Validation Accuracy')
	plt.legend()
	plt.title('Accuracy for training and validation')

	plt.subplot(1,2,2)
	plt.plot(epochs_range, training_loss,   label = 'Trainin Loss')
	plt.plot(epochs_range, validation_loss, label = 'Validation Loss')
	plt.legend()
	plt.title('Loss for training and validation')

	plt.show()


def plot_test_images(cnn_model, test_dataset, class_names, num_images = 16, scaling = 255, offset = 0.0):
	plt.figure(figsize=(20,20))
	for images, labels in test_dataset.take(1):
		for i in range(num_images):
			ax = plt.subplot(4,4, i+1)
			predicted_class, confidence = inference(cnn_model, images[i], class_names)
			actual_class = class_names[labels[i]]

			image =  (images[i] + offset) * scaling
			plt.imshow(image.numpy().astype('uint8'))
			plt.title(f'Actual: {actual_class} \n Predicted: {predicted_class} \n Confidence: {confidence:.2f}%')
			plt.axis('off')


def plot_test_images_lite(interpreter, test_dataset, class_names, num_images = 16, scaling = 255, offset = 0.0):
	plt.figure(figsize=(20,20))
	for images, labels in test_dataset.take(1):
		for i in range(num_images):
			ax = plt.subplot(4,4, i+1)
			predicted_class, confidence = inference_lite_model(interpreter, images[i], class_names, scale = scaling, show_image = False)
			actual_class = class_names[labels[i]]

			image =  (images[i] + offset) * scaling
			plt.imshow(image.numpy().astype('uint8'))
			plt.title(f'Actual: {actual_class} \n Predicted: {predicted_class} \n Confidence: {confidence:.2f}%')
			plt.axis('off')

def plotting_confusion_matrix(testing_dataset, model, class_names, show_normed = False):

	images_batch_np, labels_batch_np = tensorflow_to_numpy_dataset(testing_dataset)

	
	predictions = model.predict(images_batch_np)
	predictions = np.argmax(predictions,axis=1)

	conf_matrix = confusion_matrix(labels_batch_np, predictions)
	plot_confusion_matrix(conf_matrix, figsize=(12,12), class_names=class_names, show_normed=show_normed);