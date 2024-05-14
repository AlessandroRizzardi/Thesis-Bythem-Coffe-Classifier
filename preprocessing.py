# Imports
import numpy as np
import tensorflow as tf
######################



def dataset_partitioning(dataset, train_size = 0.8, validation_size = 0.1):
	dataset_batch_size = len(dataset)
	training_batch_size = int(train_size * dataset_batch_size)
	validation_batch_size = int(validation_size * dataset_batch_size)

	dataset = dataset.shuffle(1000)

	training_dataset =  dataset.take(training_batch_size)
	validation_dataset =  dataset.skip(training_batch_size).take(validation_batch_size)
	testing_dataset =  dataset.skip(training_batch_size).skip(validation_batch_size)

	return training_dataset, validation_dataset, testing_dataset


def dataset_prefetch(dataset):
	dataset = dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
	return dataset


def dataset_rescaling(dataset, scaling_factor, offset = 0.0):
	rescaling = tf.keras.layers.Rescaling(1./scaling_factor, offset)

	return dataset.map(lambda x,y: (rescaling(x, training=True), y))


def data_augmentation(dataset, seed=(2,3)):

	def augment(image, label):
		augmented_image = tf.image.stateless_random_flip_left_right(image, seed = seed)
		augmented_image = tf.image.stateless_random_flip_up_down(image, seed = seed)
		augmented_image = tf.image.stateless_random_brightness(image, max_delta = 0.2, seed = seed)
		augmented_image = tf.image.stateless_random_contrast(image, lower = 0.1, upper = 0.9 , seed = seed)
		augmented_image = tf.image.stateless_random_hue(image, max_delta = 0.2 , seed= seed)
		augmented_image = tf.image.stateless_random_saturation(image, lower = 0.1, upper = 0.9, seed=seed)
		augmented_image = tf.image.stateless_random_jpeg_quality(image, 75, 95, seed = seed)
		return (augmented_image, label)



	data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomBrightness(0.2),
											 tf.keras.layers.RandomContrast(0.2),
										  	 tf.keras.layers.RandomFlip("horizontal_and_vertical"), 
											 tf.keras.layers.RandomRotation(0.2), 
											 tf.keras.layers.RandomZoom(0.2),
    										 #tf.keras.layers.RandomCrop(height=224, width=224),
											 #tf.keras.layers.RandomHeight(0.2),
											 #tf.keras.layers.RandomWidth(0.2),
											 #tf.keras.layers.RandomTranslation(height_factor=0.9,width_factor=0.9)
											 ])
	

	dataset =  dataset.map(lambda x,y: (data_augmentation(x, training=True), y))
	#dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

	return dataset





def load_test_image(image_file, image_size):
	image = tf.keras.utils.load_img(image_file, target_size=(image_size, image_size))
	test_image = tf.keras.utils.img_to_array(image)

	return test_image

def tensorflow_to_numpy_dataset(tensorflow_dataset):
	images = []
	labels = []
	for image_batch, label_batch in tensorflow_dataset:
		images.append(image_batch)
		labels.append(label_batch)

	images_batch_np = np.concatenate(images)
	labels_batch_np = np.concatenate(labels)

	return images_batch_np, labels_batch_np


def dataset_preprocessing(dataset, train_size=0.8, validation_size=0.1, augmentation_flag = False, rescaling_flag = False, prefetch_flag = False, scale = 255, offset = 0):
	
	training_dataset, validation_dataset, testing_dataset = dataset_partitioning(dataset, train_size, validation_size)


	if augmentation_flag:
		training_dataset = data_augmentation(training_dataset)

	if rescaling_flag:
		training_dataset = dataset_rescaling(training_dataset, scale, offset)
		validation_dataset = dataset_rescaling(validation_dataset, scale, offset)
		testing_dataset = dataset_rescaling(testing_dataset, scale, offset)

	if prefetch_flag:
		training_dataset = dataset_prefetch(training_dataset)
		validation_dataset = dataset_prefetch(validation_dataset)
		testing_dataset = dataset_prefetch(testing_dataset)	

	return training_dataset, validation_dataset, testing_dataset