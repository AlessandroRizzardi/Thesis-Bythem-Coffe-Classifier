# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
######################


def inference(cnn_model, image, class_names, show_image = False, verbose = 0):
	
	image_array = tf.keras.utils.img_to_array(image)
	image_array = tf.expand_dims(image_array,0) #create a batch
	
	if show_image:
		plt.imshow(np.array(image_array[0]).astype('uint8'))
		plt.show()
		  
	predictions = cnn_model.predict(image_array, verbose = 0)
	predicted_class = class_names[np.argmax(predictions[0])]
	confidence =  100 * np.max(predictions[0])
  
	return predicted_class, confidence
  
def inference_lite_model(interpreter, image, class_names, scale, show_image = False):
	
	input_index = interpreter.get_input_details()[0]['index']
	output_index = interpreter.get_output_details()[0]['index']

	input_details = interpreter.get_input_details()[0]

	if input_details['dtype'] == np.uint8:
		input_scale, input_zero_point = input_details["quantization"]
		image = image / input_scale + input_zero_point
	else:
		image = image/scale

	image = np.expand_dims(image, axis=0).astype(input_details["dtype"])

	interpreter.set_tensor(input_index, image)
	interpreter.invoke()

	if show_image:
		
		if input_details['dtype'] == np.uint8:
			image = image - input_zero_point
			image = image * input_scale
			image = image*scale
		else:
			image = image*scale
		
		plt.imshow(np.array(image[0]).astype('uint8'))
		plt.show()
	

	predictions = interpreter.get_tensor(output_index)

	predicted_class = class_names[np.argmax(predictions[0])]
	confidence =  100 * np.max(predictions[0])/255


	return predicted_class, confidence
