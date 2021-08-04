from __future__ import absolute_import, division, print_function, unicode_literals

from numpy.core.fromnumeric import argmax

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import time
import argparse
import os

def test_image_generator(test_data_path, image_shape, batch_size):
	datagen_kwargs = dict(rescale=1./255)	# no split here
	test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
	test_generator = test_datagen.flow_from_directory(
		test_data_path,
		subset="validation",
		shuffle=True,
		target_size=image_shape,
		batch_size=batch_size
	)
	return test_generator

def find_label(predict_index):
	mapping_file = open("../LOC_synset_mapping.txt")
	for i, line in enumerate(mapping_file):
		if i == predict_index - 1:
			nl = line.split(" ", 1)[0]
		elif i >= predict_index:
			break
	mapping_file.close()
	return nl

def run_eval(interpreter, image):
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	image = np.reshape(image, input_details[0]['shape'])
	interpreter.set_tensor(input_details[0]['index'], image)
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details[0]['index'])
  	output = np.squeeze(output_data)
  	return output

def get_image_size(model_name):
	if model_name == "mobilenet_v2":
		return (224, 224)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--precision", required=True)
	parser.add_argument("-m", "--model-repo", required=True)
	parser.add_argument("-d", "--dataset-path", required=False)
	args = parser.parse_args()

	quant_precision = args.precision
	model_repo = args.model_repo
	if args.dataset is not None:
		dataset_path = args.dataset_path
	else:
		dataset_path = "../dataset/imagenet-mini/"

	image_shape = get_image_size(model_repo)
	test_generator = test_image_generator(dataset_path + "val", image_shape, 1) # TODO: figure out batch size
	dataset_labels = sorted(test_generator.class_indices.items(), key=lambda pair:pair[1])
	dataset_labels = np.array([key.title() for key, value in dataset_labels])

	tflite_model_name = quant_precision + ".tflite"
	tflite_model_path = os.path.join(model_repo, "tflite")
	interpreter = tf.lite.Interpreter(model_path=tflite_model_path+tflite_model_name)
	interpreter.allocate_tensors()
	
	start_time = time.time()

	correct, total = 0, 0
	for image, label in test_generator:
		total += 1
		output = run_eval(interpreter, image)
		label = np.squeeze(label)
		# ImageNet has 1000 labels while mobilenet trained with imagenet has 1001 labels including class 0 as "background"
		if np.argmax(label) == np.argmax(output) - 1:
			correct += 1
		if total % 500 == 0:
			print("Accuracy after {} images: {}".format(total, float(correct)/float(total)))
		if total >= 20000:
			break
	
	print("--- %s seconds ---" % (time.time() - start_time))


