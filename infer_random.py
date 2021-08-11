'''
for inference, randomly generated arrays can serve as the input
Goal:
	* profile execution time across the tflite models with different quantization (to reduce the impact of image loading)
	* Measure power consumption in the meantime. May need to fork()
'''

from __future__ import absolute_import, division, print_function, unicode_literals

from numpy.core.fromnumeric import argmax

import matplotlib.pylab as plt
import tensorflow as tf
print("Tensorflow using GPU: {}".format(tf.test.is_built_with_cuda()))
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import time
import argparse
import os
import threading

gpu_power_list = []
cpu_power_list = []

def measure_power():
	while True:
		global stop_threads
		if stop_threads:
			break
		# get power measurement of GPU
		with open('/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input','r') as f1: 
			for line in f1:  
				for word in line.split():
					gpu_power_list.append(int(word))
					# print("gpu: {}".format(word))
		f1.close()
		# get power measurement of CPU
		with open('/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_power1_input','r') as f2: 
			for line in f2:  
				for word in line.split():
					cpu_power_list.append(int(word))
					# print("cpu: {}".format(word))
		f2.close()
		# sample every 100ms?
		time.sleep(0.001)

def get_average_power(power_list):
	return sum(power_list)/len(power_list)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--precision", required=True)
	parser.add_argument("--model-repo", required=True)
	parser.add_argument("--power-measure", required=False)
	args = parser.parse_args()

	quant_precision = args.precision
	model_repo = args.model_repo
	if args.power_measure is not None:
		power_measure = args.power_measure
	else:
		power_measure = "off"

	stop_threads = False
	start_time = time.time()
	if power_measure == "on":
		t1 = threading.Thread(target = measure_power)
		t1.start()
		print("Wait for power measurement to start...")
		time.sleep(1)

	tflite_model_name = "lite_model_" + quant_precision + ".tflite"
	tflite_model_path = os.path.join(model_repo, "tflite", tflite_model_name)
	print("Loading model {}...".format(tflite_model_path))
	interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
	interpreter.allocate_tensors()

	for i in range(10000):
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		input_shape = input_details[0]['shape']
		input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
		interpreter.set_tensor(input_details[0]['index'], input_data)
		interpreter.invoke()
		output_data = interpreter.get_tensor(output_details[0]['index'])
		if i % 1000 == 0:
			print("Elapsed time after processing {}-th input: {} seconds".format(i, time.time()-start_time))
			print("Average power of GPU: {}".format(get_average_power(gpu_power_list)))
			print("Average power of CPU: {}".format(get_average_power(cpu_power_list)))
			

	print("--- %s seconds ---" % (time.time() - start_time))
	time.sleep(1)
	stop_threads = True
	t1.join()

	print("Writing power measurements to files...")
	cpu_filename = "power_results/cpu_power_" + quant_precision + ".txt"
	gpu_filename = "power_results/gpu_power_" + quant_precision + ".txt"
	with open(cpu_filename, "w") as f:
		for item in cpu_power_list:
			f.write("%s\n" % item)
	
	with open(gpu_filename, "w") as f:
		for item in gpu_power_list:
			f.write("%s\n" % item)