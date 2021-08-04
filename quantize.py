import tensorflow as tf
import pathlib
import argparse

quant_dict = {
	"fp32": tf.float32,
	"fp16": tf.float16,
	"int16": tf.int16,
	"int8": tf.uint8,
	"none": None
}

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--precision", required=True)
	parser.add_argument("-m", "--model-repo", required=True)

	args = parser.parse_args()

	if args.precision is not None:
		quant_precision = args.precision
	if args.model_path is not None:
		model_repo = args.model_repo

	saved_model_dir = model_repo + "/model.savedmodel"
	tflite_model_dir = model_repo + "/tflite/"

	converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

	if quant_dict[quant_precision] == "none":
		pass
	else:
		converter.optimizations = [tf.lite.Optimize.DEFAULT]
		converter.target_spec.supported_types = [quant_dict[quant_precision]]
	
	tflite_model = converter.convert()
	tflite_model_name = tflite_model_dir + str(quant_dict[quant_dict[quant_precision]]) + ".tflite"
	open(tflite_model_name, "wb").write(tflite_model)