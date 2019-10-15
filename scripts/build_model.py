"""
Usage: python build_model.py
"""
import argparse
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

def parse_AF_functs(AF_str):
	AF_funct_map = {
					'r' : 'relu',
					'l' : 'linear',
					's' : 'sigmoid',
					't' : 'tanh'
	}
	return [AF_funct_map[char] for char in AF_str]

def main():

	parser = argparse.ArgumentParser(description='Builds and trains a convolutional neural network on the given normalized data selections (.npz format)')
	parser.add_argument('-n', '--normalized_data', help='The normalized data in .npz format. The normalized data should be standardized first, then scaled from 0 to 1', required=True)
	parser.add_argument('-a','--activation_functions', help='The activation functions used for the three layers of the model. (r, l, s, t for relu, linear, sigmoid, tanh). Written as "rss", for example', required=False, default='rss')
	parser.add_argument('-o','--model_output', help='The output file for the trained model in .h5 format.', required=True)
	parser.add_argument('-p','--error_visualization', help='File name for visualization of loss (MSE) and error (MAE) of trained model over course of training regime (optional; default is no visualization', required=False, default=False)
	parser.add_argument('-e','--epochs', help='Number of epochs to run the model, default is 20', required=False, default=20)
	parser.add_argument('-b','--batch_size', help='The batch size of each epoch', required=False, default=100)
	parser.add_argument('-v','--verbose',help='The verbosity of the training output', required=False, default=1)
	args = parser.parse_args()

	AF_functs = parse_AF_functs(args.activation_functions)

	my_data = np.load(args.normalized_data)

	training_inputs = my_data['train_samples']
	training_targets = my_data['train_targets']
	testing_inputs = my_data['test_samples']
	testing_targets = my_data['test_targets']
	validation_inputs = my_data['val_samples']
	validation_targets = my_data['val_targets']


	width, channels = training_inputs[0].shape

	model = keras.Sequential()
	model.add(keras.layers.Conv1D(filters=100, kernel_size=5, strides=1, padding='VALID', input_shape=(width, channels), activation=AF_functs[0], use_bias=True))
	model.add(keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='VALID'))
	model.add(keras.layers.Conv1D(filters=50, kernel_size=4, strides=4, padding='VALID', activation=AF_functs[1], use_bias=True))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(100, activation=AF_functs[2], use_bias=True))
	model.add(keras.layers.Dense(1, activation='sigmoid'))

	model.compile(optimizer=tf.train.AdamOptimizer(0.001),
				  loss='mse',
				  metrics=['mae'])

	model_history = model.fit(training_inputs, training_targets,
			 				  epochs=args.epochs,
			 				  batch_size=args.batch_size,
							  shuffle=True,
							  validation_data=(validation_inputs, validation_targets),
							  verbose=args.verbose)

	model.save(args.model_output)

	train_loss = model_history.history['loss']
	val_loss = model_history.history['val_loss']
	train_error = model_history.history['mean_absolute_error']
	val_error = model_history.history['val_mean_absolute_error']

	if args.error_visualization != False:
		plt.style.use('dark_background')

		fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20,8))
		ax1.plot([x for x in range(len(train_loss))], train_loss, color='#fe7f02', label='training loss')
		ax1.plot([x for x in range(len(val_loss))], val_loss, color='#1f78b3', label='validation loss')
		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Mean Squared Error')
		ax1.legend()
		ax2.plot([x for x in range(len(train_error))], train_error, color='#fe7f02', label='training error')
		ax2.plot([x for x in range(len(val_error))], val_error, color='#1f78b3', label='validation error')
		ax2.set_xlabel('Epoch')
		ax2.set_ylabel('Mean Absolute Error')
		ax2.legend()
		plt.tight_layout()
		fig.savefig(args.error_visualization)

		plt.close()


if __name__ == '__main__':
	main()