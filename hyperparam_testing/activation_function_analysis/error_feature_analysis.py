"""
Usage: python error_feature_analysis.py <normalized_data.npz> <out_file.png>
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from statistics import mode


trials = []
for AF1 in ['linear']:
	for AF2 in ['linear']:
		for AF3 in ['linear']:
			for AF4 in ['sigmoid']:
				trials.append((AF1,AF2,AF3,AF4))

my_data = np.load(sys.argv[1])

training_inputs = my_data['train_samples']
training_targets = my_data['train_targets']
validation_inputs = my_data['val_samples']
validation_targets = my_data['val_targets']

width, channels = training_inputs[0].shape

plt.style.use('dark_background')
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(10,10))



# for i, ax in enumerate(axes):
for i in range(len(trials)):
	AF_functs = trials[i]
	model = keras.Sequential()
	model.add(keras.layers.Conv1D(filters=100, kernel_size=5, strides=1, padding='VALID', input_shape=(width, channels), activation=AF_functs[0], use_bias=True))
	model.add(keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='VALID'))
	model.add(keras.layers.Conv1D(filters=50, kernel_size=4, strides=4, padding='VALID', activation=AF_functs[1], use_bias=True))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(100, activation=AF_functs[2], use_bias=True))
	model.add(keras.layers.Dense(1, activation='sigmoid', use_bias=True))

	model.compile(optimizer=tf.train.AdamOptimizer(0.001),
				  loss='mse',
				  metrics=['mae'])

	model_history = model.fit(training_inputs, training_targets,
			 				  epochs=1,
			 				  batch_size=100,
							  shuffle=True,
							  validation_data=(validation_inputs, validation_targets),
							  verbose=1)

	predicted_validation = model.predict(validation_inputs)

	sorted_data = sorted(list(zip(validation_targets, predicted_validation)), key = lambda x: x[0])

	sorted_true = [x[0] for x in sorted_data]
	sorted_pred = [float(x[1]) for x in sorted_data]

	predicted_mode = mode(sorted_pred)

	mode_count = 0
	for i in range(len(predicted_validation)):
		if predicted_validation[i] == predicted_mode:
			mode_count += 1
	print(mode_count)

	error = [sorted_pred[i]-sorted_true[i] for i in range(len(sorted_true))]

	ax.scatter(x=range(len(sorted_pred)), y=sorted_pred, s=4, alpha=0.3, color='#1f78b3', label='Predicted Cscore data', zorder=5)
	ax.plot(sorted_true, color='#fe7f02', label='True Cscore data', zorder=10)
	ax.axhline(y=predicted_mode, label='y = %s' %(str(predicted_mode)), ls='--', color='red', zorder=0, alpha=0.5)
	ax.set_xticklabels([])
	ax.set_xlabel('Sorted Validation Data')
	ax.set_ylabel('Scaled Cscore')
	ax.set_title(str(AF_functs))
	leg = ax.legend()
	for lh in leg.legendHandles:
		lh.set_alpha(1)

plt.tight_layout()
fig.savefig(sys.argv[2])
plt.close()