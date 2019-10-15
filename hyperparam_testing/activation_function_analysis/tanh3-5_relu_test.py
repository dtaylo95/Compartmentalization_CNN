import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K

def custom_AF(n):
	def custom_func(x):
		return(n*K.tanh(x))
	return custom_func

my_data = np.load(sys.argv[1])

training_inputs = my_data['train_samples']
training_targets = my_data['train_targets']
validation_inputs = my_data['val_samples']
validation_targets = my_data['val_targets']

width, channels = training_inputs[0].shape

epochs=30

training_regimes = []
for AF1 in [custom_AF(3.5),'relu']:
	for AF2 in [custom_AF(3.5),'relu']:
		for AF3 in [custom_AF(3.5),'relu']:
			training_regimes.append((AF1,AF2,AF3))

training_names = []
for AF1 in ['tanh3.5','relu']:
	for AF2 in ['tanh3.5','relu']:
		for AF3 in ['tanh3.5','relu']:
			training_names.append((AF1,AF2,AF3))

plt.style.use('dark_background')
fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(20,20))

ax_count = 0
for i in range(len(training_regimes)):
	training_regime = training_regimes[i]
	training_name = training_names[i]
	model = keras.Sequential()
	model.add(keras.layers.Conv1D(filters=100, kernel_size=5, strides=1, padding='VALID', input_shape=(width, channels), activation=training_regime[0], use_bias=True))
	model.add(keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='VALID'))
	model.add(keras.layers.Conv1D(filters=50, kernel_size=4, strides=4, padding='VALID', activation=training_regime[1], use_bias=True))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(100, activation=training_regime[2], use_bias=True))
	model.add(keras.layers.Dense(1, activation='sigmoid', use_bias=True))

	model.compile(optimizer=tf.train.AdamOptimizer(0.001),
				  loss='mse',
				  metrics=['mae'])

	model_history = model.fit(training_inputs, training_targets,
			 				  epochs=epochs,
			 				  batch_size=100,
							  shuffle=True,
							  validation_data=(validation_inputs, validation_targets),
							  verbose=1)

	predicted_validation = model.predict(validation_inputs)

	sorted_data = sorted(list(zip(validation_targets, predicted_validation)), key = lambda x: x[0])

	sorted_true = [x[0] for x in sorted_data]
	sorted_pred = [float(x[1]) for x in sorted_data]

	train_error = model_history.history['mean_absolute_error']
	val_error = model_history.history['val_mean_absolute_error']

	axes.flat[ax_count].scatter(x=range(len(sorted_pred)), y=sorted_pred, s=4, alpha=0.3, color='#1f78b3', label='Predicted Cscore data', zorder=5)
	axes.flat[ax_count].plot(sorted_true, color='#fe7f02', label='True Cscore data', zorder=10)
	axes.flat[ax_count].set_xticklabels([])
	axes.flat[ax_count].set_xlabel('Sorted Validation Data')
	axes.flat[ax_count].set_ylabel('Scaled Cscore')
	axes.flat[ax_count].set_title(str(training_name))

	ax_count += 1
	axes.flat[8].plot([x for x in range(1,epochs+1)], val_error, label=str(training_name))

for ax in axes.flat:
	ax.legend()
axes.flat[8].set_ylabel('Validation Mean Absolute Error')
axes.flat[8].set_xlabel('Epoch')
plt.tight_layout()
fig.savefig(sys.argv[2])
plt.close()