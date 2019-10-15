import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from math import tanh, exp

def sigmoid(x):
	return exp(x)/(1 + exp(x))

def custom_AF1(n):
	def custom_func(x):
		return (K.sigmoid(x/n))
	return custom_func

def custom_AF2(n):
	def custom_func(x):
		return(n*K.tanh(x))
	return custom_func

def custom_AF1_notkeras(n):
	def custom_func(x):
		return (sigmoid(x/n))
	return custom_func

def custom_AF2_notkeras(n):
	def custom_func(x):
		return(n*tanh(x))
	return custom_func

def linear(x):
	return x

def relu(x):
	if x < 0:
		return 0
	return x

my_data = np.load(sys.argv[1])

training_inputs = my_data['train_samples']
training_targets = my_data['train_targets']
validation_inputs = my_data['val_samples']
validation_targets = my_data['val_targets']

width, channels = training_inputs[0].shape


n_vals = [n/2 for n in range(2,11)]

training_regimes = [custom_AF1(n=n) for n in n_vals]
training_regimes += [custom_AF2(n=n) for n in n_vals]
training_regimes += ['relu']
training_regimes += ['linear']

plot_training = [custom_AF1_notkeras(n=n) for n in n_vals]
plot_training += [custom_AF2_notkeras(n=n) for n in n_vals]
plot_training += [relu]
plot_training += [linear]

plt.style.use('dark_background')
fig, axes = plt.subplots(ncols=4, nrows=5, figsize=(40,40))

for i, ax in enumerate(axes.flat):
	training_regime = training_regimes[i]
	model = keras.Sequential()
	model.add(keras.layers.Conv1D(filters=100, kernel_size=5, strides=1, padding='VALID', input_shape=(width, channels), activation=training_regime, use_bias=True))
	model.add(keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='VALID'))
	model.add(keras.layers.Conv1D(filters=50, kernel_size=4, strides=4, padding='VALID', activation=training_regime, use_bias=True))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(100, activation=training_regime, use_bias=True))
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

	error = [sorted_pred[i]-sorted_true[i] for i in range(len(sorted_true))]

	ax.scatter(x=range(len(sorted_pred)), y=sorted_pred, s=4, alpha=0.3, color='#1f78b3', label='Predicted Cscore data', zorder=5)
	ax.plot(sorted_true, color='#fe7f02', label='True Cscore data', zorder=10)
	with plt.style.context('bmh'):
		axins = inset_axes(ax, width='30%', height='30%', loc=2)
		axins.plot([x/10 for x in range(-50,51)], [plot_training[i](x/10) for x in range(-50,51)], color='red')
		axins.set_xticks([x for x in range(-5,6)])
		axins.set_yticks([y for y in range(-5,6)])
		axins.set_xticklabels([])
		axins.set_yticklabels([])
		axins.set_xlim(-5,5)
		axins.set_ylim(-5,5)
		axins.axhline(y=0, color='black')
		axins.axvline(x=0, color='black')
	ax.set_xticklabels([])
	ax.set_xlabel('Sorted Validation Data')
	ax.set_ylabel('Scaled Cscore')

	# leg = ax.legend()
	# for lh in leg.legendHandles:
	# 	lh.set_alpha(1)

plt.tight_layout()
fig.savefig(sys.argv[2])
plt.close()