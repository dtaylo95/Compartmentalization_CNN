import multiprocessing as mp
from tensorflow import keras
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt

my_data = np.load(sys.argv[1])

training_inputs = my_data['train_samples']
training_targets = my_data['train_targets']
testing_inputs = my_data['test_samples']
testing_targets = my_data['test_targets']
validation_inputs = my_data['val_samples']
validation_targets = my_data['val_targets']

width, channels = training_inputs[0].shape

conv1AF = 'relu'
epochs = 20
iterations = 10

def run_model(epochs, conv2AF, denseAF):

	model = keras.Sequential()
	model.add(keras.layers.Conv1D(filters=100, kernel_size=5, strides=1, padding='VALID', input_shape=(width, channels), activation=conv1AF))
	model.add(keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='VALID'))
	model.add(keras.layers.Conv1D(filters=50, kernel_size=4, strides=4, padding='VALID', activation=conv2AF))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(100, activation=denseAF))
	model.add(keras.layers.Dense(1, activation='sigmoid'))

	model.compile(optimizer=tf.train.AdamOptimizer(0.001),
				  loss='mse',
				  metrics=['mae'])

	model_history = model.fit(training_inputs, training_targets,
			 				  epochs=epochs,
			 				  batch_size=100,
							  shuffle=True,
							  validation_data=(validation_inputs, validation_targets),
							  verbose=0)

	val_loss = list(model_history.history['val_loss'])
	val_error = list(model_history.history['val_mean_absolute_error'])
	print('Done!')
	return [(conv1AF,conv2AF,denseAF),val_loss,val_error]
	


data_inputs = []
for conv2AF in ['tanh', 'sigmoid', 'relu']:
	for denseAF in ['tanh','sigmoid','relu']:
		for i in range(iterations):
			data_inputs.append((epochs, conv2AF, denseAF))

pool = mp.Pool(10)
results = pool.starmap(run_model, data_inputs)
pool.close()

AF_dict = {}

for item in results:
	AF_comb = item[0]
	val_loss = [x/iterations for x in item[1]]
	val_error = [x/iterations for x in item[2]]
	if AF_comb not in AF_dict:
		AF_dict[AF_comb] = [val_loss, val_error]
	else:
		AF_dict[AF_comb][0] = [AF_dict[AF_comb][0][i] + val_loss[i] for i in range(len(val_loss))]
		AF_dict[AF_comb][1] = [AF_dict[AF_comb][1][i] + val_error[i] for i in range(len(val_error))]

plt.style.use('dark_background')

fig, axes = plt.subplots(nrows=2, figsize=(14,14))
for AF_comb in AF_dict:
	axes[0].plot([x for x in range(1, epochs+1)], AF_dict[AF_comb][0], label=str(AF_comb))
	axes[1].plot([x for x in range(1, epochs+1)], AF_dict[AF_comb][1], label=str(AF_comb))

axes[0].set_ylabel('Averaged MSE over %d iterations' %(iterations))
axes[1].set_ylabel('Averaged MAE over %d iterations' %(iterations))

axes[0].set_xlabel('Epochs')
axes[1].set_xlabel('Epochs')

axes[0].legend()
axes[1].legend()

plt.tight_layout()
fig.savefig('Activation_Function_Analysis_relu.png')

plt.close()
