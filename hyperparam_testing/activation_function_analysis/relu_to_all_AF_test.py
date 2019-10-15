"""
Usage: python build_model.py <normalized_data.npz>
"""
from tensorflow import keras
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

my_data = np.load(sys.argv[1])

training_inputs = my_data['train_samples']
training_targets = my_data['train_targets']
testing_inputs = my_data['test_samples']
testing_targets = my_data['test_targets']
validation_inputs = my_data['val_samples']
validation_targets = my_data['val_targets']

# training_inputs = training_inputs[::1000,:,:]
# training_targets = training_targets[::1000]
# validation_inputs = validation_inputs[::1000,:,:]
# validation_targets = validation_targets[::1000]

width, channels = training_inputs[0].shape
epochs=20

AF_dict = {}

conv1AF = 'relu'
iterations = 10

for conv2AF in ['linear','sigmoid','relu']:
	for denseAF in ['linear','sigmoid','relu']:
		AF_dict[conv1AF, conv2AF, denseAF] = []
		for i in range(iterations):

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
									  verbose=1)

			val_loss = list(model_history.history['val_loss'])
			val_loss = [x/iterations for x in val_loss]
			val_error = list(model_history.history['val_mean_absolute_error'])
			val_error = [x/iterations for x in val_error]
			if AF_dict[(conv1AF,conv2AF,denseAF)] == []:
				AF_dict[(conv1AF,conv2AF,denseAF)] = [val_loss,val_error]
			else:
				AF_dict[(conv1AF,conv2AF,denseAF)][0] = [AF_dict[(conv1AF,conv2AF,denseAF)][0][j] + val_loss[j] for j in range(epochs)]
				AF_dict[(conv1AF,conv2AF,denseAF)][1] = [AF_dict[(conv1AF,conv2AF,denseAF)][1][j] + val_error[j] for j in range(epochs)]



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




# fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(14,20))

# for AF_comb in AF_dict:
# 	if AF_comb[0] == 'linear':
# 		axes[0,0].plot([x for x in range(1, epochs+1)], AF_dict[AF_comb][0], label=str(AF_comb))
# 		axes[0,1].plot([x for x in range(1, epochs+1)], AF_dict[AF_comb][1], label=str(AF_comb))
# 	if AF_comb[0] == 'sigmoid':
# 		axes[1,0].plot([x for x in range(1, epochs+1)], AF_dict[AF_comb][0], label=str(AF_comb))
# 		axes[1,1].plot([x for x in range(1, epochs+1)], AF_dict[AF_comb][1], label=str(AF_comb))
# 	if AF_comb[0] == 'relu':
# 		axes[2,0].plot([x for x in range(1, epochs+1)], AF_dict[AF_comb][0], label=str(AF_comb))
# 		axes[2,1].plot([x for x in range(1, epochs+1)], AF_dict[AF_comb][1], label=str(AF_comb))

# axes[0,1].set_ylabel('Mean Absolute Error')
# axes[0,0].set_ylabel('Mean Squared Error')
# axes[1,1].set_ylabel('Mean Absolute Error')
# axes[1,0].set_ylabel('Mean Squared Error')
# axes[2,1].set_ylabel('Mean Absolute Error')
# axes[2,0].set_ylabel('Mean Squared Error')

# print(min([min(AF_dict[AF_comb][0]) for AF_comb in AF_dict]), max([max(AF_dict[AF_comb][0]) for AF_comb in AF_dict]))
# print(min([min(AF_dict[AF_comb][1]) for AF_comb in AF_dict]), max([max(AF_dict[AF_comb][1]) for AF_comb in AF_dict]))

# axes[0,0].set_ylim(min([min(AF_dict[AF_comb][0]) for AF_comb in AF_dict]), max([max(AF_dict[AF_comb][0]) for AF_comb in AF_dict]))
# axes[0,1].set_ylim(min([min(AF_dict[AF_comb][1]) for AF_comb in AF_dict]), max([max(AF_dict[AF_comb][1]) for AF_comb in AF_dict]))
# axes[1,0].set_ylim(min([min(AF_dict[AF_comb][0]) for AF_comb in AF_dict]), max([max(AF_dict[AF_comb][0]) for AF_comb in AF_dict]))
# axes[1,1].set_ylim(min([min(AF_dict[AF_comb][1]) for AF_comb in AF_dict]), max([max(AF_dict[AF_comb][1]) for AF_comb in AF_dict]))
# axes[2,0].set_ylim(min([min(AF_dict[AF_comb][0]) for AF_comb in AF_dict]), max([max(AF_dict[AF_comb][0]) for AF_comb in AF_dict]))
# axes[2,1].set_ylim(min([min(AF_dict[AF_comb][1]) for AF_comb in AF_dict]), max([max(AF_dict[AF_comb][1]) for AF_comb in AF_dict]))

# axes[0,1].set_xlabel('Epoch')
# axes[0,0].set_xlabel('Epoch')
# axes[1,1].set_xlabel('Epoch')
# axes[1,0].set_xlabel('Epoch')
# axes[2,1].set_xlabel('Epoch')
# axes[2,0].set_xlabel('Epoch')

# axes[0,1].legend()
# axes[0,0].legend()
# axes[1,1].legend()
# axes[1,0].legend()
# axes[2,1].legend()
# axes[2,0].legend()

plt.tight_layout()
fig.savefig('Activation_Function_Analysis_relu.png')

plt.close()

