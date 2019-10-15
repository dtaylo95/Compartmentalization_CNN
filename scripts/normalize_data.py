"""
Usage: python normalize_data.py <sep_train_test_data.npz> [output.npz]
"""
from tensorflow import keras
import tensorflow as tf
import numpy as np
import sys

my_data = np.load(sys.argv[1])

training_inputs = my_data['train_samples']
training_targets = my_data['train_targets']
testing_inputs = my_data['test_samples']
testing_targets = my_data['test_targets']
validation_inputs = my_data['val_samples']
validation_targets = my_data['val_targets']


training_inputs = training_inputs[:,:,1:].astype(float)
testing_inputs = testing_inputs[:,:,1:].astype(float)
validation_inputs = validation_inputs[:,:,1:].astype(float)

training_targets = (training_targets + 1)/2
testing_targets = (testing_targets + 1)/2
validation_targets = (validation_targets + 1)/2

total_inputs = np.append(np.append(training_inputs, testing_inputs, axis=0), validation_inputs, axis=0).astype(float)

for i in range(total_inputs.shape[2]):
	state_data = total_inputs[:,:,i].flatten().astype(float)
	state_mean = np.mean(state_data)
	state_std = np.std(state_data)
	training_inputs[:,:,i] = (training_inputs[:,:,i] - state_mean)/state_std
	testing_inputs[:,:,i] = (testing_inputs[:,:,i] - state_mean)/state_std
	validation_inputs[:,:,i] = (validation_inputs[:,:,i] - state_mean)/state_std

outfile = sys.argv[1]
if len(sys.argv) == 3:
	outfile = sys.argv[2]

np.savez(outfile, train_samples=training_inputs, train_targets=training_targets, test_samples=testing_inputs, test_targets=testing_targets, val_samples=validation_inputs, val_targets=validation_targets)
