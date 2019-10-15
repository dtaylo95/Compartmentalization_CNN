"""
Usage: python gen_train_test_sets.py <train_percent> <test_percent> <validate_percent> [FILES.npz] <out_file>
"""

import sys
import numpy as np


train_portion = float(sys.argv[1])/100
test_portion  = float(sys.argv[2])/100
val_portion = float(sys.argv[3])/100

if sum([int(x) for x in sys.argv[1:4]]) != 100:
	print('Proposed split does not add to 100%, please fix these values and try again')
	exit()

first_dataset = np.load(sys.argv[4])
total_data_inputs = first_dataset['input_data']
total_data_targets = first_dataset['target_data']
print('Read in first dataset')

if len(sys.argv) > 5:
	for i in range(5, len(sys.argv)-1):
		print(sys.argv[i])
		local_data = np.load(sys.argv[i])
		total_data_inputs = np.append(total_data_inputs, local_data['input_data'], axis=0)
		total_data_targets = np.append(total_data_targets, local_data['target_data'], axis=0)
		print('Read in data')
"""
The following lines determine how many windows in the full data set are entirely in the zero-state
"""
# zero_array = [50] + [0]*26
# print(zero_array)

# total_count = 0
# non_zero_count = 0
# for axis0_obj in total_data_inputs:
# 	all_zero = True
# 	for axis1_obj in axis0_obj:
# 		if list(axis1_obj) != zero_array:
# 			all_zero = False
# 			non_zero_count += 1
# 			break
# 	total_count += 1

# print(non_zero_count, total_count)


out_file = sys.argv[-1] 

samples = total_data_inputs.shape[0]

test_size = int(samples*test_portion)
val_size = int(samples*val_portion)

test_rows = np.random.choice(samples, test_size, replace=False)
test_samples = np.array([total_data_inputs[i] for i in test_rows])
test_targets = np.array([total_data_targets[i] for i in test_rows])

remaining_inputs = np.delete(total_data_inputs, test_rows, axis=0)
remaining_targets = np.delete(total_data_targets, test_rows, axis=0)

val_rows = np.random.choice(remaining_inputs.shape[0], val_size, replace=False)
val_samples = np.array([remaining_inputs[i] for i in val_rows])
val_targets = np.array([remaining_targets[i] for i in val_rows])

train_samples = np.delete(remaining_inputs, val_rows, axis=0)
train_targets = np.delete(remaining_targets, val_rows, axis=0)


if val_samples.shape[0] == 0:
	np.savez(out_file, train_samples=train_samples, train_targets=train_targets, test_samples=test_samples, test_targets=test_targets)
else:
	np.savez(out_file, train_samples=train_samples, train_targets=train_targets, test_samples=test_samples, test_targets=test_targets, val_samples=val_samples, val_targets=val_targets)

