import numpy as np
import sys

my_data = np.load(sys.argv[1])

validation_inputs = my_data['val_samples']
validation_inputs = validation_inputs[:,:,1:].astype(float)

zero_state_count = 0
for sample in validation_inputs:
	if np.sum(sample) == 0:
		zero_state_count +=1
print(zero_state_count)