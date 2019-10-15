"""
Usage: pytohn test_load_model.py <model_file.h5> <normalized_data.npz>
"""

import sys
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model(sys.argv[1])

my_data = np.load(sys.argv[2])

validation_inputs = my_data['val_samples']
validation_targets = my_data['val_targets']

predicted_validation = model.predict(validation_inputs)

sorted_data = sorted(list(zip(validation_targets, predicted_validation)), key = lambda x: x[0])

sorted_true = [x[0] for x in sorted_data]
sorted_pred = [float(x[1]) for x in sorted_data]

error = [sorted_pred[i]-sorted_true[i] for i in range(len(sorted_true))]

plt.style.use('dark_background')

fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(20,8))
ax1.scatter(x=range(len(sorted_pred)), y=sorted_pred, s=4, alpha=0.3, color='#1f78b3', label='Predicted Cscore data')
ax1.plot(sorted_true, color='#fe7f02', label='True Cscore data')
ax1.set_xticklabels([])
ax1.set_xlabel('Sorted Validation Data')
ax1.set_ylabel('Scaled Cscore')
leg1 = ax1.legend()
for lh in leg1.legendHandles:
	lh.set_alpha(1)

ax2.scatter(x=sorted_true, y=error, s=4, alpha=0.3, color='#1f78b3')
ax2.set_ylim((-1,1))
ax2.plot([0,1], [0,0], color='#fe7f02')
ax2.set_xlabel('Validation True Cscore')
ax2.set_ylabel('Raw Cscore Error (predicted - true)')
plt.tight_layout()
fig.savefig('10-14-19_pred_error_cscores.png')
plt.close()