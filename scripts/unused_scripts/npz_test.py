import numpy as np
import sys

# np.set_printoptions(suppress=True)

npz_file = sys.argv[1]

my_data = np.load(npz_file)

for label in my_data:
	print(label)
	print(my_data[label])