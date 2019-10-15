import sys
import numpy as np

my_data = np.load(sys.argv[1])

min_val = -1
max_val = 1
avg_bins = 100

targets = my_data['train_targets']
weighted_targets = targets*avg_bins

distance_sum = 0.0
distance_count = 0
# for i in range(min_val*avg_bins, max_val*avg_bins + 1):
# 	for target in targets:
# 		distance = abs(i - target)
# 		distance_sum += distance
# 		distance_count += 1

for i in range(min_val*avg_bins, max_val*avg_bins + 1):
	for j in [min_val*avg_bins, max_val*avg_bins]:
		distance = abs(i-j)
		distance_sum += distance
		distance_count += 1

avg_distance = distance_sum/distance_count
corrected_avg_distance = avg_distance/avg_bins

