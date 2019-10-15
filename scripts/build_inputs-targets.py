"""
Parses a .wig file containing cscore data and .bed file containing
IDEAS states in the same cell type to build inputs/targets for CNN model
described on 9/23/19.

Usage: python build_inputs-targets.py <cscore_data.wig> <ideas_states.bed> <out_file.npz>
"""

import numpy as np
import sys
import time
import matplotlib.pyplot as plt

###################################################################################################
"""FUNCTIONS"""

# Multi-hot encode over a chunk of IDEAS states (num_states includes the zero state, so 27 states for mouse hematopoetic lineage)
def encode_chunk(chunk, num_states):
	encoded_chunk = [0 for  i in range(num_states)]

	for state in chunk:
		encoded_chunk[state] += 1

	return encoded_chunk


def remaining_time(start, job_id, total_jobs):
	run_time = time.time() - start_time
	return (run_time/job_id)*(total_jobs-job_id)


def readablesecs(time_in_secs):
	hours = int(time_in_secs/3600)
	minutes = int((time_in_secs - (hours*3600))/60)
	secs = float(time_in_secs - (hours*3600) - (minutes*60))

	hours_str =  ('0' + str(hours))[-2:]
	minutes_str = ('0' + str(minutes))[-2:]
	secs_str = ('0' + str(secs).split('.')[0])[-2:] + '.' + (str(secs).split('.')[1] + '00')[:2]

	return '%sh %sm %ss ' %(hours_str, minutes_str, secs_str)

###################################################################################################
"""DATA SHAPE VALUES"""

input_window_size = 100000 # The size of the input window (bp)
input_unit = 5000 # The size of units in the input (for example, 10000 would be the size of the cscore window, 200 would be the size of the IDEAS states)
				   # This must be a multiple of the IDEAS window (200 bp) and will sum IDEAS counts for each state if larger than 200

cscore_window_size = 10000 # The window size of the cscore data in the .wig file
IDEAS_window_size = 200 # The window size of the IDEAS data in the .bed file

num_IDEAS_states = 27 # The number of IDEAS states (including the 0 state)

###################################################################################################
"""EXECUTED CODE"""

cscore_file = open(sys.argv[1], 'r')
IDEAS_file = open(sys.argv[2], 'r')

# Store the data contained in the .wig file in a dictionary by chromosome and 
# region start site
cscore_data = {}
for line in cscore_file:
	if line.startswith('#'):
		continue
	fields = line.strip().split()
	chromosome = fields[0]
	region_start = int(fields[1])
	cscore = float(fields[3])

	cscore_data.setdefault(chromosome, {})
	cscore_data[chromosome][region_start] = cscore

# Store the data contained in the .bed file in a dictionary by chromosome and
# region start site
IDEAS_data = {}
for line in IDEAS_file:
	fields = line.strip().split()
	chromosome = fields[0]
	region_start = int(fields[1])
	region_end = int(fields[2])
	state = int(fields[3])

	IDEAS_data.setdefault(chromosome, {})

	for i in range(region_start, region_end, IDEAS_window_size):
		IDEAS_data[chromosome][i] = state

# Build input regions from cscore data
input_array = []
target_array = []
data_points = sum([len(cscore_data[chromosome]) for chromosome in cscore_data])
counter = 0
start_time = time.time()

predicted_times = []

input_regions = {}

for chromosome in cscore_data:
	input_regions[chromosome] = []	
	for cscore_start in cscore_data[chromosome]:
		
		covered_test = False
		for input_region in input_regions[chromosome]:
			if cscore_start in range(input_region[0], input_region[1]):
				covered_test = True
				break
		if covered_test:
			counter += 1
			continue

		if counter >= 1:
			sys.stdout.write('\r' + readablesecs(remaining_time(start_time, counter, data_points)) + ' remaining')
			sys.stdout.flush()
		cscore_target = cscore_data[chromosome][cscore_start]
		input_start = cscore_start - int((input_window_size/2) - (cscore_window_size/2))
		input_end = input_start + input_window_size
		local_IDEAS_states = []
		
		for i in range(input_start, input_end, IDEAS_window_size):
			if i in IDEAS_data[chromosome]:
				local_IDEAS_states.append(IDEAS_data[chromosome][i])
			else:
				break
		
		if len(local_IDEAS_states) != int(input_window_size/IDEAS_window_size):
			continue
		input_regions[chromosome].append((input_start, input_end))
		chunk_size = int(input_unit/IDEAS_window_size)
		local_state_chunks = [local_IDEAS_states[i:i+chunk_size] for i in range(0, len(local_IDEAS_states), chunk_size)]
		
		encoded_local_states = []
		for chunk in local_state_chunks:
			encoded_local_states.append(encode_chunk(chunk, num_IDEAS_states))
		input_array.append(encoded_local_states)
		target_array.append(cscore_target)
		counter += 1

input_array = np.array(input_array)
target_array = np.array(target_array)

print(input_array.shape, target_array.shape)

np.savez(sys.argv[3], input_data=input_array, target_data=target_array)
sys.stdout.write('\n')
sys.stdout.flush()