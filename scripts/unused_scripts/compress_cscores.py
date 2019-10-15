"""
Usage: python compress_cscores.py <out_window_size> <cscores.wig> <out_file>
"""

import sys


def avg(lst):
	return float(sum(lst))/len(lst)


out_window_size = int(sys.argv[1])
cscore_file = open(sys.argv[2],'r')

cscore_data = {}

out_file = open(sys.argv[3],'w')

in_window_size = set()

for line in cscore_file:
	if line.startswith('#'):
		continue
	fields = line.strip().split()
	chromosome = fields[0]
	window_start = int(fields[1])
	window_end = int(fields[2])
	cscore = float(fields[3])
	window_size = window_end - window_start
	in_window_size.add(window_size)


	cscore_data.setdefault(chromosome, {})
	cscore_data[chromosome][window_start] = cscore

if len(in_window_size) == 1:
	in_window_size = list(in_window_size)[0]
else:
	print('Not all windows are of equal length. Remove windows of unequal size')
	exit()

if out_window_size % in_window_size != 0:
	print('Desired window size is not a multiple of current window size.\n' +
		  'Output window size must be a multiple of %d' % (in_window_size))
	exit()

for chromosome in cscore_data:
	while cscore_data[chromosome] != {}:
		first_window = min(cscore_data[chromosome].keys())
		new_start = first_window
		cscores = [cscore_data[chromosome][first_window]]
		del cscore_data[chromosome][new_start]
		for i in range(1, int(out_window_size/in_window_size)):
			start = new_start + (i*(in_window_size))
			if not start in cscore_data[chromosome]:
				print(chromosome, start)
				break
			cscores.append(cscore_data[chromosome][start])
			del cscore_data[chromosome][start]
		new_cscore = avg(cscores)
		new_end = start + in_window_size
		if new_end - new_start == out_window_size:
			out_file.write('%s\t%d\t%d\t%s\n' %(chromosome, new_start, new_end, float(new_cscore)))

out_file.close()
