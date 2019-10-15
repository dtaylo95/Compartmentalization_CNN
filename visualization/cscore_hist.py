"""
Usage: python score_hist.py <cscores.wig> <cscore_hist.png>
"""

import sys
import matplotlib.pyplot as plt

wig_file = open(sys.argv[1], 'r')

cscores = []

for line in wig_file:
	if line.startswith('#'):
		continue
	else:
		fields = line.strip().split()
		cscore = fields[3]
		cscores.append(float(cscore))

print('Done reading file')

fig, ax = plt.subplots()
ax.hist(cscores, bins=100)
ax.set_ylim(0,10000)
fig.savefig(sys.argv[2])
plt.close()