"""
Usage: python plot_comp_over_chr.py <cscores.wig> <chr_name> <hist_name.png>
"""

import sys
import matplotlib.pyplot as plt

chroi = sys.argv[2]

wig_file = open(sys.argv[1], 'r')


loci = []
cscores = []

for line in wig_file:
	if line.startswith('#'):
		continue
	else:
		fields = line.strip().split()
		chr_name = fields[0]
		if chr_name == chroi:
			locus_start = int(fields[1])
			locus_cscore = float(fields[3])
			loci.append(locus_start)
			cscores.append(locus_cscore)

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(loci, cscores)
ax.set_xlim(0,100000000)
fig.savefig(sys.argv[3])