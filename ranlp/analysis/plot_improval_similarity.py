
import sys
from matplotlib import pyplot
import numpy
from collections import defaultdict

from quoll.classification_pipeline.functions import docreader

data_in = sys.argv[1]
barchart_out = sys.argv[2]

# read data
dr = docreader.Docreader()
data = dr.parse_csv(data_in,delim=';')

# read data
system_pos = defaultdict(list)
system_neg = defaultdict(list)
man_system = defaultdict(list)
current_man = False
man_legend = []
for line in data[1:]:
	print(line)
	man = line[0]
	if not current_man:
		current_man = man
		man_legend.append(man)
		print('MAN LEGEND',man)
	else:
		if man != '': # new manipulation
			if 'SPTK' not in man_system[current_man]:
				system_pos['SPTK'].append(0.0)
				system_neg['SPTK'].append(0.0)
			if 'TRLM' not in man_system[current_man]:
				system_pos['TRLM'].append(0.0)
				system_neg['TRLM'].append(0.0)
			if 'Soft' not in man_system[current_man]:
				system_pos['Soft'].append(0.0)
				system_neg['Soft'].append(0.0)
			current_man = man
			man_legend.append(man)
			print('MAN LEGEND',man)
	system = line[1]
	pos = float(line[3])
	neg = float(line[5]) * -1
	system_pos[system].append(pos)
	system_neg[system].append(neg)
	man_system[current_man].append(system)
if 'SPTK' not in man_system[current_man]:
	system_pos['SPTK'].append(0.0)
	system_neg['SPTK'].append(0.0)
if 'TRLM' not in man_system[current_man]:
	system_pos['TRLM'].append(0.0)
	system_neg['TRLM'].append(0.0)
if 'Soft' not in man_system[current_man]:
	system_pos['Soft'].append(0.0)
	system_neg['Soft'].append(0.0)

# plot data
fig, ax = pyplot.subplots()
print('MAN SYSTEMS',man_system.keys())
index = numpy.arange(len(man_system.keys()))
bar_width = 0.10
opacity = 0.8

print('INDEX',index)

print('TRLM system pos',system_pos['TRLM'])

trlm_pos = pyplot.bar(index+bar_width, system_pos['TRLM'],bar_width,alpha=opacity,color='r',label='TRLM')
trlm_neg = pyplot.bar(index+bar_width, system_neg['TRLM'],bar_width,alpha=opacity,color='r')

soft_pos = pyplot.bar(index+(bar_width*2), system_pos['Soft'],bar_width,alpha=opacity,color='b',label='Soft')
soft_neg = pyplot.bar(index+(bar_width*2), system_neg['Soft'],bar_width,alpha=opacity,color='b')

print('SPTK pos',system_pos['SPTK'])
print('SPTK neg',system_neg['SPTK'])

sptk_pos = pyplot.bar(index+(bar_width*3), system_pos['SPTK'],bar_width,alpha=opacity,color='m',label='SPTK')
sptk_neg = pyplot.bar(index+(bar_width*3), system_neg['SPTK'],bar_width,alpha=opacity,color='m')

pyplot.ylim((-0.5,0.5))
pyplot.xlabel('Similarity metric')
pyplot.ylabel('Improved and worsened rankings compared to default metric of system')
pyplot.xticks(index + bar_width, man_legend)
pyplot.legend(loc='upper left')

# pyplot.tight_layout()
pyplot.savefig(barchart_out)
pyplot.clf()
