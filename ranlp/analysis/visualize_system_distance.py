
import sys
import numpy
from scipy.spatial import distance
from matplotlib import pyplot
import networkx

system_vectors_in = sys.argv[1]
fig_out = sys.argv[2]

def calculate_distance(vector1,vector2):
    vectordist = distance.euclidean(vector1,vector2)
    return vectordist

def generate_system_distance_matrix(systems_performance):
    distance_matrix = []
    for system_row in systems_performance:
        row = []
        for system_column in systems_performance:
            dist = calculate_distance(system_row,system_column)
            row.append(dist)
        distance_matrix.append(row)
    distance_matrix = numpy.array(distance_matrix)
    return distance_matrix

def generate_heatmap(vectors,rownames):
    pyplot.pcolor(vectors, norm=None, cmap='Blues')
    pyplot.gca().invert_yaxis()        
  #   if self.group_names:
  #       ticks_groups = []
  #       bounds = []
  #       current_group = False
  #       start = 0
  #       for i,doc in enumerate(self.document_names):
  #           group = self.document_group_dict[doc]
  #           if group != current_group: 
  #               if i != 0:
  #                   bounds.append(i-1)
  #                   ticks_groups[start+int((i-start)/2)] = current_group
  #               current_group = group
  #               start=i
  #           ticks_groups.append('')
  #       ticks_groups[start+int((i-start)/2)] = current_group
  # #      if self.rows > self.columns:
  #       pyplot.xticks(numpy.arange(columns)+0.5,ticks_groups, fontsize=11)
  #       if set_topics:
  #           for index in set_topics:
  #               pyplot.axhline(y=index)
  #           topic_names = self.return_topic_names(set_topics)
  #           pyplot.yticks(set_topics,topic_names,fontsize=8)
  #       else:
    pyplot.yticks(numpy.arange(rows)+0.5, rownames, fontsize=8)
        #pyplot.tight_layout()
    for bound in bounds:
        pyplot.axvline(x=bound)
   #     else:
        #pyplot.xticks(numpy.arange(columns)+0.5, columnnames)
            #pyplot.xticks(rotation=90)                         
        #pyplot.yticks(numpy.arange(rows)+0.5,ticks_groups)
        #for bound in bounds:
        #    pyplot.axhline(y=bound)
    #else:
    #    pyplot.xticks(numpy.arange(columns)+0.5, columnnames)
    #    pyplot.yticks(numpy.arange(rows)+0.5, rownames)
    #pyplot.xticks()                         
    pyplot.colorbar(cmap='Blues')
    #self.fig.tight_layout()
    pyplot.savefig(outfile)
    pyplot.clf()

with open(system_vectors_in,'r',encoding='utf-8') as file_in:
    lines = file_in.read().strip().split('\n')

systems = []
systems_performance = []
for line in lines:
    tokens = line.split('\t')
    systems.append(tokens[0])
    systems_performance.append([float(x) for x in tokens[1].split(',')])

pyplot.pcolor(systems_performance, norm=None, cmap='Blues')
pyplot.gca().invert_yaxis()        
pyplot.yticks(numpy.arange(len(systems))+0.5, systems, fontsize=7)
pyplot.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    right=False,      # ticks along the bottom edge are off
    left=False)
pyplot.colorbar(cmap='Blues')
pyplot.savefig(fig_out)
pyplot.clf()

# distance_matrix = generate_system_distance_matrix(systems_performance)
# graph = networkx.from_numpy_matrix(distance_matrix)
# pos = networkx.spectral_layout(graph)
# for key in pos.keys():
#     position = pos[key]
#     pyplot.scatter(position[0], position[1], marker=r'$ {} $'.format(systems[key]), s=700, c=(position[1]/10.0, 0, 1 - position[1]/10.0), edgecolor='None')
#     pyplot.tight_layout()
#     # pyplot.axis('equal')
#     pyplot.savefig(fig_out)           
# pyplot.clf()