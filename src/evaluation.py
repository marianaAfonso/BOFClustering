import numpy as np
import os
from sklearn import metrics

def get_imlist(path):
    """    Returns a list of filenames for 
        all files in a directory (they must be images) """
        
    return [f[:-4] for f in os.listdir(path)]

pathImages = '/Users/Mariana/mieec/Tese/Development/ImageDatabases/Graz-01_sample'

imList = get_imlist(pathImages)

print imList

labels = []
class_names = []

for im in imList:
    
    if im in '.DS_Store':
        continue
    else:
        
        name = im.split("_")[0]
        
        if name in class_names:
            index = class_names.index(name)
            labels.append(index)
        else:
            class_names.append(name)
            index = class_names.index(name)
            labels.append(index)            
            print im
            print name
                                   
clusters = np.loadtxt('saveClusters_Apr-06-1010PM-2015_.txt', dtype=int, delimiter=',')

if len(clusters) == len(labels):
    
    rand_index = metrics.adjusted_rand_score(labels, clusters)
    print 'rand_index = ' + str(rand_index)
        
    NMI_index = metrics.normalized_mutual_info_score(labels, clusters)
    print 'NMI_index = ' + str(NMI_index)