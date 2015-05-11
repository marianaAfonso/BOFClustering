from sklearn.cluster import MeanShift as MEANSHIFT, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
import numpy as np

class MeanSift:
   def __init__(self, _nclusters = None):
      self.nclusters = _nclusters
     
   def obtainClusters(self, hist):

      print 'Obatining clusters using MeanShift from skilean...'
      
      hist = np.array(hist)
      hist = hist.astype(float)      
      scaled_vec = StandardScaler().fit_transform(hist)
      
      bandwidth = estimate_bandwidth(scaled_vec, quantile=0.3)
      ms = MEANSHIFT(bandwidth=bandwidth, bin_seeding=True)

      clusters = ms.fit_predict(scaled_vec)

      print 'Clusters obtained using MeanShift'
      
      return clusters
   
   def writeFileCluster(self,f):
      f.write("Clustering algorithm MeanShift with parameters: ")
      f.write("Number of clusters = " + str(self.nclusters) + " ")
      f.write('\n') 
      
      
      
      
   
   
