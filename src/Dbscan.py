from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import scipy.spatial.distance as scipyd
import numpy as np
import random
import itertools

class Dbscan:
   def __init__(self, _dist="cosine", _eps_proportion=0.25, _min_samples=4):
      self.dist = _dist
      self.eps_proportion = _eps_proportion
      self.min_samples = _min_samples
     
   def obtainClusters(self, hist):

      print 'Applying DBSCAN clustering algorithm...'
      
      hist = np.array(hist)
      hist = hist.astype(float)
      scaled_hist = StandardScaler().fit_transform(hist)
      
      dist_matrix = scipyd.cdist(scaled_hist, scaled_hist, self.dist)
      eps_value = self.eps_proportion*dist_matrix.max()
      
      db = DBSCAN(eps=eps_value, metric = "precomputed", min_samples=self.min_samples).fit(dist_matrix)
      
      clusters = db.labels_
      
      #to make the noise points to random numbers instead of -1
      #n_clusters = clusters.max()+1
      #n_images = len(hist)
      #indexes_noise = np.where(clusters==-1)[0]
      #random_numbers = random.sample(range(n_clusters,n_clusters+len(indexes_noise)), len(indexes_noise))
      #for i,r in itertools.izip(indexes_noise,random_numbers):
         #clusters[i] = r    
      
      print 'DBSCAN clustering algorithm applied'
      
      return clusters
   
   def writeFileCodebook(self,f):
      f.write("Codebook construction method DBSCAN with parameters: ")
      f.write("EPS proportion = " + str(self.eps_proportion) + " ")
      f.write("Min samples = " + str(self.min_samples) + " ")
      f.write('\n')   
      
   def writeFileCluster(self,f):
      f.write("Clustering algorithm DBSCAN with parameters: ")
      f.write("EPS proportion = " + str(self.eps_proportion) + " ")
      f.write("Min samples = " + str(self.min_samples) + " ")
      f.write('\n')    