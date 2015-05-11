from sklearn.cluster import Birch as BIRCH
from sklearn.preprocessing import StandardScaler
import numpy as np

class Birch:
   def __init__(self, _nclusters = None, _branching_factor=50, _threshold=0.2):
      self.nclusters = _nclusters
      self.branching_factor = _branching_factor
      self.threshold = _threshold
     
   def obtainClusters(self, hist):

      print 'Obatining clusters using Birch from skilean...'
   
      hist = np.array(hist)
      hist = hist.astype(float)      
      scaled_vec = StandardScaler().fit_transform(hist)
      
      brc = BIRCH(branching_factor=self.branching_factor, n_clusters=self.nclusters, threshold=self.threshold, compute_labels=True)
      
      #obatin the codebook and the projections of the images on the codebook (clusters of words)
      codebook = brc.fit(scaled_vec)
      clusters = brc.predict(scaled_vec)
      
      print 'Clusters obtained.'
      
      return clusters

   def obtainCodebook(self, sampled_x, x):

      print 'Obatining codebook using Birch from skilean...'
   
      scaled_x_sampled = StandardScaler().fit_transform(sampled_x)
      scaled_x = StandardScaler().fit_transform(x)
      
      brc = BIRCH(branching_factor=self.branching_factor, n_clusters=self.nclusters, threshold=self.threshold, compute_labels=True)
      
      #obatin the codebook and the projections of the images on the codebook (clusters of words)
      codebook = brc.fit(scaled_x_sampled)
      clusters = brc.predict(scaled_x)
      
      print 'Clusters obtained.'
      
      return codebook, clusters 
   
   def writeFileCodebook(self,f):
      f.write("Codebook construction method BIRCH with parameters: ")
      f.write("Number of clusters = " + str(self.nclusters) + " ")
      f.write("Branching factor = " + str(self.branching_factor) + " ")
      f.write("Threshold = " + str(self.threshold) + " ")
      f.write('\n')    
      
   def writeFileCluster(self,f):
      f.write("Clustering algorithm BIRCH with parameters: ")
      f.write("Number of clusters = " + str(self.nclusters) + " ")
      f.write("Branching factor = " + str(self.branching_factor) + " ")
      f.write("Threshold = " + str(self.threshold) + " ")
      f.write('\n')     