from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import numpy as np

class Hierarchical:
   def __init__(self, _nclusters, _dist="euclidean", _linkage="average"):
      self.nclusters = _nclusters
      self.dist = _dist
      self.linkage = _linkage
     
   def obtainClusters(self, hist):

      print 'Obatining clusters using Agglomerative Clustering from skilean...'
      
      hist = np.array(hist)
      hist = hist.astype(float)         
      scaled_vec = StandardScaler().fit_transform(hist)
      
      hc = AgglomerativeClustering(n_clusters=self.nclusters, linkage=self.linkage, affinity=self.dist)
      
      #obatin the clusters
      clusters = hc.fit_predict(scaled_vec,None)
      
      print 'Clusters obtained.'
      
      return clusters

   def obtainCodebook(self, hist):

      print 'Obatining clusters using Agglomerative Clustering from skilean...'
   
      scaled_vec = StandardScaler().fit_transform(hist)
      
      # connectivity matrix for structured Ward
      connectivity = kneighbors_graph(scaled_vec, n_neighbors=3, include_self=False)
      # make connectivity symmetric
      connectivity = 0.5 * (connectivity + connectivity.T)      
      
      hc = AgglomerativeClustering(n_clusters=self.nclusters, linkage=self.linkage, connectivity=connectivity, compute_full_tree=False, affinity=self.dist)
      
      #obatin the codebook and the projections of the images on the codebook (clusters of words)
      clusters = hc.fit_predict(scaled_vec,None)
      
      print 'Clusters obtained.'
      
      return None, clusters
   
   def writeFileCodebook(self,f):
      f.write("Codebook construction method Hierarchical from Sklearn with parameters: ")
      f.write("Number of clusters = " + str(self.nclusters) + " ")
      f.write("Distance = " + str(self.dist) + " ")
      f.write("Linkage method = " + str(self.linkage) + " ")
      f.write('\n')    
         
   def writeFileCluster(self,f):
      f.write("Clustering algorithm Hierarchical from Sklearn with parameters: ")
      f.write("Number of clusters = " + str(self.nclusters) + " ")
      f.write("Distance = " + str(self.dist) + " ")
      f.write("Linkage method = " + str(self.linkage) + " ")
      f.write('\n')        
