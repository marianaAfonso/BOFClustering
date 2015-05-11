import scipy.cluster.hierarchy as scipyH
import scipy.cluster.vq as vq
import scipy.spatial as sciSp
import numpy as np
import matplotlib.pylab as plt

class HierarchicalScipy:
   def __init__(self, _dist, _linkage="average", _stop_method="distance", _proportion_dist=0.4):
      #goooooooood with _proportion_dist=0.3, 0.4 and cosine difference
      self.dist = _dist
      self.linkage = _linkage
      self.stop_method = _stop_method
      self.proportion_dist = _proportion_dist
     
   def obtainClusters(self, hist):

      print 'Obatining clusters using Hierarchical Clustering from Scipy...'
   
      scaled_hist = vq.whiten(hist)
      
      d = sciSp.distance.pdist(scaled_hist, self.dist)
      
      Z = scipyH.linkage(d, method=self.linkage)
      
      #obatin the clusters
      clusters = scipyH.fcluster(Z, self.proportion_dist*d.max(), self.stop_method)
      
      print 'Clusters obtained.'  
      #scipyH.dendrogram(Z)
      #plt.show()
      return clusters

   def obtainCodebook(self, hist):

      print 'Obatining clusters using Agglomerative Clustering from skilean...'
   
      scaled_vec = StandardScaler().fit_transform(hist)
      
      # connectivity matrix for structured Ward
      connectivity = kneighbors_graph(scaled_vec, n_neighbors=10, include_self=False)
      # make connectivity symmetric
      connectivity = 0.5 * (connectivity + connectivity.T)      
      
      hc = AgglomerativeClustering(n_clusters=self.nclusters, linkage=self.linkage, connectivity=connectivity, compute_full_tree=False, affinity=self.dist)
      
      #obatin the codebook and the projections of the images on the codebook (clusters of words)
      clusters = hc.fit_predict(scaled_vec,None)
      
      print 'Clusters obtained.'
      
      return None, clusters
   
   def writeFileCodebook(self,f):
      f.write("Codebook construction method Hierarchical from Scipy with parameters: ")
      f.write("Distance = " + str(self.dist) + " ")
      f.write("Linkage method = " + str(self.linkage) + " ")
      f.write('\n')    
         
   def writeFileCluster(self,f):
      f.write("Clustering algorithm Hierarchical from Scipy with parameters: ")
      f.write("Distance = " + str(self.dist) + " ")
      f.write("Linkage method = " + str(self.linkage) + " ")
      f.write("Stop method = " + str(self.stop_method) + " ")
      f.write("ProportionDist = " + str(self.proportion_dist) + " ")      
      f.write('\n')          

#mat = np.array([[1, 0.5, 0.9],[0.5, 1, -0.5],[0.9, -0.5, 1]])

#h = HierarchicalScipy()
#h.obtainClusters(mat)