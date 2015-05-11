import numpy as np
import scipy.cluster.vq 
from sklearn.metrics import silhouette_samples, silhouette_score

class KMeans1:
   def __init__(self, _size = 500):     
      self.size = _size
   
   def obtainCodebook(self, x_sampled, x):
      
      print 'Applying Kmeans clustering...'
      
      #scale each feature before clustering
      scaled_x_sampled = scipy.cluster.vq.whiten(x_sampled)
      scaled_x = scipy.cluster.vq.whiten(x)
      
      #kmeans clustering
      centers = scipy.cluster.vq.kmeans(scaled_x_sampled, self.size, iter=10, thresh=1e-05)
      
      #obatin the projections of the images on the codebook (clusters of words)
      result = scipy.cluster.vq.vq(scaled_x,centers[0])
      projections = result[0]
      
      print 'Kmeans clustering applied'
      
      return centers, projections
   
   def obtainClusters(self, x):
      
      print 'Applying Kmeans clustering...'
      
      #scale each feature before clustering
      scaled_x = scipy.cluster.vq.whiten(x)
      
      projections = np.zeros((self.size[-1]+1, len(x)))
      silhouette_avg = np.ones(self.size[-1]+1)
      silhouette_avg = np.multiply(silhouette_avg,-1000)
      
      #print str(silhouette_avg)
      
      for n_clusters in self.size:      
         #kmeans clustering
         centers = scipy.cluster.vq.kmeans(scaled_x, n_clusters, iter=10, thresh=1e-05)
         
         #print centers
      
         #obatin the projections of the images on the codebook (clusters of words)
         result = scipy.cluster.vq.vq(scaled_x,centers[0])
         projections[n_clusters] = result[0]
         
         #print 'clustering result = ' + str(projections[n_clusters])
         
         silhouette_avg[n_clusters] = silhouette_score(scaled_x,  projections[n_clusters])
         #print "For n_clusters = " +str(n_clusters) + " The average silhouette_score is : " + str(silhouette_avg[n_clusters])
         
         ## Compute the silhouette scores for each sample
         #sample_silhouette_values = silhouette_samples(scaled_x,  projections[n_clusters])
         
         #ith_cluster_max = np.zeros(n_clusters)
         
         #for i in range(n_clusters):
         ## Aggregate the silhouette scores for samples belonging to
         ## cluster i, and sort them
            #ith_cluster_silhouette_values = sample_silhouette_values[ projections[n_clusters] == i]
            #ith_cluster_max[i] = ith_cluster_silhouette_values.max()
            
            ##print 'max value of silhouette for cluster ' + str(i) + ' = ' + str(ith_cluster_max[i])
         
      
      #print silhouette_avg
      #print silhouette_avg.shape
      index_max = np.where(silhouette_avg==silhouette_avg.max())[0][0]
      
      #print index_max
      
      print 'Kmeans clustering applied'
      
      return projections[index_max]   
   
   def writeFileCodebook(self,f):
      f.write("Codebook construction method Kmeans with parameters: ")
      f.write("Number of clusters (k) = " + str(self.size) + " ")
      f.write('\n')
      
   def writeFileCluster(self,f):
      f.write("Clustering algorithm Kmeans with parameters: ")
      f.write("Number of clusters (k) = " + str(self.size) + " ")
      f.write('\n')   
      
      