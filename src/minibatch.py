import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

class MiniBatch:
   def __init__(self, _size = 500, _batch_size=300, ):     
      self.size = _size
      self.batch_size = _batch_size
   
   def obtainCodebook(self, sampled_x, x):

      print 'Obatining codebook using online k-means...'
      
      #normalize
      scaled_x_sampled = StandardScaler().fit_transform(sampled_x)
      scaled_x = StandardScaler().fit_transform(x)
      
      des_vector_suffled = scaled_x_sampled
       
      #shuffle list of descriptors
      np.random.shuffle(des_vector_suffled)
      
      minibatch = MiniBatchKMeans(n_clusters=self.size, init='k-means++', batch_size=self.batch_size, n_init=10, max_no_improvement=10, verbose=0, random_state=0)
      
      codebook = minibatch.fit(des_vector_suffled, y=None)
      
      #for n in range(0,len(des_vector_suffled)/batchsize+1):
         #if n!=len(des_vector_suffled)/batchsize:  
            #data = des_vector_suffled[n*batchsize:n*batchsize+batchsize]
         #else:
            #data = des_vector_suffled[n*batchsize:]
         #kmeans.partial_fit(data)
          
      projections = minibatch.predict(scaled_x)
      
      print 'Codebook obtained.'
      
      return codebook, projections
   
   def writeFileCodebook(self,f):
      f.write("Codebook construction method MiniBatch with parameters: ")
      f.write("Number of clusters = " + str(self.size) + " ")
      f.write("Batch size = " + str(self.batch_size) + " ")
      f.write('\n')    
