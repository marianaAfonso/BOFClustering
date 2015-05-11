import numpy as np
import scipy.cluster.vq 
import random

class RandomVectors:
   def __init__(self, _size = 500, ):     
      self.size = _size
   
   def obtainCodebook(self, sampled_x, x):
      
      codebook = random.sample(x,self.size)
      codebook = np.vstack(codebook)
      result = scipy.cluster.vq.vq(x,codebook)
      projections = result[0]      
      
      return codebook, projections
   
   def writeFileCodebook(self,f):
      f.write("Codebook construction method Random from Feature Vectors with parameters: ")
      f.write("Number of clusters = " + str(self.size) + " ")
      f.write('\n')    