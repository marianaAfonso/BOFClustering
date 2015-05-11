import numpy as np
import scipy.cluster.vq 

class AllRandom:
   def __init__(self, _size, _high=5):     
      self.high = _high
      self.size = _size
   
   def obtainCodebook(self, x):
      
      codebook = np.random.randint(self.high, size=(self.size,x.shape[1]))
      result = scipy.cluster.vq.vq(x,codebook)
      projections = result[0]      
      
      return codebook, projections
   
   def writeFileCodebook(self,f):
      f.write("Codebook construction method All Random with parameters: ")
      f.write("Number of clusters = " + str(self.size) + " ")
      f.write("Max value for each feature = " + str(self.high) + " ")
      f.write('\n')       