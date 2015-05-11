import numpy as np
import math
import simpleBinarization

class Okapi:
   
   def __init__(self, _average_words):
      self.average_words = _average_words
        
   def normalizeHist(self, hist, n_words, n_images):  
      print 'Applying Okapi transformation...'
      
      k1 = 1.2
      b = 0.75
      
      new_hist = [[0 for x in range(n_words)] for x in range(n_images)] 
      
      bi = simpleBinarization.SimpleBi()
      bi_hist = bi.normalizeHist(hist, n_words, n_images)
      bi_hist = np.asarray(bi_hist)
      
      hist_np = np.asarray(hist)
         
      N = n_images
      
      for i in range(0,n_images):
         for j in range(0,n_words):
            nid = hist[i][j]
            nd = hist_np[i].sum()
            nt = bi_hist[:, j].sum()
            tf = float(k1*nid)/(nid+k1*(1-b+b*(nd/self.average_words)))
            idf = math.log(float(N-nt+0.5)/(nt+0.5))
            if idf<0:
               new_hist[i][j]=0
            else:
               new_hist[i][j] = tf*idf
                  
      print 'Okapi method applied'
      
      return new_hist
   
   def writeFile(self, f):
      f.write("Histogram normalization type Okapi Tf-Idf \n")   
   
#counts = [[3, 0, 1], [2, 0, 0], [3, 0, 0],[4, 0, 0],[3, 2, 0], [3, 0, 2]]

#tdif = Okapi(3)
#norm = tdif.normalizeHist(counts,3,6)

#print norm