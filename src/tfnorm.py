import numpy as np
import math

class Tfnorm:

   def normalizeHist(self, hist, n_words, n_images):  
      print 'Applying tf normalization...'
      
      new_hist = [[0 for x in range(n_words)] for x in range(n_images)] 
      
      hist_np = np.asarray(hist)
      
      for i in range(0,n_images):
         for j in range(0,n_words):
            nid = hist_np[i][j]
            nd = hist_np[i].sum()
            if nd!= 0:  
               new_hist[i][j] = float(nid)/nd
            else:
               new_hist[i][j] = 0
                  
      print 'tf normalization applied'
      
      return new_hist
   
   def writeFile(self, f):
      f.write("Histogram normalization type Tf Normalized \n")   
   
#counts = [[3, 0, 1], [2, 0, 0], [3, 0, 0],[4, 0, 0],[3, 2, 0], [3, 0, 2]]

#tdif = Tfnorm()
#norm = tdif.normalizeHist(counts,3,6)

#print norm
