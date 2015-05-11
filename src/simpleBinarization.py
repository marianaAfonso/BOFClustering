import numpy as np
import math

class SimpleBi:

   def normalizeHist(self, hist, n_words, n_images):  
      print 'Applying Binarization to the histogram...'
      
      new_hist = [[0 for x in range(n_words)] for x in range(n_images)] 
      
      for i in range(0,n_images):
         for j in range(0,n_words):
            
            if hist[i][j] > 0:
               new_hist[i][j] = 1
            else:
               new_hist[i][j] = 0
                  
      print 'Binarization applied'
      
      return new_hist
   
   def writeFile(self, f):
      f.write("Histogram normalization type Simple Binarization \n")