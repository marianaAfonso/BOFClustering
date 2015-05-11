import numpy as np
import math
from sklearn.feature_extraction.text import TfidfTransformer
import simpleBinarization

class TfIdfnorm:

   def normalizeHist(self, hist, n_words, n_images):  
      print 'Applying tf-idf transformation...'
      
      new_hist = [[0 for x in range(n_words)] for x in range(n_images)] 
      new_new_hist = [[0 for x in range(n_words)] for x in range(n_images)] 
      
      bi = simpleBinarization.SimpleBi()
      bi_hist = bi.normalizeHist(hist, n_words, n_images)
      
      N = n_images
      
      for i in range(0,n_images):
         for j in range(0,n_words):
            nid = hist[i][j]
            colunm_bi_hist = [row[j] for row in bi_hist]
            nt = sum(colunm_bi_hist)
            if nt!= 0:  
               new_hist[i][j] = (float(nid))*math.log(1+float(N)/nt)
            else:
               new_hist[i][j] = 0
      
      for i in range(0,n_images):
         for j in range(0,n_words):
            tfidf = new_hist[i][j]
            sum_line = sum(new_hist[i])
            if sum_line!= 0:  
               new_new_hist[i][j] = float(tfidf)/sum_line
            else:
               new_hist[i][j] = 0      
                  
      print 'tf-idf applied'
      
      return new_new_hist
   
   def writeFile(self, f):
      f.write("Histogram normalization type Tf-Idf Normalized \n")   
   

#counts = [[3, 0, 1], [2, 0, 0], [3, 0, 0],[4, 0, 0],[3, 2, 0], [3, 0, 2]]

#tdif = TfIdfnorm()
#norm1,norm2 = tdif.normalizeHist(counts,3,6)

#print norm1
#print norm2