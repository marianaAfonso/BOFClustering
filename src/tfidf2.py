import numpy as np
import math
from sklearn.feature_extraction.text import TfidfTransformer
import simpleBinarization

class TfIdf2:

   def normalizeHist(self, hist, n_words, n_images):  
      print 'Applying tf-idf transformation...'
      
      new_hist = [[0 for x in range(n_words)] for x in range(n_images)] 
      
      bi = simpleBinarization.SimpleBi()
      bi_hist = bi.normalizeHist(hist, n_words, n_images)
      bi_hist = np.asarray(bi_hist)
      
      N = n_images
      
      for i in range(0,n_images):
         for j in range(0,n_words):
            nid = hist[i][j]
            nt = bi_hist[:, j].sum()
            if nt!= 0:  
               new_hist[i][j] = (float(nid))*math.log(1+float(N)/nt)
            else:
               new_hist[i][j] = 0
                  
      print 'tf-idf applied'
      
      return new_hist
   
   def writeFile(self, f):
      f.write("Histogram normalization type Tf-Idf 2 \n")   

#counts = [[3, 0, 1], [2, 0, 0], [3, 0, 0],[4, 0, 0],[3, 2, 0], [3, 0, 2]]

#tdif = TfIdf2()
#norm = tdif.normalizeHist(counts,3,6)

#print norm

##transformer = TfidfTransformer(norm=None)
##tfidf = transformer.fit_transform(counts)

##print tfidf.toarray()