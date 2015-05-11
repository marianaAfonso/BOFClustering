import numpy as np
import math
from sklearn.feature_extraction.text import TfidfTransformer

class TfIdf:

   def normalizeHist(self, hist, n_words, n_images):  
      print 'Applying tf-idf transformation...'
      
      new_hist = [[0 for x in range(n_words)] for x in range(n_images)] 
      
      hist_np = np.asarray(hist)
      
      N = hist_np.sum()
      
      for i in range(0,n_images):
         for j in range(0,n_words):
            nid = hist_np[i][j]
            nd = hist_np[i].sum()
            ni = hist_np[:, j].sum()
            if ni!= 0:  
               new_hist[i][j] = (float(nid)/nd)*math.log(float(N)/ni)
            else:
               new_hist[i][j] = 0
                  
      print 'tf-idf applied'
      
      return new_hist
   
   def writeFile(self, f):
      f.write("Histogram normalization type Tf-Idf 1 \n")   

#counts = [[3, 0, 1], [2, 0, 0], [3, 0, 0],[4, 0, 0],[3, 2, 0], [3, 0, 2]]

#tdif = TfIdf()
#norm = tdif.normalizeHist(counts,3,6)

#print norm

#transformer = TfidfTransformer(norm=None)
#tfidf = transformer.fit_transform(counts)

#print tfidf.toarray()