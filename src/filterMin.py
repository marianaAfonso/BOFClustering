import simpleBinarization
import numpy as np

class WordFilterMin:
    def __init__(self, _minThres): 
        self.minThres = _minThres

    def applyFilter(self,hist,n_words,n_images):
        
        bi = simpleBinarization.SimpleBi()
        bi_hist = bi.normalizeHist(hist, n_words, n_images)
        bi_hist = np.array(bi_hist)
        
        words_freq = np.zeros(n_words)
        indexes_remove = []
        
        for i in range(0,n_words):
            word_freq = float(bi_hist[:,i].sum())/n_images
            if word_freq <= self.minThres:
                indexes_remove.append(i)
                
        new_hist = np.delete(np.array(hist), indexes_remove, axis=1)
        
        return new_hist
    
    def writeFile(self,f):
        f.write("Feature selection method Min with parameters: ")
        f.write("Min Threshold = " + str(self.minThres))
        f.write("\n")
    
    
#hist = [[1,2,0],[1,0,0],[1,2,4]]

#new_hist = applyMaxMinFilter(hist,3,3,0.8,0.2)

#print new_hist