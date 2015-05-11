import simpleBinarization
import numpy as np

class WordFilterMax:
    def __init__(self, _maxThres): 
        self.maxThres = _maxThres

    def applyFilter(self,hist,n_words,n_images):
        
        bi = simpleBinarization.SimpleBi()
        bi_hist = bi.normalizeHist(hist, n_words, n_images)
        bi_hist = np.array(bi_hist)
        
        words_freq = np.zeros(n_words)
        indexes_remove = []
        
        for i in range(0,n_words):
            word_freq = float(bi_hist[:,i].sum())/n_images
            if word_freq > self.maxThres:
                indexes_remove.append(i)
                
        new_hist = np.delete(np.array(hist), indexes_remove, axis=1)
        
        return new_hist
    
    def writeFile(self,f):
        f.write("Feature selection method Max with parameters: ")
        f.write("Max Threshold = " + str(self.maxThres))
        f.write("\n")
    
    
#hist = [[1,2,0],[1,0,0],[1,2,4]]

#w = WordFilterMax(0.8)

#new_hist = w.applyFilter(hist,3,3)

#print new_hist