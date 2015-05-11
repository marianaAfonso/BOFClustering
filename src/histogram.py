def computeHist(projections, n_words, n_images, number_of_kp):
    #obtain the histograms of the BOF model
    vec = [[0 for x in range(n_words)] for x in range(n_images)] 
    
    j = 0
    
    for i in range(0,len(projections)):
        #print str(i)
        
        if i==sum(number_of_kp[:j+1]) and i!=0:
            j += 1
        
        vec[j][projections[i]] += 1
    
    return vec