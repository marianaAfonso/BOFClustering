import numpy as np
import math
import random

class SamplingAllKey:
    
    def __init__(self, _kpPercentage = 0.3):
        self.kpPercentage = _kpPercentage

    def sampleKeypoints(self, des_vector):
        
        num_kp_to_sample = int(math.ceil(len(des_vector)*self.kpPercentage))
        
        kp_picked = random.sample(range(0,len(des_vector)), num_kp_to_sample)
        
        des_vector_sampled = []
        
        for kp in kp_picked:
            des_vector_sampled.append(des_vector[kp])
            
        return des_vector_sampled
    
    def writeFile(self, f):
        f.write("Sampling keypoints randomly. Parameters: ")
        f.write("Keypoint Percentage = " + str(self.kpPercentage) + " ")
        f.write("\n")    

#des_vector = []
#des_vector.append([1,2,3])
#des_vector.append([0,0,0])
#des_vector.append([4,5,6])
#des_vector.append([2,2,2])
#des_vector.append([7,8,9])
#des_vector.append([10,11,12])
#des_vector.append([13,14,15])
#des_vector.append([1,1,1])
#des_vector.append([1,1,1])
#des_vector.append([1,1,1])
#des_vector.append([1,1,1])

#n_images = 3
#kp_percentage = 0.5

#des_vector_new = sampleImageAndKeypoints(des_vector, kp_percentage)

#print des_vector_new