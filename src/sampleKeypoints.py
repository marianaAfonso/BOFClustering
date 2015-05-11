import numpy as np
import math
import random

class SamplingImandKey:
    
    def __init__(self, _n_images, _number_of_kp, _average_words, _image_percentage = 0.3):
        self.n_images = _n_images
        self.number_of_kp = _number_of_kp
        self.average_words = _average_words
        self.image_percentage = _image_percentage

    def sampleKeypoints(self, des_vector):
        
        num_im_to_sample = int(self.image_percentage*self.n_images)
        
        im_picked = random.sample(range(0,self.n_images), num_im_to_sample)
        
        des_vector_sampled = []
        
        for i in im_picked:
            num_kp = self.number_of_kp[i]
            
            if num_kp<0.1*self.average_words:
                num_kp_to_sample = num_kp
            
            elif num_kp>1.8*self.average_words:
                num_kp_to_sample = 0.1*num_kp
                
            else:
                num_kp_to_sample = (-1.0/(2*self.average_words))*num_kp + 1
                num_kp_to_sample = num_kp_to_sample*num_kp
            
            num_kp_to_sample = math.ceil(num_kp_to_sample)
                
            kp_picked = random.sample(range(sum(self.number_of_kp[:i]),sum(self.number_of_kp[:i])+self.number_of_kp[i]), int(num_kp_to_sample))
            
            #print num_kp_to_sample
            
            for j in kp_picked:
                des_vector_sampled.append(des_vector[j])
            
        return des_vector_sampled
    
    def writeFile(self, f):
        f.write("Sampling images and keypoints to reduce standard deviation of number of keypoints per image. Parameters: ")
        f.write("Image percentage = " + str(self.image_percentage) + " ")
        f.write("Equation used =  -1/2*avg * num_kp + 1")
        f.write("Limits of equation =  0.1*avg -> 100% and 1.8*avg -> 10% ")
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
#number_of_kp = [2,2,6]
#average_words = 3.33
#image_percentage = 0.4

#des_vector_new = sampleImageAndKeypoints(des_vector, n_images, number_of_kp, average_words, image_percentage)

#print des_vector_new