import numpy as np
import cv2
import time
import random

class Fast:
   def __init__(self, _numpatch=1000, _equalnum=True, _threshold=17):     
      self.threshold = _threshold
      self.equalnum = _equalnum
      self.numpatch = _numpatch

   def detectKp(self, img, mask):
      
      if self.equalnum == True:
         # Initiate FAST object with default values
         fast = cv2.FastFeatureDetector(threshold = 15)
         best_kp = fast.detect(img,mask)  
         print len(best_kp)         
         if len(best_kp) >= self.numpatch:
            best_kp = random.sample(best_kp,self.numpatch)          
      else:
         fast = cv2.FastFeatureDetector(threshold = self.threshold)
         best_kp = fast.detect(img,mask) 
         
      #remove keypoints from the border of the image to be able to use the binary descriptors
      #border = 40 #size of the border
      #height, width = img.shape[:2]
      
      #best_kp_no_border = []   
      #for i in range(0,n):
         #kp = best_kp[i]
         #if kp.pt[0] < border or kp.pt[0] > (width - border) or kp.pt[1] < border or kp.pt[1] > (height - border):
            #True
         #else:
            #best_kp_no_border.append(kp)
         
      # draw the keypoints
      #img3 = cv2.drawKeypoints(img,best_kp_no_border, color=(255,0,0))
      #cv2.imshow('image2',img3)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()  
      
      return best_kp
   
   def writeParametersDet(self, f):
      if self.equalnum == True:
         f.write("Keypoint Detector FAST with parameters: ")
         f.write("Threshold = " + str(20) + " ")
      else:
         f.write("Keypoint Detector SURF with parameters: ")
         f.write("Threshold = " + str(self.threshold) + " ")
      f.write('\n')