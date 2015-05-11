import numpy as np
import cv2
import time
import random

class Star:
   def __init__(self, _numpatch=1000, _equalnum=True, _threshold=0):     
      self.threshold = _threshold
      self.equalnum = _equalnum
      self.numpatch = _numpatch

   def detectKp(self, img, mask):
      
      if self.equalnum == True:
         # Star detector
         star = cv2.StarDetector(16, 0, 10, 8, 5)         
         best_kp = star.detect(img)  
         if len(best_kp) >= self.numpatch:
            best_kp = random.sample(best_kp,self.numpatch)         
      else:
         star = cv2.StarDetector(30, 0, 10, 8, 5)    
         best_kp = star.detect(img) 
         
      #remove keypoints from the border of the image to be able to use the binary descriptors
      border = 40 #size of the border
      height, width = img.shape[:2]
      
      n = len(best_kp)
      
      best_kp_no_border = []
         
      for i in range(0,n):
         kp = best_kp[i]
         if kp.pt[0] < border or kp.pt[0] > (width - border) or kp.pt[1] < border or kp.pt[1] > (height - border):
            True
         else:
            best_kp_no_border.append(kp)
      best_kp = best_kp_no_border
         
      # draw the keypoints
      #img3 = cv2.drawKeypoints(img,best_kp_no_border, color=(255,0,0))
      #cv2.imshow('image2',img3)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()  
      
      return best_kp 
   
   def writeParametersDet(self, f):
      if self.equalnum == True:
         f.write("Keypoint Detector STAR with parameters: ")
         f.write("Threshold = " + str(0) + " ")
      else:
         f.write("Keypoint Detector STAR with parameters: ")
         f.write("Threshold = " + str(self.threshold) + " ")
      f.write('\n')      