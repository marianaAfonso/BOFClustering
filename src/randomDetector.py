import numpy as np
import cv2
import time
import random

class Random:
   def __init__(self, _numpatch=1000, _step=5):     
      self.numpatch = _numpatch
      self.step = _step

   def detectKp(self, img,mask):
      
      border = 40
      
      dense = cv2.FeatureDetector_create("Dense")
      dense.setInt('initXyStep',5)  
      dense.setInt('initImgBound',border)
      kp = dense.detect(img) 
      best_kp = random.sample(kp,self.numpatch)
         
      #draw the keypoints
      #img3 = cv2.drawKeypoints(img, best_kp, color=(255,0,0))
      #cv2.imshow('image2',img3)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()  
      
      return best_kp
   
   def writeParametersDet(self, f):
      f.write("Keypoint Detector RANDOM (DENSE) with parameters: ")
      f.write("XYStep = " + str(5) + " ")
      f.write('\n')         