import cv2
import numpy as np
import sys

class Orb:
   def __init__(self, _nfeatures=500, _equalnum=False, _patchsize=15):
      self.nfeatures = _nfeatures
      self.equalnum = _equalnum
      self.patchsize = _patchsize     

   def detectKp(self, img, mask):
      """ Compute the orb descriptors of a given image, returns kp,des
          where des are the descriptors and kp are the interest points """
   
      if self.equalnum == True:
         orb = cv2.ORB(nfeatures=self.nfeatures, patchSize=self.patchsize, edgeThreshold=self.patchsize)
         kp = orb.detect(img,mask)  
         best_kp = kp[:self.nfeatures]   
      else:
         orb = cv2.ORB(nfeatures=self.nfeatures, patchSize=self.patchsize, edgeThreshold=self.patchsize)
         best_kp = orb.detect(img,mask)     
   
      #Draw image with keypoints
      #img_key = cv2.drawKeypoints(img,best_kp)
      #cv2.imshow('image', img_key)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()   
   
      return best_kp
   
   def computeDes(self, img, kp):
      """ Compute the orb descriptors of a given image, returns kp,des
          where des are the descriptors and kp are the interest points """
   
      orb = cv2.ORB(nfeatures=self.nfeatures, patchSize=self.patchsize, edgeThreshold=self.patchsize)
   
      #Draw image with keypoints
      #img_key = cv2.drawKeypoints(img,kp)
      #cv2.imshow('image', img_key)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows() 
      
      len_kp_before = len(kp)
   
      kp,des = orb.compute(img,kp)
      
      len_kp_after = len(kp)
      
      if len_kp_before != len_kp_after:
         print "ERROR: NUMBER OF KP DIFFERENT AFTER DESCRIPTOR"
         sys.exit()        
   
      return des
   
   def writeParametersDes(self, f):
      f.write("Keypoint Descriptor ORB with parameters: ")
      f.write("None")
      f.write('\n')   
   
   def writeParametersDet(self, f):
      f.write("Keypoint Detector ORB with parameters: ")
      f.write("Max Features = " + str(self.nfeatures) + " ")
      f.write("Patch Size = " + str(self.patchsize))
      f.write('\n')     