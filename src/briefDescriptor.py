import numpy as np
import cv2
import time
import random
import sys

class Brief:
   def __init__(self, _np=32):     
      self.np = _np

   def computeDes(self, img, kp):
      
      len_kp_before = len(kp)
      
      #print len(kp)
      
      brief = cv2.DescriptorExtractor_create("BRIEF")
      brief.setInt('bytes',self.np)
      kp, des = brief.compute(img, kp)
      
      len_kp_after = len(kp)
      
      if len_kp_before != len_kp_after:
         print "ERROR: NUMBER OF KP DIFFERENT AFTER DESCRIPTOR"
         sys.exit()     
      
      return des
   
   def writeParametersDes(self, f):
      f.write("Keypoint Descriptor BRIEF with parameters: ")
      f.write("Bytes = " + str(self.np) + " ")
      f.write('\n')   