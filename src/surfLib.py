import cv2
import numpy as np
import sys
import sys

class Surf:
   def __init__(self, _numpatch=1000, _equalnum=True, _hessianThreshold=450):     
      self.hessianThreshold = _hessianThreshold
      self.equalnum = _equalnum
      self.numpatch = _numpatch

   def detectKp(self, img, mask):
      """ Compute the sift descriptors of a given image, returns kp,des
          where des are the descriptors and kp are the interest points """
   
      if self.equalnum == True:  
         surf = cv2.SURF(hessianThreshold = 0)
         kp = surf.detect(img,mask)
         #get only the best #numpatch keypoints 
         best_kp = kp[:self.numpatch]
      else:
         surf = cv2.SURF(hessianThreshold = self.hessianThreshold)
         best_kp = surf.detect(img,mask)         
      
      #remove keypoints from the border of the image to be able to use the binary descriptors
      #border = 40 #size of the border
      #height, width = img.shape[:2]
      
      #n = len(best_kp)
      
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
   
   def computeDes(self, img, kp):
      """ Compute the sift descriptors of a given image, returns kp,des
          where des are the descriptors and kp are the interest points """
   
      if self.equalnum == True:  
         surf = cv2.SURF(hessianThreshold = 0)
      else:
         surf = cv2.SURF(hessianThreshold = self.hessianThreshold)
         
      #Draw image with keypoints
      #img_key = cv2.drawKeypoints(gray,kp)
      #cv2.imshow('image', img_key)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows() 
      
      len_kp_before = len(kp)
   
      kp,des = surf.compute(img,kp)
      
      len_kp_after = len(kp)
            
      if len_kp_before != len_kp_after:
         print "ERROR: NUMBER OF KP DIFFERENT AFTER DESCRIPTOR"
         sys.exit()         
   
      return des

   def writeParametersDet(self, f):
      if self.equalnum == True:
         f.write("Keypoint Detector SURF with parameters: ")
         f.write("Hessian Threshold = " + str(0) + " ")
      else:
         f.write("Keypoint Detector SURF with parameters: ")
         f.write("Hessian Threshold = " + str(self.hessianThreshold) + " ")
      f.write('\n')
   
   def writeParametersDes(self, f):
      f.write("Keypoint Descriptor SURF with parameters: ")
      f.write("DEFAULT")
      f.write('\n')     

   def matchSift(self, filename1, filname2, kp1, des1, kp2, des2):
      """ Match the sift descriptors from two images
          Returns the matches found """
      
      img1 = cv2.imread(filename1,0)       
      img2 = cv2.imread(filename2,0) 
   
      # BFMatcher with default params
      bf = cv2.BFMatcher()
      matches = bf.match(des1,des2)
   
      matches = sorted(matches, key=lambda val: val.distance)
   
      return matches
   
   def drawMatches(self, img1, kp1, img2, kp2, matches):
      """  This function takes in two images with their associated 
      keypoints, as well as a list of DMatch data structure (matches) 
      that contains which keypoints matched in which images.
   
      An image will be produced where a montage is shown with
      the first image followed by the second image beside it.
   
      Keypoints are delineated with circles, while lines are connected
      between matching keypoints.
   
      img1,img2 - Grayscale images
      kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
                detection algorithms
      matches - A list of matches of corresponding keypoints through any
                OpenCV keypoint matching algorithm    """
   
      # Create a new output image that concatenates the two images together
      # (a.k.a) a montage
      rows1 = img1.shape[0]
      cols1 = img1.shape[1]
      rows2 = img2.shape[0]
      cols2 = img2.shape[1]
   
      out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
   
      # Place the first image to the left
      out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])
   
      # Place the next image to the right of it
      out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])
   
      # For each pair of points we have between both images
      # draw circles, then connect a line between them
      for mat in matches:
   
         # Get the matching keypoints for each of the images
         img1_idx = mat.queryIdx
         img2_idx = mat.trainIdx
      
         # x - columns
         # y - rows
         (x1,y1) = kp1[img1_idx].pt
         (x2,y2) = kp2[img2_idx].pt
      
         # Draw a small circle at both co-ordinates
         # radius 4
         # colour blue
         # thickness = 1
         cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
         cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
      
         # Draw a line in between the two points
         # thickness = 1
         # colour blue
         cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
   
      # Show the image
      cv2.imshow('Matched Features', out)
      cv2.waitKey(0)
      cv2.destroyAllWindows()


#NOTAS

#Reference to functions: http://docs.opencv.org/trunk/modules/nonfree/doc/feature_detection.html
