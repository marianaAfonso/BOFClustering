#!/usr/bin/python

import sys, getopt
import os
import cv2
import numpy as np
import siftLib
import surfLib
import fastDetector
import starDetector
import randomDetector
import orbLib
import briefDescriptor
import freakDescriptor
import KMeans1
import histogram
import tfidf
import tfidf2
import tfnorm
import tfidfnorm
import time
import datetime
import Dbscan
import Birch
import hierarchicalClustering
import hierarchicalClustScipy
import minibatch
import meanSift
import randomSamplesBook
import allrandom
from sklearn import metrics
import simpleBinarization
import filterMin
import filterMax
import filterMaxMin
import okapi
import sampleKeypoints
import sampleAllKeypoints
#import warnings
import statistics

import xlsxwriter

def main(argv):
   
   #optional argument rep (reptitions) - default value 1
   rep = 1
   
   try:
      opts, args = getopt.getopt(argv,"p:k:n:s:d:g:c:t:f:h:a:m:r:",["help", "path=","keypnt=", "numpatch=", "equalnum=", "imdes=", "imkeysample=", "codebook=", "size=", "fselec=", "histnorm=","clust=","dist=","rep="])
   except getopt.GetoptError:
      print 'ERROR'
      print 'main.py -p <path for images> -k <keypoint detector> -n <number of patches> -s <equal number of patches per image> -d <descriptor> -g <Sampling method> -c <codebook method> -t <size of codebook> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure>'
      sys.exit(2)
   for opt, arg in opts:
      if opt in "help":
         print opt
         print 'main.py -p <path for images> -k <keypoint detector> -n <number of patches> -s <equal number of patches per image> -d <descriptor> -g <Sampling method> -c <codebook method> -t <size of codebook> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -m <distance measure> -r <number of repetitions>'
         sys.exit()
      elif opt in ("-p", "--path"):
         pathImages = arg      
      elif opt in ("-k", "--keypnt"):
         keypnt = arg
      elif opt in ("-n", "--numpatch"):
         numpatch = int(arg)
      elif opt in ("-s", "--equalnum"):
         if arg == 'True':
            equalnum = True
         else:
            equalnum = False
      elif opt in ("-d", "--imdes"):
         imdes = arg
      elif opt in ("-g", "--imkeysample"):
         arg_split = arg.split(":")
         imsample = arg_split[0]
         percentage = float(arg_split[1])
      elif opt in ("-c", "--codebook"):
         codebook = arg           
      elif opt in ("-t", "--size"):
         size = int(arg)     
      elif opt in ("-h", "--histnorm"):
         histnorm = arg
      elif opt in ("-f", "--fselec"):
         arg_split = arg.split(":") 
         fselec = arg_split[0] 
         fselec_perc = [1, 0]
         if fselec == 'FMAX':
            fselec_perc[0] = float(arg_split[1])
         elif fselec == 'FMIN':
            fselec_perc[1] = float(arg_split[1])
         elif fselec == 'FMAXMIN':
            fselec_perc[0] = float(arg_split[1])
            fselec_perc[1] = float(arg_split[2])
      elif opt in ("-a", "--clust"):
         clust = arg
      elif opt in ("-m", "--dist"):
         dist = arg
      elif opt in ("-r", "--rep"):
         rep = int(arg)            
   
   print '\n#############################\n   Arguments for testing\n#############################\n'
   print 'Image database path: ' + pathImages + '\n'
   print '1.1) Keypoint detector: ' + keypnt
   print '1.2) Number of patches: ' + str(numpatch)
   print '1.3) Same or different number of pacthes per image: ' + str(equalnum)
   print '2) Descriptor: ' + imdes
   print '3) Image and keypoint sampling method: ' + str(imsample)
   print '      Percentage of images/keypoints: ' + str(percentage)
   print '4.1) Codebook construction algorithm: ' + codebook
   print '4.2) Size of codebook: ' + str(size)
   print '5) Feature selection: ' + fselec
   print '      Max threshold for visual words filtering: ' + str(fselec_perc[0]) 
   print '      Min threshold for visual words filtering: ' + str(fselec_perc[1]) 
   print '6) Histogram normalization: ' + histnorm
   print '7.1) Clustering algorithm: ' + clust      
   print '7.2) Distance measure: ' + dist
   
   print 'Repetitions = ' + str(rep)
   print '\n'
   
   return pathImages,keypnt,numpatch,equalnum,imdes,imsample,percentage,codebook,dist,size,fselec,fselec_perc,histnorm,clust,rep
   
def get_imlist(path):        
   return sorted([os.path.join(path,f) for f in os.listdir(path)])
   
if __name__ == "__main__":
   
   #################################################################
   #
   # Get parameters for testing
   #
   #################################################################   
   
   pathImages,keypnt,numpatch,equalnum,imdes,imsample,percentage,codebook,dist,size,fselec,fselec_perc,histnorm,clust,rep = main(sys.argv[1:])

   #################################################################
   #
   # Initializations and result file configurations
   #
   #################################################################   
   
   #warnings.simplefilter("error")
   
   if os.path.exists('save_HIST.txt')==True:
      os.remove('save_HIST.txt')
   
   if os.path.exists('save_dist.txt')==True:
      os.remove('save_dist.txt')
   
   if os.path.exists('saveClustersKmeans.txt')==True:
      os.remove('saveClustersKmeans.txt')
      
   im_dataset_name= pathImages.split('/')[-1]
   
   date_time = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')
   
   name_results_file = im_dataset_name + '_' + keypnt + '_' + str(numpatch) + '_' + str(equalnum) + '_' + imdes + '_' + imsample + '_' + codebook + '_' + str(size) + '_' + fselec + '_' + histnorm + '_' + clust + '_'+ dist + '_' + date_time
   
   directory_results = 'Results'
   
   if not os.path.exists(directory_results):
      os.makedirs(directory_results)  
      
   file_count = 2
   file_name = os.path.join(directory_results,name_results_file)
   print file_name
   while os.path.exists(file_name + ".txt"):
      print "existe o file_name"
      file_name = os.path.join(directory_results,name_results_file) + "_" + str(file_count)
      file_count = file_count + 1
   print file_name
   f = open(file_name + ".txt", 'w')
   #open file to write results to
   
   #################################################################
   #
   # Get images
   #
   #################################################################
   
   #pathImages = '/Users/Mariana/mieec/Tese/Development/ImageDatabases/Graz-01_sample'
   
   imList = get_imlist(pathImages)
   
   print 'Number of images read = ' + str(len(imList))
   f.write("Number of images in dataset read: " + str(len(imList)) + "\n")
   
   #################################################################
   #
   # Image description
   #
   #################################################################
   
   #Get detector classes
   det_sift = siftLib.Sift(numpatch, equalnum)
   det_surf = surfLib.Surf(numpatch, equalnum)
   det_fast = fastDetector.Fast(numpatch, equalnum)
   det_star = starDetector.Star(numpatch, equalnum)
   det_orb = orbLib.Orb(numpatch, equalnum)
   det_random = randomDetector.Random(numpatch)
   
   names_detectors = np.array(["SIFT", "SURF", "FAST", "STAR", "ORB", "RANDOM"])
   detectors = np.array([det_sift, det_surf, det_fast, det_star, det_orb, det_random])
   
   #Get the detector passed in the -k argument
   index = np.where(names_detectors==keypnt)[0]
   if index.size > 0:
      detector_to_use = detectors[index[0]]
   else:
      print 'Wrong detector name passed in the -k argument. Options: SIFT, SURF, FAST, STAR, ORB and RANDOM'
      sys.exit()
      
   #FOR RESULTS FILE
   detector_to_use.writeParametersDet(f)
   
   #Get descriptor classes
   des_sift = siftLib.Sift(numpatch, equalnum)
   des_surf = surfLib.Surf(numpatch, equalnum)
   des_orb = orbLib.Orb(numpatch)
   des_brief = briefDescriptor.Brief()
   des_freak = freakDescriptor.Freak()
      
   names_descriptors = np.array(["SIFT", "SURF", "ORB", "BRIEF", "FREAK"])
   descriptors = np.array([des_sift, des_surf, des_orb, des_brief, des_freak])
   
   #Get the detector passed in the -d argument
   index = np.where(names_descriptors==imdes)[0]
   if index.size > 0:
      descriptor_to_use = descriptors[index[0]]
   else:
      print 'Wrong descriptor name passed in the -d argument. Options: SIFT, SURF, ORB, BRIEF and FREAK'
      sys.exit()
      
   #FOR RESULTS FILE
   descriptor_to_use.writeParametersDes(f)   
   
   kp_vector = [] #vector with the keypoints object
   des_vector = [] #vector wih the descriptors (in order to obtain the codebook)
   number_of_kp = [] #vector with the number of keypoints per image
      
   counter = 1
      
   #save current time
   start_time = time.time()   
   
   labels = []
   class_names = []   
   kp_class = []
   kp_count = 0
   image_count = 0
   image_class = []
   #detect the keypoints and compute the sift descriptors for each image
   for im in imList:
      if 'DS_Store' not in im:
         print 'image: ' + str(im) + ' number: ' + str(counter)
         #read image
         img = cv2.imread(im,0)
         
         #mask in order to avoid keypoints in border of image. size = 40 pixels
         border = 40
         height, width = img.shape
         mask = np.zeros(img.shape, dtype=np.uint8)
         mask[border:height-border,border:width-border] = 1            
         
         #get keypoints from detector
         kp = detector_to_use.detectKp(img,mask)
         
         #get features from descriptor
         des = descriptor_to_use.computeDes(img,kp)
         
         number_of_kp.append(len(kp))
         kp_vector.append(kp)
         if counter==1:
            des_vector = des
         else:
            des_vector = np.concatenate((des_vector,des),axis=0)
         counter += 1   
         
         #for evaluation
         name1 = im.split("/")[-1]
         name = name1.split("_")[0]
                 
         if name in class_names:
            #create the true labels
            index = class_names.index(name)
            labels.append(index)
            
            #to count the number of features for each class
            kp_count = kp_count + len(kp)
            image_count = image_count + 1
         else:
            #create the true labels
            class_names.append(name)
            index = class_names.index(name)
            labels.append(index)   
            
             #to count the number of features for each class
            if index != 0:
               kp_class.append(kp_count)  
               kp_count = len(kp)
               image_class.append(image_count)
               image_count = 1
            else:
               kp_count = kp_count + len(kp)
               image_count = image_count + 1
   
   #number of features of last class
   kp_class.append(kp_count)  
   image_class.append(image_count)
   
   print kp_class
   print image_class
   
   avg_kp_class = []
   for c in range(0,len(kp_class)):
      avg_kp_class.append(float(kp_class[c])/image_class[c])
   
   print 'Number of keypoints per class: ' + str(avg_kp_class)
   
   #measure the time to compute the description of each image (divide time elapsed by # of images)
   elapsed_time = (time.time() - start_time) / len(imList)
   print 'Time to compute detector and descriptor for each image = ' + str(elapsed_time)   
   
   f.write('Average time to compute detector and descriptor for each image = ' + str(elapsed_time) + '\n')
   
   n_images = len(kp_vector)
   
   average_words = sum(number_of_kp)/float(len(number_of_kp))
   
   print 'Total number of features = ' + str(len(des_vector)) 
   f.write('Total number of features obtained = ' + str(len(des_vector)) + '\n') 
   print 'Average number of keypoints per image = ' + str(average_words) 
   f.write('Average number of keypoints per image = ' + str(average_words) + '\n')
   
   #################################################################
   #
   # Image and Keypoint sampling
   #
   ################################################################# 
   
   rand_indexes = []
   nmi_indexes = []
   
   for iteraction in range(0,rep):
      
      print "\nIteraction #" + str(iteraction+1) + '\n'
      f.write("\nIteraction #" + str(iteraction+1) + '\n')
   
      print 'Sampling images and keypoints prior to codebook computation...'
      
      if imsample != "NONE":
         
         sampleKp = sampleKeypoints.SamplingImandKey(n_images, number_of_kp, average_words, percentage)
         sampleallKp = sampleAllKeypoints.SamplingAllKey(percentage)
         
         names_sampling = np.array(["SAMPLEI", "SAMPLEP"])
         sample_method = np.array([sampleKp, sampleallKp])   
         
         #Get the detector passed in the -g argument
         index = np.where(names_sampling==imsample)[0]
         if index.size > 0:
            sampling_to_use = sample_method[index[0]]
         else:
            print 'Wrong sampling method passed in the -g argument. Options: NONE, SAMPLEI, SAMPLEP'
            sys.exit()
            
         #FOR RESULTS FILE
         sampling_to_use.writeFile(f)
      
         des_vector_sampled = sampling_to_use.sampleKeypoints(des_vector)
            
         print 'Total number of features after sampling = ' + str(len(des_vector_sampled))
         f.write('Total number of features after sampling = ' + str(len(des_vector_sampled)) + '\n')
            
         print 'Images and keypoints sampled...'
         
      else:
         print 'No sampling method chosen'
         #FOR RESULTS FILE
         f.write("No method of keypoint sampling chosen. Use all keypoints for codebook construction \n")
         des_vector_sampled = des_vector
      
      #################################################################
      #
      # Codebook computation
      #
      #################################################################
   
      print 'Obtaining codebook...'
      
      #save current time
      start_time = time.time()   
      
      #Get detector classes
      codebook_kmeans = KMeans1.KMeans1(size)
      codebook_birch = Birch.Birch(size)
      codebook_minibatch = minibatch.MiniBatch(size)
      codebook_randomv = randomSamplesBook.RandomVectors(size)
      codebook_allrandom = allrandom.AllRandom(size)
      
      names_codebook = np.array(["KMEANS", "BIRCH", "MINIBATCH", "RANDOMV", "RANDOM"])
      codebook_algorithm = np.array([codebook_kmeans, codebook_birch, codebook_minibatch, codebook_randomv, codebook_allrandom])
      
      #Get the detector passed in the -c argument
      index = np.where(names_codebook==codebook)[0]
      if index.size > 0:
         codebook_to_use = codebook_algorithm[index[0]]
      else:
         print 'Wrong codebook construction algorithm name passed in the -c argument. Options: KMEANS, MINIBATCH, RANDOMV and RANDOM'
         sys.exit()   
         
      #FOR RESULTS FILE
      codebook_to_use.writeFileCodebook(f)
         
      #Get centers and projections using codebook algorithm
      ceters, projections = codebook_to_use.obtainCodebook(des_vector_sampled,des_vector)
      
      elapsed_time = (time.time() - start_time)
      print 'Time to compute codebook = ' + str(elapsed_time)   
      f.write('Time to compute codebook = ' + str(elapsed_time) +'\n')
      
      #################################################################
      #
      # Obtain Histogram
      #
      #################################################################   
   
      print 'Obtaining histograms...'
      
      #print 'projection shape = '+ str(projections.shape)
      #print 'size = ' + str(size)
      #print 'n of images = ' + str(n_images)
      #print 'number of kp' + str(number_of_kp)
      
      hist = histogram.computeHist(projections, size, n_images, number_of_kp)
      
      print 'Histograms obtained'
      
      ################################################################
      #
      # Feature selection
      #
      #################################################################  
      
      print 'Number of visual words = '+str(len(hist[0]))
      
      if fselec != "NONE":
         
         print 'Applying feature selection to descriptors...'
         
         filter_max = filterMax.WordFilterMax(fselec_perc[0])
         filter_min = filterMin.WordFilterMin(fselec_perc[1])
         filter_maxmin = filterMaxMin.WordFilterMaxMin(fselec_perc[0], fselec_perc[1])
         
         names_filter = np.array(["FMAX", "FMIN", "FMAXMIN"])
         filter_method = np.array([filter_max, filter_min, filter_maxmin])
            
         #Get the detector passed in the -f argument
         index = np.where(names_filter==fselec)[0]
         if index.size > 0:
            filter_to_use = filter_method[index[0]]
         else:
            print 'Wrong codebook construction algorithm name passed in the -f argument. Options: NONE, FMAX, FMIN, FMAXMIN'
            sys.exit()      
         
         hist = filter_to_use.applyFilter(hist,size,n_images)
         
         #FOR RESULTS FILE
         filter_to_use.writeFile(f)
            
         new_size = hist.shape[1]
         
         print 'Visual words Filtered'
         print 'Number of visual words filtered = '+str(size-new_size)
         f.write("Number of visual words filtered = " + str(size-new_size) + '\n')
         print 'Final number of visual words = '+str(new_size)
         f.write('Final number of visual words = '+str(new_size) + '\n')
         
      else:
         #FOR RESULTS FILE
         filter_min = filterMin.WordFilterMin(0)
         hist = filter_min.applyFilter(hist,size,n_images)
         new_size = hist.shape[1]
         print 'Number of visual words filtered = '+str(size-new_size)
         f.write("No feature selection applied \n")
      
      #################################################################
      #
      # Histogram Normalization
      #
      #################################################################      
      
      if histnorm != "NONE":
         
         #Get detector classes
         norm_sbin = simpleBinarization.SimpleBi()
         norm_tfnorm = tfnorm.Tfnorm()
         norm_tfidf = tfidf.TfIdf()
         norm_tfidf2 = tfidf2.TfIdf2()
         norm_tfidfnorm = tfidfnorm.TfIdfnorm()
         norm_okapi = okapi.Okapi(average_words)
      
         names_normalization = np.array(["SBIN","TFNORM","TFIDF","TFIDF2","TFIDFNORM", "OKAPI"])
         normalization_method = np.array([norm_sbin,norm_tfnorm,norm_tfidf,norm_tfidf2, norm_tfidfnorm, norm_okapi])
         
         #Get the detector passed in the -h argument
         index = np.where(names_normalization==histnorm)[0]
         if index.size > 0:
            normalization_to_use = normalization_method[index[0]]
            new_hist = normalization_to_use.normalizeHist(hist, new_size, n_images)
         else:
            print 'Wrong normalization name passed in the -h argument. Options: SBIN, TFNORM, TFIDF and TFIDF2'
            sys.exit()     
         
         #FOR RESULTS FILE
         normalization_to_use.writeFile(f)      
            
      else:
         #FOR RESULTS FILE
         f.write("No histogram normalization applied\n")
         new_hist = hist
      
      #################################################################
      #
      # Clustering of the features
      #
      #################################################################     
      
      #save current time
      start_time = time.time()     
   
      #Get detector classes
      clust_dbscan = Dbscan.Dbscan(dist)
      clust_kmeans = KMeans1.KMeans1([20])
      clust_birch = Birch.Birch(20)
      clust_meanSift = meanSift.MeanSift(20)
      clust_hierar1 = hierarchicalClustering.Hierarchical(20, dist)
      clust_hierar2 = hierarchicalClustScipy.HierarchicalScipy(dist)
      
      names_clustering = np.array(["DBSCAN", "KMEANS", "BIRCH", "MEANSIFT", "HIERAR1", "HIERAR2"])
      clustering_algorithm = np.array([clust_dbscan, clust_kmeans, clust_birch, clust_meanSift, clust_hierar1, clust_hierar2])
      
      #Get the detector passed in the -a argument
      index = np.where(names_clustering==clust)[0]
      if index.size > 0:
         clustering_to_use = clustering_algorithm[index[0]]
      else:
         print 'Wrong clustering algorithm name passed in the -a argument. Options: DBSCAN, KMEANS, BIRCH, MEANSIFT'
         sys.exit()      
         
      clusters = clustering_to_use.obtainClusters(new_hist)   
      
      #FOR RESULTS FILE
      clustering_to_use.writeFileCluster(f)
      
      elapsed_time = (time.time() - start_time)
      print 'Time to run clustering algorithm = ' + str(elapsed_time) 
      f.write('Time to run clustering algorithm = ' + str(elapsed_time) + '\n')
      
      print 'Number of clusters obtained = ' + str(max(clusters)+1)
      f.write('Number of clusters obtained = ' + str(max(clusters)+1) + '\n')
      
      print 'Clusters obtained = ' + str(np.asarray(clusters))
      
      #date_time = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')
      #np.savetxt('saveClusters_'+date_time+'_.txt', clusters, '%i', ',')
      
      #################################################################
      #
      # Evaluation
      #
      #################################################################   
      
      f.write('True Labels = ')
      f.write(str(labels))
      f.write('Clusters Obtained = ')
      f.write(str(clusters.tolist()))
      
      if len(clusters) == len(labels):
         
         f.write('Clusters Obtained = ' + str(np.asarray(labels)))
         
         f.write("\nResults\n")
          
         rand_index = metrics.adjusted_rand_score(labels, clusters)
         rand_indexes.append(rand_index)
         print 'rand_index = ' + str(rand_index)
         f.write("Rand Index = " + str(rand_index) + "\n")
              
         NMI_index = metrics.normalized_mutual_info_score(labels, clusters)
         nmi_indexes.append(NMI_index)
         print 'NMI_index = ' + str(NMI_index)   
         f.write("NMI Index = " + str(NMI_index) + "\n")
   
   if rep > 1:
      f.write("\nFINAL RESULTS\n")
      f.write("Avg Rand Index = " + str(float(sum(rand_indexes))/rep) + "\n")
      f.write("Std Rand Index = " + str(statistics.stdev(rand_indexes)) + "\n")
      f.write("Avg NMI Index = " + str(float(sum(nmi_indexes))/rep) + "\n")
      f.write("Std NMI Index = " + str(statistics.stdev(nmi_indexes)) + "\n")
   f.close()