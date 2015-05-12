# BOFClustering

  Application of the Bag-of-Features model for the clustering of images. It uses three Python libraries: OpenCV, Scipy and Scikit-Learn.

# 1) Details
  This repository contains Python scripts that apply the Bag-of-Features model for the images clustering.
  The Bag-of-Features model is a way of representing an image as a histogram of occurence of visual words, from a codebook, with no spacial information. First, features have to be extracted from the images. Here, this is done by using feature detectors (that extact keypoints or pacthes) and descriptors for image data. The detectors used were: SIFT, SURF, FAST, ORB and RANDOM. The RANDOM detector outputs random patches from the image. The descriptors used were: SIFT, SURF, BRIEF, FREAK and ORB. The average number of keypoints exctracted from each image is a varible that can be changed here. Next, the features are sampled for the construction of the codebook. Two sampling methods are available: SAMPLEP and SAMPLEI. SAMPLEP is a simple sampling of all the features while SAMPLEI samples less features from the images that have more keypoints, in order to reduce the variability of the number of keypoints per images, that could influence the codebook. After that, a technique is used to construct the codebook of visual words. Three clustering algorithms can be used: KMEANS, BIRCH, and MINIBATCH (Mini-Batch K-Means). Also, two algorithms for creating random codebooks can be tested. The first, RANDOMV, selects random feature vectors from the images as the visual words. The other, RANDOM, creates random feature vectors as the visual words. The size of the codebook also needs to be specified. Next, feature selection can be applied to the visual words from the codebook. Three possibilities can be tested: FMIN, FMAX and FMAXMIN. FMIN is to remove the visual words that appear very rarely in the images, FMAX removes the visual words that appear too often and FMAXMIN removes both. The histograms built for each images can, then, be normalized following different stategies: SBIN (sample binarization), TFIDF (term frequency-inverse document frequency), TDIDF2 (term frequency-inverse document frequency variation), TFNORM (term frequency normalized), TFIDFNORM (term frequency-inverse document frequency normalized) and OKAPI (Okapi term frequency-inverse document frequency). Finaly, the clustering algorithms for clustering the images are: KMEANS, DBSCAN, BIRCH, HIERAR1 (from Scipy library), HIERAR2 (from Scikit-Learn library). For some of the clustering, the distance measure can be specified. 
  
# 2) Requirements and how to install

# Ubuntu

- Install Python > version 2.7
- Opencv: Follow the tutorial on: http://www.samontab.com/web/2014/06/installing-opencv-2-4-9-in-ubuntu-14-04-lts/ but change the version of OpenCV downloaded to 2.4.10 instead of 2.4.9.
- Install pip: sudo apt-get install python-pip python-dev build-essential 
  sudo pip install --upgrade pip 
	sudo pip install --upgrade virtualenv 
- Install python modules required:
  sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
  sudo pip install -U scikit-learn
  
