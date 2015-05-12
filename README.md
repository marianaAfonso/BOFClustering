# BOFClustering

  Application of the Bag-of-Features model for the clustering of images. It uses three Python libraries: OpenCV, Scipy and Scikit-Learn.

## 1. Details
  This repository contains Python scripts that apply the Bag-of-Features model for the images clustering.
  The Bag-of-Features model is a way of representing an image as a histogram of occurence of visual words, from a codebook, with no spacial information. First, features have to be extracted from the images. Here, this is done by using feature detectors (that extact keypoints or pacthes) and descriptors for image data. The detectors used were: SIFT, SURF, FAST, ORB and RANDOM. The RANDOM detector outputs random patches from the image. The descriptors used were: SIFT, SURF, BRIEF, FREAK and ORB. The average number of keypoints exctracted from each image is a varible that can be changed here. Next, the features are sampled for the construction of the codebook. Two sampling methods are available: SAMPLEP and SAMPLEI. SAMPLEP is a simple sampling of all the features while SAMPLEI samples less features from the images that have more keypoints, in order to reduce the variability of the number of keypoints per images, that could influence the codebook. After that, a technique is used to construct the codebook of visual words. Three clustering algorithms can be used: KMEANS, BIRCH, and MINIBATCH (Mini-Batch K-Means). Also, two algorithms for creating random codebooks can be tested. The first, RANDOMV, selects random feature vectors from the images as the visual words. The other, RANDOM, creates random feature vectors as the visual words. The size of the codebook also needs to be specified. Next, feature selection can be applied to the visual words from the codebook. Three possibilities can be tested: FMIN, FMAX and FMAXMIN. FMIN is to remove the visual words that appear very rarely in the images, FMAX removes the visual words that appear too often and FMAXMIN removes both. The histograms built for each images can, then, be normalized following different stategies: SBIN (sample binarization), TFIDF (term frequency-inverse document frequency), TDIDF2 (term frequency-inverse document frequency variation), TFNORM (term frequency normalized), TFIDFNORM (term frequency-inverse document frequency normalized) and OKAPI (Okapi term frequency-inverse document frequency). Finaly, the clustering algorithms for clustering the images are: KMEANS, DBSCAN, BIRCH, HIERAR1 (from Scipy library), HIERAR2 (from Scikit-Learn library). For some of the clustering, the distance measure can be specified. 
  
## 2. Requirements and how to install

### Ubuntu

- Install Python > version 2.7
- Opencv: Follow the tutorial on: http://www.samontab.com/web/2014/06/installing-opencv-2-4-9-in-ubuntu-14-04-lts/ but change the version of OpenCV downloaded to 2.4.10 instead of 2.4.9.
- Install pip: 
 	- sudo apt-get install python-pip python-dev build-essential 
  	- sudo pip install --upgrade pip 
	- sudo pip install --upgrade virtualenv 
- Install python modules required:
  	- sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
  	- sudo pip install -U scikit-learn

### OS X (Mac)

- Install Python > version 2.7 (already installed)
- Opencv: Follow the tutorial on: https://jjyap.wordpress.com/2014/05/24/installing-opencv-2-4-9-on-mac-osx-with-python-support/ but change the version of OpenCV to 2.4.10 instead of 2.4.9.
- Install pip: 
 	- sudo easy_install pip 
- Install python modules required:
  	- sudo pip install numpy scipy matplotlib
  	- sudo pip install -U numpy scipy scikit-learn
 
## 3. How to use

### Image Dataset

All the images have to be in a the same folder with the in the format:
- *class* _ *number inside the class* 

### Execution

Go to src folder

call: 

- `main.py -p <path to images> -k <keypoint detector> -n <number of keypoints> -s <equal number of keypoints per image> -d <keypoint descriptor> -g <Sampling method:percentage of features/images> -c <codebook method> -m <distance measure> -t <size of codebook> -f <feature selection method> -h <histogram normalization> -a <clustering algorithm> -r <number of repetitions>`

- Also if `<feature selection method>` is different than NONE: 
	- `if feature selection method = FMAX: -f FMAX:max occurence allowed (0-1)`
	- `if feature selection method = FMIN: -f FMIN:min occurence allowed (0-1)`
	- `if feature selection method = FMAXMIN: -f FMAXMIN:max occurence allowed (0-1):min occurence allowed (0-1)`

*Options:
- **path to images:** any path to a folder with images
- **keypoint detector:** SIFT, SURF, FAST, STAR, ORB or RANDOM.
- **number of keypoints:** any integer number
- **equal number of keypoints per image:** TRUE if equal number of keypoints per image is required (at least maximum) or FALSE if not
- **keypoint descriptor:** SIFT, SURF, ORB, BRIEF or FREAK
- **Sampling method:** SAMPLEP, SAMPLEI
- **percentage of features/images:** any fractionay number from 0 to 1
- **codebook method:** KMEANS, BIRCH, MINIBATCH, RANDOMV or RANDOM
- **distance measure:** euclidean, city-block, cosine, correlation... (see http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html)
- **size of codebook:** any integer number
- **feature selection method:** NONE, FMAX, FMIN or FMAXMIN
- **histogram normalization:** NONE, SBIN, TFIDF, TDIDF2, TFNORM , TFIDFNORM and OKAPI
- **clustering algorithm:** KMEANS, DBSCAN, BIRCH, HIERAR1 or HIERAR2
- **number of repetitions:** any integer number

## 4. Output

The program will create a output .txt file containing all the details regarding the process performed. The file should look like this:

```
Number of images in dataset read: 1580
Keypoint Detector RANDOM (DENSE) with parameters: XYStep = 5 
Keypoint Descriptor SURF with parameters: DEFAULT
Average time to compute detector and descriptor for each image = 0.340125562119
Total number of features obtained = 2368500
Average number of keypoints per image = 1500.0

Iteraction #1
Sampling keypoints randomly. Parameters: Keypoint Percentage = 0.01 
Total number of features after sampling = 23685
Codebook construction method Random from Feature Vectors with parameters: Number of clusters = 500 
Time to compute codebook = 7.32728815079
No feature selection applied 
No histogram normalization applied
Clustering algorithm Hierarchical from Scipy with parameters: Distance = cosine Linkage method = average Stop method = distance ProportionDist = 0.4 
Time to run clustering algorithm = 1.94860482216
Number of clusters obtained = 18

Results
Clusters Obtained = [11 11 11 ..., 11  5 10]
Labels = [0 0 0 ..., 7 7 7]

Rand Index = 0.0748990074655
NMI Index = 0.180574162527
```
  
