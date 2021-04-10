# David Cabezas Berrido

# toLow.py

# Lower the resolutions of images in a folder

import cv2
import matplotlib.pyplot as plt
import glob

from joblib import Parallel, delayed

# LONG: to shape 1024 x 1536 (H x W)

# Single image
"""
def low_img(img_path):
    newpath='results-low/long/'+img_path.split('/')[-1]
    img=plt.imread(img_path)
    img=cv2.resize(img, dsize=(1536,1024), interpolation=cv2.INTER_AREA)
    plt.imsave(newpath,img)
    
# Set (folder) with several images
def low(img_set):
    path=img_set+'*'+'.jpg'
    Parallel(n_jobs=4)(delayed(low_img)(img_path) for img_path in glob.glob(path) if img_path.split('/')[-1].split('-')[1][0]=='2') # Only test images
low('../Data/jpeg/long/')
"""

# SHORT: to shape 512 x 768
"""
# Single image
def low_img(img_path):
    newpath='results-low/short/'+img_path.split('/')[-1]
    img=plt.imread(img_path)
    img=cv2.resize(img, dsize=(768,512), interpolation=cv2.INTER_AREA)
    plt.imsave(newpath,img)
    
# Set (folder) with several images
def low(img_set):
    path=img_set+'*'+'.jpg'
    Parallel(n_jobs=4)(delayed(low_img)(img_path) for img_path in glob.glob(path) if img_path.split('/')[-1].split('-')[1][0]=='2') # Only test images
low('../data/jpeg/short/')
"""

# Results

# Single image
def low_img(img_path):
    newpath=img_path.replace('results','results-low')
    img=plt.imread(img_path)
    img=cv2.resize(img, dsize=(768,512), interpolation=cv2.INTER_AREA)
    plt.imsave(newpath,img)
    

# Set (folder) with several images
def low(img_set):
    path=img_set+'*'+'.jpg'
    Parallel(n_jobs=4)(delayed(low_img)(img_path) for img_path in glob.glob(path))

#low('results/p2pGAN/')
#low('results/histEqHLS/')
#low('results/histEqHSV/')
#low('results/p2pGANlow/') # Processed by generator with lower resolution (1024 x 1536)
#low('results/p2pGANlow2/') # Processed by generator with lower resolution (2048 x 3072)
#low('results/p2pGANepoch70/')
#low('results/p2pGANepoch60/')
#low('results/p2pGAN-train-false/')
#low('results/border-0/')
#low('results/border-copy/')

