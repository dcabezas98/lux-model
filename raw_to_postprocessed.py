# David Cabezas Berrido

# raw_to_postprocessed.py

# Raw images to JPEG, PNG or TIFF

import rawpy
import imageio
from PIL import Image
import glob

from joblib import Parallel, delayed

# Folder paths
RAW='data/raw/'
RAW_SONY=RAW+'Sony/'
RAW_FUJI=RAW+'Fuji/'
EXT_SONY='.ARW'
EXT_FUJI='.RAF'
JPEG='data/jpeg/'
JPEG_LONG=JPEG+'long/'
JPEG_SHORT=JPEG+'short/'
TIFF='data/tiff/'
PNG='data/png/'


BRIGHT=16 #12 #8 #6 #4 #2 #1

# Postprocess to JPEG format

# Single image
def rawToJpeg_img(img_path, out_path, ext, short):
    if '_00_0.1s' not in img_path and short: # Only short exposure images with 0.1s exposure time
        return
    newpath=out_path+img_path.split('/')[-1].replace(ext,'.jpg')
    if short:
        bright = BRIGHT # Select bright for short exposure images
        newpath=newpath.replace('s.','s_x'+str(bright)+'.').replace('_','-').replace('-00-0.1s','') # LATEX has problems to include images with _ or . in the filename
    else:
        bright = 1
    with rawpy.imread(img_path) as raw:
        rgb=raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8, bright=bright) # Postprocess with the desired bright
    Image.fromarray(rgb).save(newpath,quality=95,optimize=True) # Save as JPEG

# Set (folder) with several images
def rawToJpeg(img_set, out_path, ext, short=False):
    path=img_set+'*'+ext
    Parallel(n_jobs=4)(delayed(rawToJpeg_img)(img_path, out_path, ext, short) for img_path in glob.glob(path))

# JPEG, long
img_set=RAW_SONY+'long/'
out_path=JPEG_LONG+'sony-'
rawToJpeg(img_set, out_path, EXT_SONY)

img_set=RAW_FUJI+'long/'
out_path=JPEG_LONG+'fuji-'
rawToJpeg(img_set, out_path, EXT_FUJI)

# JPEG, short
img_set=RAW_SONY+'short/'
out_path=JPEG_SHORT+'sony-'
rawToJpeg(img_set, out_path, EXT_SONY, True)

img_set=RAW_FUJI+'short/'
out_path=JPEG_SHORT+'fuji-'
rawToJpeg(img_set, out_path, EXT_FUJI, True)

# Postprocess to TIFF or PNG format, only short exposure images for test purposes

# Single image
def rawToTiffPng_img(img_path, out_path, ext, out_ext):
    if '_00_0.1s' not in img_path: # Only short exposure images with 0.1s exposure time
        return
    img_path2=img_path.split('/')[-1]
    if img_path2.split('_')[0][0] in {'0','1'}: # The image is in the train set
        return
    newpath=out_path+img_path2.replace(ext, out_ext).replace('s.','s_x'+str(BRIGHT)+'.').replace('_','-').replace('-00-0.1s','') # LATEX has problems to include images with _ or . in the filename

    with rawpy.imread(img_path) as raw:
        rgb=raw.postprocess(use_camera_wb=True, no_auto_bright=True, bright=BRIGHT) # Postprocess with the desired bright
    imageio.imwrite(newpath, rgb) # Save as TIFF or PNG

    # Set (folder) with several images
def rawToTiffPng(img_set, out_path, ext, out_ext):
    path=img_set+'*'+ext
    Parallel(n_jobs=4)(delayed(rawToTiffPng_img)(img_path, out_path, ext, out_ext) for img_path in glob.glob(path))

# TIFF    
img_set=RAW_SONY+'short/'
out_path=TIFF+'sony-'
rawToTiffPng(img_set, out_path, EXT_SONY, '.tif')

img_set=RAW_FUJI+'short/'
out_path=TIFF+'fuji-'
rawToTiffPng(img_set, out_path, EXT_FUJI, '.tif')

# PNG    
img_set=RAW_SONY+'short/'
out_path=PNG+'sony-'
rawToTiffPng(img_set, out_path, EXT_SONY, '.png')

img_set=RAW_FUJI+'short/'
out_path=PNG+'fuji-'
rawToTiffPng(img_set, out_path, EXT_FUJI, '.png')
