## Import dataset folter, return filtered dataset

import numpy as np
from os.path import join
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import time
import os,sys
import astra
import cv2


## Ignoring warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

## Ask for dataset and make output_dir
input_dir = r"D:\MCNP\CT sample\MauDX\Al\140kV\1254-1"
output_dir = r'D:\CBCT_KC05\Reconstruction_Program\SIRT_CUDA\1254-1-1200\150' 

# Configuration.
distance_source_origin = 760 #(mm)
distance_origin_detector = 125 #(mm)
detector_pixel_size = 0.1 #(mm)
detector_rows = 520 #(pix)
detector_cols = 520 #(pix)
num_of_projections = 900
#num_of_projections = len(os.listdir(input_dir)) ## Auto-check number of projections
#print(num_of_projections)
#output_dir="reconstruction-200x200-%d-SIRT3D150" %num_of_projections
#os.mkdir(output_dir)


angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)

print("Loading dataset ... \n")
start = time.time()
projections = np.zeros((detector_rows, num_of_projections, detector_cols))
for i in range(num_of_projections):
    print("Loading Projection %04d" %i)
    im = imread(join(input_dir, '%04d.png' %i))
    cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
    projections[:,i,:] = im

print("Start reconstruction .... ")
# Copy projection images into ASTRA Toolbox.
proj_geom = \
  astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
                         (distance_source_origin + distance_origin_detector) /
                         detector_pixel_size, 0)
projections_id = astra.data3d.create('-sino', proj_geom, projections)

# Create reconstruction.
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                          detector_rows)
reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
alg_cfg = astra.astra_dict('SIRT3D_CUDA')
alg_cfg['ProjectionDataId'] = projections_id
alg_cfg['ReconstructionDataId'] = reconstruction_id
algorithm_id = astra.algorithm.create(alg_cfg)
astra.algorithm.run(algorithm_id,1000)
reconstruction = astra.data3d.get(reconstruction_id)

# Limit and scale reconstruction.
reconstruction[reconstruction < 0] = 0
reconstruction /= np.max(reconstruction)
reconstruction = np.round(reconstruction * 255).astype(np.uint8)

# Save reconstruction.
print("Export reconstruction ... ")
for i in range(detector_rows):
    im = reconstruction[:, i, :]*4
#    im = np.flipud(im)
    imwrite(join(output_dir, 'reco%04d.png' % i), im)

# Cleanup.
astra.algorithm.delete(algorithm_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projections_id)

stop = time.time()
print("Reconstruction time for " + input_dir +" = ",stop-start,"sec") 
