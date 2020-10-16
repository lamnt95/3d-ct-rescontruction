# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:42:06 2020

@author: buiha
"""

from imutils import paths
import numpy as np
import cv2

folder_path = r"1e12"
img_paths = list(paths.list_images(folder_path))

data_3D = []
for imagePath in img_paths:
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.fastNlMeansDenoising(img)
    data_3D.append(img)
    

np.save("1e12",data_3D)

# cv2.imshow('img',data_3D[:,:,100,:])