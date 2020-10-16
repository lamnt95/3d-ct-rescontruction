# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:00:06 2020

@author: DUONGTT
"""

import numpy as np
from os.path import join
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import time
#import os,sys
#import define_filter
#import astra
import cv2

##Khai báo chương trình con
""" Áp dụng các hàm lọc cho dữ liệu hình chiếu trước khi tái tạo """
def frequency(_im,_filter_name):
      x_size = len(_im[0])
      y_size = len(_im[1])
      fft_array = np.zeros((x_size,y_size))
      ifft_img = np.zeros((x_size,y_size))

      f = np.fft.fftshift(abs(np.mgrid[-1:1:2/x_size])).reshape(-1,1)  
      w = 2*np.pi*f

      if _filter_name == "ram": ##Ram-lak default
         pass 
      elif _filter_name == "shepp": ## Shepp-Logan
         f = f*np.sin(w/2)/(w/2) 
      elif _filter_name == "cosine": ## Cosine
         f = f*np.cos(w/2)
         f = f.clip(min=0)
      elif _filter_name == "hann": ## Hann
         f = f*(1 + np.cos(w/2))/2   
      elif _filter_name == "hamming": ## Hamming
         f = f*(0.54 + 0.46*np.cos(w/2)) ## Hamming
      elif _filter_name == "none":
         f[:] = 1
      else:
         raise ValueError("Unknown filter: %s" % filter_name)

      fft_array = _im

      f = np.fft.ifft(f)    

      plot_fft = np.zeros((size,size), dtype = complex)

      for i in range(x_size):
         for j in range(y_size):
            plot_fft[i,j] = fft_array[i,j]*f[j]

      return plot_fft
  
def space(_im,_filter_name):
    if _filter_name=="gaussian":
       KW=9
       KH=9
       sigma_X=3
       sigma_Y=3
       imf=cv2.GaussianBlur(_im, ksize=(KW, KH), sigmaX=sigma_X, sigmaY=sigma_Y)       
    elif _filter_name=="normal":
       kernel = np.ones((5,5),np.float32)/9
       imf= cv2.filter2D(_im,-1,kernel)
    elif _filter_name=="bilateral":
       imf=cv2.bilateralFilter(_im,5, 20, 20)
    elif _filter_name=="median":
       imf=cv2.medianBlur(_im, 5)
         
    return imf

#Chương trình chính
im = cv2.imread('reco0280.png')                # Đọc ảnh RGB
#im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   # Convert RGB sang ảnh thang xám
cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
filter_method = input("Chọn phương pháp lọc (frequency, space)=")
if filter_method=="frequency":
    filter_name=input("Chọn hàm lọc (ram,shepp,cosine,hamming,hann,none) = ")
    im = frequency(im.astype(float),filter_name)
elif filter_method=="space":
    filter_name=input("Chọn hàm lọc (gaussian, normal, bilateral,median)= ")
    im = space(im.astype(float),filter_name)
else: pass    
#Lưu file
cv2.imwrite(join('reco0280-n.png'), im)
plt.imshow(im)
