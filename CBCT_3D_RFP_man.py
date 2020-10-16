## Import dataset folter, return filtered dataset

import numpy as np
from os.path import join
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import time
import os,sys
#import define_filter
import astra
import cv2

## Ignoring warnings
import warnings
if not sys.warnoptions:warnings.simplefilter("ignore")


##Khai báo chương trình con
""" Áp dụng các hàm lọc cho dữ liệu hình chiếu trước khi tái tạo """
def filtered_img(_im,_filter_name):
      x_size = len(_im[0])
      y_size = len(_im[1])
      fft_array = np.zeros((x_size,y_size), dtype=complex)
      ifft_img = np.zeros((x_size,y_size))

      f = np.fft.fftshift(abs(np.mgrid[-1:1:2/x_size])).reshape(-1,1)  
      w = 2*np.pi*f

      if _filter_name == "ram": ##Ram-lak default
         pass 
      elif _filter_name == "shepp": ## Shepp-Logan
         f = f*np.sin(w/2)/(w/2)/2 
      elif _filter_name == "cosine": ## Cosine
         f = f*np.cos(w/2)/2
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

      plot_fft = np.zeros((x_size,y_size), dtype = complex)

      for i in range(x_size):
         for j in range(y_size):
            plot_fft[i,j] = fft_array[i,j]*f[j]

      return plot_fft

# Thông số cấu hình hình chiếu
SDD = int(input(" SOD (mặc định: 760mm) = "))                        # Khoảng cách từ nguồn tới đầu dò [mm]
ODD = int(input(" ODD(mặc định: 120mm) = "))                         # Khoảng cách từ vật tới đầu dò [mm]
SOD = SDD-ODD                                                        # Khoảng cách từ nguồn tới vật [mm]
detector_pixel_size = float(input(" Kích thước một ô đầu dò (mặc định: 0.1mm) = "))  # [mm]
detector_rows = int(input(" Chiều ngang của đầu dò (mặc định:1000) = "))          # Vertical size of detector [pixels].
detector_cols = int(input(" Chiều dài của đầu dò (mặc định:1000) = "))          # Horizontal size of detector [pixels].
filter_name = input("Chọn hàm lọc (ram,shepp,cosine,hamming,hann,none) = ")
num_of_projections = 720
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)

#Tải và lọc dữ liệu hình chiếu
print("Tải dữ liệu hình chiếu ... \n")
start = time.time()
projections = np.zeros((detector_rows, num_of_projections, detector_cols))
for i in range(num_of_projections): 
    im = cv2.imread('1e12.jpg')                # Đọc ảnh RGB
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   # Convert RGB sang ảnh thang xám
    cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
    if not(filter_name=="none"):
        im = filtered_img(im.astype(float),filter_name)
#   im = -np.log(im/np.max(im))                 # Nghịch đảo màu cho giống với ảnh X quang
    projections[:,i,:] = im
stop = time.time()

print("Thời gian tải dữ liệu = ", stop - start)

print("Bắt đầu tái tạo .... ")
start = time.time()
# Copy projection images into ASTRA Toolbox.
proj_geom = astra.create_proj_geom('cone',  1, 1, 
                                   detector_rows, detector_cols, angles, 
                                   SOD/detector_pixel_size, ODD/detector_pixel_size)

# Khởi tạo biến lưu trữ hình chiếu (ID - kiểu int), đây là một sinogram 3D với thông số
# hình học chiếu lấy từ biến proj_geom. Biến projections_id được dùng để định danh 
# hình chiếu và được sử dụng để cấu hình thuật toán chiếu ngược sau này.
projections_id = astra.data3d.create('-sino', proj_geom, projections)

# Khởi tạo không gian tái tạo vật thể
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, detector_rows)
# Khởi tạo biến lưu trữ ID của không gian tái tạo 3D, với kích thước của không gian 
# được lưu trong biến vol_geom, giá trị ban đầu của mỗi điểm anh trong không gian
# được khai báo bằng 0
reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
# Biến alg_cfg có kiểu dict, dùng để lưu trông số cấu hình cho thuật toán tái tạo
alg_cfg = astra.astra_dict('FDK_CUDA')
alg_cfg['ProjectionDataId'] = projections_id
alg_cfg['ReconstructionDataId'] = reconstruction_id

# Khởi tạo thuật toán từ biến alg_cfg
algorithm_id = astra.algorithm.create(alg_cfg)

# Chạy thuật toán, hàm này có hai biến đầu vào:
#    - algorithm_id: ID của thuật toán, kiểu int
#    - iterations vòng lặp cho thuật toán đại số, mặc định bằng 1 với thuật toán chiếu ngược
astra.algorithm.run(algorithm_id)
print ("==> Chạy thuật toán tái tạo... ") 
# Thu dữ liệu tái tạo, dữ liệu trả về có dạng numpy.ndarray
reconstruction = astra.data3d.get(reconstruction_id)

# Loại bỏ giá trị âm của dữ liệu tái tạo gây ra do noise
# Định dạng lại dữ liệu ra có kiểu 8 bit
reconstruction[reconstruction < 0] = 0
reconstruction /= np.max(reconstruction)
reconstruction = np.round(reconstruction * 255).astype(np.uint8)

# Lưu trữ ảnh tái tạo
print("==> Xuất ảnh tái tạo  ... ")
output_dir = r'z1e12 -shep' 
for i in range(detector_rows):
    recon_img = reconstruction[i, :, :]*2       # Tăng độ sáng của ảnh
    #recon_img = np.flipud(recon_img)
    cv2.imwrite(join(output_dir, 'reco%04d.png' % i), recon_img)

print ("=> Saving 3D array data... ")
np.save(join(output_dir, "d760ata_3D"),data_3D)

# Cleanup.
astra.algorithm.delete(algorithm_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projections_id)
stop = time.time()
print("Thời gian tái tạo = ",stop-start,"sec")
