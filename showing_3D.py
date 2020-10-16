# -*- coding: utf-8 -*-
"""
Created on Sun May  3 20:40:20 2020

@author: buiha
"""

from __future__ import division
import numpy as np
from os.path import join
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import cv2
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# ------------------------ Khai báo chương trình con ------------------------ #
# --------------------------------------------------------------------------- #
def showing_3D (matrix_3D=np.zeros((100,100,100)), Trans=120):
    # N là kích thước của vật thể 3 chiều
    # Mapping ảnh GrayScale sang ảnh màu RGB
    red   = np.zeros((matrix_3D.shape), dtype = np.uint8)
    green = np.zeros((matrix_3D.shape), dtype = np.uint8)
    blue  = np.zeros((matrix_3D.shape), dtype = np.uint8)
    # Thực hiện biến đổi màu cho toàn bộ không gian 3D
    for i in range (0, matrix_3D.shape[2]):
        RGB_Img = cv2.applyColorMap(matrix_3D[:,:,i], cv2.COLORMAP_JET)
        red[:,:,i]   = RGB_Img[:,:,0]
        green[:,:,i] = RGB_Img[:,:,1]
        blue[:,:,i]  = RGB_Img[:,:,2]
        # Tạo không gian màu 4 chiều RGBA
        object_4D = np.empty(matrix_3D.shape + (4,), dtype=np.uint8)
        # Nạp giá trị màu vào không gian
        object_4D[:,:,:,0] = blue
        object_4D[:,:,:,1] = green
        object_4D[:,:,:,2] = red
        object_4D[:,:,:,3] = 120
        # Hiển thị trục tọa độ
        object_4D[:, 0, 0] = [255,0,0,100]
        object_4D[0, :, 0] = [0,255,0,100]
        object_4D[0, 0, :] = [0,0,255,100]
        
    # Khởi tạo giao diện hiển thị 3D dùng PyQT
    app = QtGui.QApplication([])
    window_3D = gl.GLViewWidget()
    window_3D.opts['distance'] = 500
    window_3D.show()
    window_3D.setWindowTitle('Reconstructed Object')
    
    ## Khởi tạo vật thể 3D từ ma trận màu 4D
    volume = gl.GLVolumeItem(object_4D)
    
    ## Tọa độ của điểm tâm
    volume.translate(-100,-100,-100)
    # Hiển thị vật thể 3D
    window_3D.addItem(volume)
    
    # ## Start Qt event loop unless running in interactive mode.
    # if __name__ == '__main__':
        #     import sys
        #     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            #         QtGui.QApplication.instance().exec_()      
            # Configuration volume
    return
