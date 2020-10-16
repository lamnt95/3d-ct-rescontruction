from __future__ import division
import numpy as np
from os.path import join
import astra
import cv2

# Directory
output_dir = r'H1e12-n25-720' 

# Thông số cấu hình hình chiếu
SDD = 760                                # Source to Detector Distance [mm] 
SOD = 640                                # Source to Object Distance   [mm]
ODD = SDD - SOD                          # Object to Detector Distance [mm] 
detector_pixel_size = 0.1              # Pixel size                  [mm] 
detector_rows = 400                      # Vertical size of detector   [pixels].
detector_cols = 400                      # Horizontal size of detector [pixels].
num_of_projections = 720
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)

# Load projections.
projections = np.zeros((detector_rows, num_of_projections, detector_cols))
for i in range(num_of_projections):
    #im = imread('A4124.png').astype(float)
    im = cv2.imread('1e12-n25.png')                # Đọc ảnh RGB
    #  im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   # Convert RGB sang ảnh thang xám
    #  im = im /2                                  # Giảm giá trị đồ sáng
    # im = -np.log(im/np.max(im))                 # Nghịch đảo màu cho giống với ảnh X quang
    projections[:, i, :] = im

""" Creating a 3D cone beam geometry """                         
        # det_spacing_x: distance between the centers of two horizontally adjacent detector pixels
        # det_spacing_y: distance between the centers of two vertically adjacent detector pixels
        # det_row_count: number of detector rows in a single projection
        # det_col_count: number of detector columns in a single projection
        # angles: projection angles in radians
        # source_origin: distance between the source and the center of rotation (center of Object)
        # origin_det: distance between the center of rotation and the detector array
proj_geom = astra.create_proj_geom('cone',  1, 1, 
                                   detector_rows, detector_cols, angles, 
                                   SOD/detector_pixel_size, ODD/detector_pixel_size)

# Khởi tạo biến lưu trữ hình chiếu (ID - kiểu int), đây là một sinogram 3D với thông số
# hình học chiếu lấy từ biến proj_geom. Biến projections_id được dùng để định danh 
# hình chiếu và được sử dụng để cấu hình thuật toán chiếu ngược sau này.
projections_id = astra.data3d.create('-sino', proj_geom, projections)

# Khởi tạo không gian tái tạo vật thể
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                          detector_rows)
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

# Thu dữ liệu tái tạo, dữ liệu trả về có dạng numpy.ndarray
reconstruction = astra.data3d.get(reconstruction_id)

# Loại bỏ giá trị âm của dữ liệu tái tạo gây ra do noise
# Định dạng lại dữ liệu ra có kiểu 8 bit
reconstruction[reconstruction < 0] = 0
reconstruction /= np.max(reconstruction)
reconstruction = np.round(reconstruction * 255).astype(np.uint8)

# Saving reconstructed image.
for i in range(detector_rows):
    recon_img = reconstruction[:, i, :]*2       # Tăng độ sáng của ảnh
    #recon_img = np.flipud(recon_img)
    cv2.imwrite(join(output_dir, 'reco%04d.png' % i), recon_img)


# Cleanup.
astra.algorithm.delete(algorithm_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projections_id)
