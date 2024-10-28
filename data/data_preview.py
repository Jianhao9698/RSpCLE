import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import os


bin_filename = '/Users/vault/PycharmProjects/data/MCM_67/STAINED' \
               '/CONFOCAL/Run_1/2021_07_14_14_49_41.bin'
txt_filename = '/Users/vault/PycharmProjects/data/MCM_67/STAINED' \
               '/CONFOCAL/Run_1/2021_07_14_14_49_41.txt'

# data_root = './data'
# filename_list = os.listdir(data_root)
# bin_filename = 'your_data.bin'
# txt_filename = 'your_data.txt'
# 数据类型和图像尺寸
dtype = np.uint8
width1, height1, num_frames1 = 512, 512, 446
width2, height2, num_frames2 = 256, 256, 1784
width3, height3, num_frames3 = 128, 128, 7136
width4, height4, num_frames4 = 64, 64, 28544



data1 = np.fromfile(bin_filename, dtype=dtype, count=width1 * height1 * num_frames1)# 读取二进制文件
data1 = data1.reshape((num_frames1, height1, width1))# 重塑数据为 (帧数, 高度, 宽度)

data2 = np.fromfile(bin_filename, dtype=dtype, count=width2 * height2 * num_frames2)
data2 = data2.reshape((num_frames2, height2, width2))

data3 = np.fromfile(bin_filename, dtype=dtype, count=width3 * height3 * num_frames3)
data3 = data3.reshape((num_frames3, height3, width3))

data4 = np.fromfile(bin_filename, dtype=dtype, count=width4 * height4 * num_frames4)
data4 = data4.reshape((num_frames4, height4, width4))

plt.figure
plt.subplot(2, 2, 1)
# 显示第一帧图像
plt.imshow(data1[128], cmap='gray')
plt.title('Frame 1')
plt.axis('off')
plt.subplot(2, 2, 2)
# 显示第一帧图像
plt.imshow(data2[128], cmap='gray')
plt.title('Frame 2')
plt.axis('off')
plt.subplot(2, 2, 3)
# 显示第一帧图像
plt.imshow(data3[128], cmap='gray')
plt.title('Frame 3')
plt.axis('off')
plt.subplot(2, 2, 4)
# 显示第一帧图像
plt.imshow(data4[128], cmap='gray')
plt.title('Frame 4')
plt.axis('off')

plt.show()