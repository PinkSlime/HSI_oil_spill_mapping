import os
import numpy as np
import random
import openpyxl
import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import io
from sklearn import metrics
import  time
from sklearn.decomposition import PCA
import cv2

import wzh_spec



# setting parameters
DataPath = 'D:/oil/oil.mat'
TRPath = 'D:/oil/TR.mat'
GTPath = 'D:/oil/.mat'



# load data
Data = io.loadmat(DataPath)
TrLabel = io.loadmat(TRPath)
GT = io.loadmat(GTPath)

# Data =  h5py.File('DataPath','r')
# TrLabel =h5py.File('TRPath')
# TsLabel = h5py.File('TSPath')
# GT = h5py.File('GTPath')

Data = Data['img']
Data = Data.astype(np.float16)
TrLabel = TrLabel['Tr']

GT = GT['GT']

TsLabel = GT - TrLabel

# color_map = np.array([
#     [0, 0, 0],
#     [120, 152, 225],
#     [118, 218, 145],
#     [248, 149, 136],
#     [248, 203, 127],
#     [124, 214, 207],
#     [250, 109, 29],
#     [153, 135, 206],
#     [145, 146, 171],
#     [118, 80, 5]
# ])
# color_map = np.array([
#     [0, 0, 0],
#     [245, 147, 17],
#     [22, 175, 204],
#     [209, 42, 106],
# ])
color_map = np.array([
    [0, 0, 0],
    [255, 243, 219],
    [172, 165, 199],
    [95, 69, 255]
])

my_color_map  =color_map /255
out_put,prediction_matrix=wzh_spec.wzh_oil(Data ,TsLabel,TrLabel,GT, my_color_map  )

color_seg = np.zeros((GT.shape[0], GT.shape[1], 3), dtype=np.uint8)
for label, color in enumerate(color_map):
    color_seg[prediction_matrix == label, :] = color
color_seg = color_seg[..., ::-1]

cv2.imwrite('D:/oil/add/DWH/re/data/re.png', color_seg)


excel_name = openpyxl.load_workbook('D:/oil/add/DWH/re/sf.xlsx')

excel= excel_name['Sheet1']

for out_put_index in range (6):
    excel[ '{}{}'.format(chr(66+out_put_index))] =out_put[out_put_index]
# for out_put_index in range(8):
#     excel['{}{}'.format(chr(66 + out_put_index), 55)] = out_put[out_put_index]


excel_name.save('D:/oil/add/DWH/re/sf.xlsx')





