"""
这个函数主要是用来存放一些环境相关的工具函数，以及引入一些环境相关的包
"""

import numpy as np
import torch
from typing import *
import cv2

RowCol = np.ndarray
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")