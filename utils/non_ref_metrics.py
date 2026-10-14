import cv2
import numpy as np

def calculate_uciqe(img):
    # UCIQE 计算逻辑 (简版)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    # 饱和度 (chroma) 的标准差
    chroma = np.sqrt(a.astype(np.float32)**2 + b.astype(np.float32)**2)
    sigma_c = np.std(chroma)
    # 亮度的对比度
    con_l = np.max(l) - np.min(l)
    # 平均饱和度
    mu_c = np.mean(chroma)
    # 经验系数 (标准公式)
    uciqe = 0.4662 * sigma_c + 0.2745 * con_l + 0.2593 * mu_c
    return uciqe / 100.0 # 归一化处理
