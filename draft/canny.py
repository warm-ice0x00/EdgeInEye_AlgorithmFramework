import cv2
import numpy as np

# 1. 读取图像（Canny 必须在灰度图上进行）
image = cv2.imread('../img/base/img_1.png')  # 请替换为你的图片路径
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 高斯模糊降噪（非常重要！抑制无效的噪声边缘）
# (5, 5) 是高斯核大小，数字越大越模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. 运行 Canny 边缘检测
# 100 和 200 分别是低阈值（threshold1）和高阈值（threshold2）
edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

# # 4. 显示结果
# cv2.imshow('Original', image)
# cv2.imshow('Canny Edges', edges)
#
# # 等待按键后关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import matplotlib.pyplot as plt

plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()

# save
cv2.imwrite('../img/canny/img_1.png', edges)  # 请替换为你想保存的路径

plt.imshow(image)
plt.axis('off')
plt.show()