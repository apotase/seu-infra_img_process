import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# 读取灰度图像
img_gray = cv2.imread('data/FW_1_0mA.tiff', cv2.IMREAD_GRAYSCALE)
# cv2.imwrite('gray.png',img_gray)
# 将像素值转换为浮点数类型
img_gray = np.float32(img_gray)
cv2.imwrite('gray.png',img_gray)
# 使用 Sobel 算子计算 x 和 y 方向上的梯度
grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)

# 计算梯度的幅值
grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
# 使用 Matplotlib 库绘制三维图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(np.arange(img_gray.shape[1]), np.arange(img_gray.shape[0]))
ax.plot_surface(x, y, img_gray, cmap='gray')
plt.savefig('3.png')
