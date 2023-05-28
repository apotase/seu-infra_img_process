import numpy as np
import cv2

# 定义铁红映射
cdict = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
        }

# 创建颜色映射查找表
lut = np.zeros((256, 1, 3), dtype=np.uint8)

for i in range(256):
    r = np.interp(i/256.0, cdict['red'][:,0], cdict['red'][:,1])
    g = np.interp(i/256.0, cdict['green'][:,0], cdict['green'][:,1])
    b = np.interp(i/256.0, cdict['blue'][:,0], cdict['blue'][:,1])
    lut[i, 0, 0] = b*255
    lut[i, 0, 1] = g*255
    lut[i, 0, 2] = r*255

# 读取图像
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# 应用颜色映射
img_color = cv2.LUT(img, lut)

# 显示图像
cv2.imshow('image', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
