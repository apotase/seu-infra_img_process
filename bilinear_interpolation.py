import numpy as np
import cv2
from k_means import Kmeans
def bilinear_interpolation(img):
    h, w = img.shape[:2]
    out_h, out_w = 2 * h, 2 * w
    out_img = np.zeros((out_h, out_w), dtype=np.uint8)

    for i in range(out_h):
        for j in range(out_w):
            x, y = i/2, j/2
            if x % 1 == 0 and y % 1 == 0:
                out_img[i, j] = img[int(x), int(y)]
            elif x % 1 == 0:
                out_img[i, j] = (1 - (y % 1)) * img[int(x), int(y)] + (y % 1) * img[int(x), int(y) + 1]
            elif y % 1 == 0:
                out_img[i, j] = (1 - (x % 1)) * img[int(x), int(y)] + (x % 1) * img[int(x) + 1, int(y)]
            else:
                w00 = (1 - (x % 1)) * (1 - (y % 1))
                w01 = (1 - (x % 1)) * (y % 1)
                w10 = (x % 1) * (1 - (y % 1))
                w11 = (x % 1) * (y % 1)
                out_img[i, j] = w00 * img[int(x), int(y)] + w01 * img[int(x), int(y) + 1] \
                                + w10 * img[int(x) + 1, int(y)] + w11 * img[int(x) + 1, int(y) + 1]

    return out_img


img = cv2.imread('data/FW_2_10mA.tiff')[200:310,300:400]
# img = cv2.imread('data/hall_bar/0mA.tiff')[185:285,250:355]
k = Kmeans(img)
labels,centers = k.process()
mask = (labels == 0)  # 只保留类别为 0 的像素
output_img = np.zeros_like(img)
output_img[mask.reshape(img.shape[:2])] = img[mask.reshape(img.shape[:2])]
mask1 = (mask==0)
mask1 = mask1.astype(np.uint8) * 255
mask1 = mask1.reshape(img.shape[:2])
cv2.imwrite('test0.png',output_img)
dst = cv2.inpaint(img, mask1, 10, cv2.INPAINT_TELEA)
cv2.imwrite('test.jpg',dst)
print(labels,centers)