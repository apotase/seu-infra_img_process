import cv2
import numpy as np
cold = cv2.imread('data/hall_bar/0mA.tiff',cv2.IMREAD_GRAYSCALE)
hot = cv2.imread("data/hall_bar/10mA.tiff",cv2.IMREAD_GRAYSCALE)
color_map = cv2.applyColorMap(cold-hot, cv2.COLORMAP_JET)
cv2.imshow('test',color_map)
cv2.waitKey(0)
cv2.destroyAllWindows()