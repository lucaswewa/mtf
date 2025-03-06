import cv2
import numpy as np

img = cv2.imread("test_red_horizontal+22deg_22m_rot.png", cv2.IMREAD_GRAYSCALE)
print(img.shape)
cv2.imwrite("h.png", img[1395:1527, 993:1090])
