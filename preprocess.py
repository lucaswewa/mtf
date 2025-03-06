import cv2
import numpy as np

filenames = [
    ("test_red_horizontal+22deg_22m_rot.png", 1395, 1527, 993, 1090, 'h'),
    ("test_red_horizontal+22deg_22m_rot.png", 1224, 1388, 1136, 1268, 'v'),
    ("test_red_horizontal-22deg_22m_rot.png", 1172, 1356, 1328, 1500, 'h'),
    ("test_red_horizontal-22deg_22m_rot.png", 1016, 1208, 1500, 1620, 'v'),
    ("test_red_horizontal0deg_22m_rot.png", 1256, 1392, 976, 1128, 'h'),
    ("test_red_horizontal0deg_22m_rot.png", 1060, 1248, 1144, 1260, 'v'),
    ("test_red_vertical+22deg_22m_rot.png", 1964, 2080, 708, 900, 'h'),
    ("test_red_vertical+22deg_22m_rot.png", 1736, 1896, 888, 1004, 'v'),
]

for f in filenames:
    img = cv2.imread(f[0], cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    cv2.imwrite(f"{f[5]}_{f[0]}", img[f[1]:f[2], f[3]:f[4]])
