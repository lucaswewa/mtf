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

filenames = [
    ("red_hori_-22deg_3333_rot.png", 988, 1076, 1300, 1404, 'h'),
    ("red_hori_-22deg_3333_rot.png", 816, 964, 1400, 1500, 'v'),
    ("red_hori_0deg_1111_rot.png", 932, 1072, 960, 1044, 'h'),
    ("red_hori_0deg_1111_rot.png", 752, 900, 1068, 1148, 'v'),
    ("red_hori_+22deg_2222_rot.png", 1268, 1368, 1056, 1220, 'h'),
    ("red_hori_+22deg_2222_rot.png", 1120, 1280, 1220, 1340, 'v'),

    ("green_hori_-22deg_1666_rot.png", 1340, 1440, 864, 984, "h"),
    ("green_hori_-22deg_1666_rot.png", 1172, 1332, 956, 1064, "v"),
    ("green_hori_0deg_1111_rot.png", 1356, 1436, 876, 992, "h"),
    ("green_hori_0deg_1111_rot.png", 1240, 1352, 972, 1060, "v"),
    ("green_hori_+22deg_1111_rot.png", 1352, 1432, 876, 996, "h"),
    ("green_hori_+22deg_1111_rot.png", 1204, 1348, 992, 1072, "v"),

    ("blue_hori_-22deg_4444_rot.png", 1224, 1324, 284, 464, "h"),
    ("blue_hori_-22deg_4444_rot.png", 1044, 1248, 464, 544, "v"),
    ("blue_hori_0deg_4444_rot.png", 1176, 1260, 1180, 1324, "h"),
    ("blue_hori_0deg_4444_rot.png", 1056, 1184, 1328, 1408, "v"),
    ("blue_hori_+22deg_5555_rot.png", 1284, 1372, 448, 604, "h"),
    ("blue_hori_+22deg_5555_rot.png", 1148, 1296, 600, 680, "v"),

    ("white_baseline_222_rot.png", 1389, 1455, 786, 891, "h"),
    ("white_baseline_222_rot.png", 1272, 1383, 912, 984, "v"),
]

for f in filenames:
    img = cv2.imread(f[0], cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    cv2.imwrite(f"{f[5]}_{f[0]}", img[f[1]:f[2], f[3]:f[4]])
