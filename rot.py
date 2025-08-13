import cv2
from scipy import ndimage, datasets

files = [(f"xcube_data/0803/mirror_n0p65d_350us_0/{idx/1000:.3f}.png", f"xcube_data/0803/mirror_n0p65d_350us_0/{idx/1000:.3f}_rot5.png") for idx in range(20110, 20800, 10) ]

for (fi, fo) in files:
    img = cv2.imread(fi)
    img_5 = ndimage.rotate(img, -5, reshape=False)
    cv2.imwrite(fo, img_5)