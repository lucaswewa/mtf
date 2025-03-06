import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import rotate
import PIL
import matplotlib.pyplot as plt
import math
from scipy.fft import fft, fftfreq, fftshift
from libmtf import read_image, Point, ROI, get_roi, get_hamming, get_deriv1, get_centroid, get_polyfit, get_fir2fix, project2

# step 0
cross_hair_image = cv2.imread("v.png", cv2.IMREAD_GRAYSCALE)

# step 1.
is_edge = False
roi_image = cross_hair_image

# step 2: for edge

# step 3: get derivative
if is_edge:
    """"""
else:
    deriv = roi_image

# step 4: apply window and compute centroid
centroid = get_centroid(deriv) -0.5

x = np.arange(len(centroid))
y = centroid
deg = 1
fit = get_polyfit(x, y, deg)


# step 5: compute polynomial fit to central locations
centroid_place = np.polyval(fit, np.arange(len(centroid)))
# print(centroid_place)
hamming_width = deriv.shape[1]


win2 = [get_hamming(hamming_width, centroid_place[i]) for i in range(len(centroid_place))]
win2 = np.array(win2)
wflag=0
# print("wflag:", wflag)
# print("deriv:", deriv)
# print("win2:", win2)
if wflag == 1:
    deriv_hamming_windowed = deriv * win2
else:
    deriv_hamming_windowed = deriv
    
# plt.imshow(deriv_hamming_windowed)

centroid = get_centroid(deriv_hamming_windowed)
# print(centroid)
# plt.plot(centroid, np.arange(len(centroid)), 'r')

x = np.arange(len(centroid))
y = centroid
# print(x)
# print(y)
fit = get_polyfit(x, y, deg)
# print("fit:", fit)
# plt.plot(np.polyval(fit, x), x, "y")


slout = -fit[0]
# print("slope:", slout)
slout = 180*math.atan(slout)/math.pi
# print("slope angle of fitme (deg):", slout)

# Evaluate equation at the middle line as edge location
midloc = np.polyval(fit, deriv.shape[0]/2)

# Limit number of lines to integer (npix*line slope as per ISO 12233
nlin = deriv.shape[0]
a = math.floor(nlin*abs(slout))
b = abs(slout)
nlinl = round(a/b)
# print("limit of line number:", nlinl)

# Limit the number of lines to integer(npix*line slope) as per ISO 12233
roi_image = roi_image[:nlinl, :]
# print("roi_image shape limited to:", roi_image.shape)
# plt.imshow(roi_image)

vslope = -fit[0]
slope_deg = slout

delimage = del1

# correct sampling interval for sampling normal to edge
delfac = math.cos(math.atan(vslope))
# print("delfac:", delfac)

# input pixel sampling normal to edge
del1n = del1*delfac
# print("del1n:", del1n)

# super-sampling interval normal to edge
del2 = del1n/nbin
# print("del2", del2)

nn = math.ceil(deriv.shape[1]*nbin)
nn2 = math.floor(nn/2) + 1
# print("nn, nn2:", nn, nn2)

# dcorr corrects SFR for response of FIR filter
dcorr = get_fir2fix(nn2, 3)

freqlim = 1
if nbin == 1:
    freqlim = 2
    
nn2out = round(nn2*freqlim/2)

# half-sampling frequency
nfreq = nn/(2*delimage*nn)

print(cross_hair_image.shape)