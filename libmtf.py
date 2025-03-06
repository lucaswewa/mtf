import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import rotate
import PIL
import matplotlib.pyplot as plt
import math
from scipy.fft import fft, fftfreq, fftshift

def a(b):
    return b+1

def read_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return image

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class ROI:
    def __init__(self, ul, lr):
        self.ul = ul
        self.lr = lr

# get_roi
def get_roi(image, topleft, bottomright):
    ul = Point(topleft[0], topleft[1])
    lr = Point(bottomright[0], bottomright[1])
    roi = ROI(ul, lr)
    roi_image = image[roi.ul.y : roi.lr.y, roi.ul.x : roi.lr.x]
    return (roi_image, roi)

# get_hamming
def get_hamming(window_width, mid = None):
    if mid is None:
        mid = (window_width - 1) / 2
        
    wid1 = mid
    wid2 = window_width - 1 - mid
    wid = max(wid1, wid2)
    pie = math.pi
    data = [0.54+0.45*math.cos(pie*(i-mid)/wid) for i in range(window_width)]
    return data

# get_deriv1
def get_deriv1(a, fil):
    result = np.convolve(a, fil, mode="same")
    result[0] = result[1]
    result[-1] = result[-2]
    return result

# get_centroid
def get_centroid(a): 
#     print(a.shape)
    dist = np.arange(a.shape[1])
    # print(dist.shape)
    weight = a
    moment = weight * dist
#     print(moment.shape)
    weight_sum = np.sum(weight, axis=1)
    moment_sum = np.sum(moment, axis=1)
    result = moment_sum/weight_sum
    return result

# get_polyfit
def get_polyfit(x, y, deg):
    p = np.polyfit(x, y, deg, full=False)
    return p

def get_fir2fix(n, m):
    """
    n = frequency data length [0-half-sampling (Nyquist) frequency]
    m = length of difference filter
        e.g. 2-point difference m = 2
             3-point difference m = 3
             
    returns: nxl MTF correction array (limited to a maximum of 10)
    """
    m = m - 1
    scale = 1
    correct = [abs((math.pi * (i+1) * m/(2*(n+1)))/math.sin(math.pi*(i+1)*m/(2*(n+1)))) for i in range(n)]
    correct[0] = 1
    correct = np.array(correct)
    gt10 = correct > 10
    correct[gt10] = 10
    return correct
    
def project2(ary, fit, fac = 4):
    """
    Projects the data in array ary along the direction defined by
      npix = (1/slope)*nlin.
      
    Data is accumulated in 'bins' that have width (1/fac) pixel.
    
    The smooth, supersampled one-dimensional vector is returned.
    
    ary: input data array
    fit: the polynomial fit
    fac: oversampling (binning) factor, default = 4
    """
    
    nlin = ary.shape[0]
    npix = ary.shape[1]
    slope = fit[0]
    nn = math.floor(npix * fac)
    slope = 1/slope
    offset = round(fac * (0 - (nlin - 1) / slope))
    del1 = abs(offset)
    if offset > 0:
        offset = 0
        
    bwidth = nn + del1 + 150
    
    # projection and binning
    p2 = [np.polyval(fit, y) - fit[1] for y in range(nlin)]
    
    barray = np.zeros((2, bwidth))
    
    for n in range(npix):
        for m in range(nlin):
            x = n
            y = m
            ling = math.ceil((x-p2[m])*fac) + 1 - offset
            if ling < 1:
                ling = 1
            elif ling > bwidth:
                lin = bwidth
                
            barray[0, ling] = barray[0, ling] + 1
            barray[1, ling] = barray[1, ling] + ary[m, n]
    
#     print(barray[:, 0:200])
    
    # TODO: check for zero counts
    start = 1 + round(0.5 * del1)
#     print(start)
    esf = [barray[1, i + start]/barray[0, i + start] for i in range(nn)]
#     print(esf)
    return esf