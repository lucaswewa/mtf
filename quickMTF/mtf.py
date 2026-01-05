import numpy as np
import numpy.typing as npt
import scipy.signal
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass, field
from .helper import differentiate, centroid, find_edge, calc_distance, pick_valid_roi_rotation, project_and_bin, filter_window, calc_mtf

@dataclass
class cSet:
    x: np.ndarray
    y: np.ndarray

@dataclass
class cESF:
    raw_esf: cSet
    interp_esf: cSet
    threshold: float
    width: float
    angle: float
    edge_poly: np.ndarray

@dataclass
class cMTF:
    x: np.ndarray
    y: np.ndarray
    mtf_at_nyquist: float
    width: float

@dataclass
class cSFRSetttings:
    super_sampling: int = 4
    mtf_index: float = 0.5
    diff_kernel: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, -0.5]))
    diff_offset: float = 0.0
    diff_ft: int = 2  # factor used in the correction of the numerical derivation
    sequence: int = 0
    show_plots: int = 5
    return_fig: bool = False

@dataclass
class Centroid_Info:
    image_for_mtf: np.ndarray
    diff: np.ndarray
    centr: np.ndarray
    win: np.ndarray
    win_width: int
    sum_arr: np.ndarray
    sum_arr_x: np.ndarray
    rotated: bool

@dataclass
class Edge_Info:
    pcoefs: np.ndarray
    slope: float
    offset: float
    angle: float
    idx: list
    patch_shape: list
    centr: np.ndarray
    dist: np.ndarray

@dataclass
class ROI_Info:
    image: np.ndarray
    rotated: bool
    centroid_info: Centroid_Info
    edge_info: Edge_Info


@dataclass
class ESF_Info:
    esf: np.ndarray
    super_sampling: int

@dataclass
class Window_Info:
    hann_win: np.ndarray
    hann_width: int
    idx2: list

@dataclass
class LSF_Info:
    lsf: np.ndarray
    window_info: Window_Info

@dataclass
class MTF_Info:
    mtf_result: np.ndarray
    cp_filter: np.ndarray
    angle: float
    mtf_nyquist: float

class SFR:
    def __init__(self, original_roi_image: np.ndarray, sfr_settings: cSFRSetttings):
        self.original_roi_image = original_roi_image
        self.sample_image = original_roi_image
        self.sample_rotated = False
        self.sfr_settings = sfr_settings

    def calc_roi_info(self, image):
        # calculate centroids for the ROI
        diff = differentiate(image, self.sfr_settings.diff_kernel)
        centr, win, win_width, sum_arr, sum_arr_x = centroid(diff, verbose=True)
        centr = centr + self.sfr_settings.diff_offset

        # find edge
        pcoefs, slope, offset, angle, idx, patch_shape, centr = find_edge(centr, image.shape, False, angle=None, verbose=False)

        if abs(angle) < 0.9 : # ingore the less than 0.9 degree slant edge
            print("angle is less than 0.9 degs")

        quadratic_fit = False
        pcoefs = [0.0, slope, offset] if not quadratic_fit else pcoefs
        dist = calc_distance(image.shape, pcoefs, quadratic_fit=quadratic_fit)

        centroid_info = Centroid_Info(
            image_for_mtf=image, 
            diff=diff, 
            centr=centr, 
            win=win,
            win_width=win_width,
            sum_arr=sum_arr, 
            sum_arr_x=sum_arr_x, 
            rotated=False)
        edge_info = Edge_Info(
            pcoefs=pcoefs, 
            slope=slope, 
            offset=offset, 
            angle=angle, 
            idx=idx, 
            patch_shape=patch_shape, 
            centr=centr, 
            dist=dist)

        roi_info = ROI_Info(
            image = image,
            rotated = False,
            centroid_info=centroid_info,
            edge_info=edge_info)
        
        return roi_info
                
    def preprocess(self):
        # calculate centroids for the ROI
        self.original_roi_info = self.calc_roi_info(self.sample_image)

        # calculate centroids for rotated ROI
        image_rot90 = self.sample_image.T[:, ::-1]  # rotate by transposing and mirroring
        self.rotated_roi_info = self.calc_roi_info(image_rot90)

        self.roi_info = pick_valid_roi_rotation(
            self.original_roi_info,
            self.rotated_roi_info)

    def calculate_esf(self):
        esf = project_and_bin(
            self.roi_info.centroid_info.image_for_mtf, 
            self.roi_info.edge_info.dist, 
            self.sfr_settings.super_sampling)  # edge spread function

        self.esf_info = ESF_Info(
            esf=esf, 
            super_sampling=self.sfr_settings.super_sampling)

    def calculate_lsf(self):
        lsf = differentiate(self.esf_info.esf, self.sfr_settings.diff_kernel)

        hann_win, hann_width, idx2 = filter_window(lsf, self.sfr_settings.super_sampling)  # define window to be applied on LSF
        if hann_width > 350:  # sorting out no slant edge
            print("wrong!")

        window_info = Window_Info(
            hann_win = hann_win,
            hann_width = hann_width,
            idx2 = idx2)
        self.lsf_info = LSF_Info(
            lsf = lsf, 
            window_info = window_info)

    def calculate_mtf(self):
        mtf_result = calc_mtf(
            self.lsf_info.lsf, 
            self.lsf_info.window_info.hann_win, 
            self.lsf_info.window_info.idx2, 
            self.sfr_settings.super_sampling, 
            self.sfr_settings.diff_ft)

        filtered_first_elements = mtf_result[:, 1]
        absolute_diff = np.abs(filtered_first_elements - self.sfr_settings.mtf_index)
        closest_index = np.argmin(absolute_diff)
        cp_filter = mtf_result[closest_index, 0]
        filtered_first_elements = mtf_result[:, 0]
        absolute_diff = np.abs(filtered_first_elements - 0.5)
        closest_index = np.argmin(absolute_diff)
        mtf_nyquist = mtf_result[closest_index, 1] * 100

        self.mtf_info = MTF_Info(
            mtf_result=mtf_result, 
            cp_filter=round(cp_filter, 2), 
            angle=round(self.roi_info.edge_info.angle, 2), 
            mtf_nyquist=round(mtf_nyquist, 2)
            )

        angle_cw = self.roi_info.rotated * 90 - self.roi_info.edge_info.angle  # angle clockwise from vertical axis
        self.status_info = {
            "rotated": self.roi_info.rotated,
            "angle": angle_cw,
            "offset": self.roi_info.edge_info.offset
        }


    def get_mtf(self):
        # preprocess
        self.preprocess()

        # ESF
        self.calculate_esf()

        # LSF
        self.calculate_lsf()

        # MTF
        self.calculate_mtf()

        return self.original_roi_info, self.rotated_roi_info, self.roi_info, self.esf_info, self.lsf_info, self.mtf_info, self.status_info

def plot_roi_image(image):
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='gray')
    fig.colorbar(im, ax=ax)
    plt.title("ROI image")
    plt.show()

def plot_centroid_and_stats(roi_info: ROI_Info):

    diff = roi_info.centroid_info.diff
    centr = roi_info.centroid_info.centr
    win = roi_info.centroid_info.win
    win_width = roi_info.centroid_info.win_width
    sum_arr = roi_info.centroid_info.sum_arr
    sum_arr_x = roi_info.centroid_info.sum_arr_x

    fig = plt.figure(figsize=(10, 5))

    ax00 = fig.add_subplot(2, 3, 1)
    im00 = ax00.imshow(diff, cmap="viridis", interpolation='nearest')
    fig.colorbar(im00)
    ax00.set_title("row diff heatmap")

    ax01 = fig.add_subplot(2, 3, 2, projection="3d")
    X = np.arange(0, diff.shape[1])
    Y = np.arange(0, diff.shape[0])
    X, Y = np.meshgrid(X, Y)
    Z = diff
    surf = ax01.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
    ax01.set_title("row diff 3d map")

    ax02 = fig.add_subplot(2, 3, 3)
    im02 = ax02.imshow(win)
    fig.colorbar(im02)
    ax02.set_title(f"win map - win_width:{win_width}")

    ax10 = fig.add_subplot(2, 3, 4)
    ax10.plot(sum_arr)
    ax10.set_title("sum_arr: sum on win_width")

    ax11 = fig.add_subplot(2, 3, 5)
    ax11.plot(sum_arr_x)
    ax11.set_title("sum_arr_x: sum on row")

    ax12 = fig.add_subplot(2, 3, 6)
    ax12.plot(centr)
    ax12.set_title("row centroid positions: sum_arr_x / sum_arr")
    
    plt.tight_layout()
    plt.show()

def plot_dist_and_stats(dist):
    fig = plt.figure(figsize=(10, 4))
    
    ax00 = fig.add_subplot(1, 2, 1, projection="3d")
    X = np.arange(0, dist.shape[1])
    Y = np.arange(0, dist.shape[0])
    X, Y = np.meshgrid(X, Y)
    Z = dist
    surf = ax00.plot_surface(X, Y, Z,
                           linewidth=0, antialiased=False)
    ax00.set_title("row dist 3d map")
    
    ax01 = fig.add_subplot(1, 2, 2)
    ax01.set_title("dist map")
    im0 = ax01.imshow(dist, cmap="gray")
    fig.colorbar(im0)

    plt.tight_layout()
    plt.show()

def plot_esf_and_stats(esf):
    fig = plt.figure(figsize=(10,4))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(esf)
    ax.set_title("ESF")
    ax.grid()
    plt.tight_layout()
    plt.show()

def plot_lsf_and_stats(lsf):
    fig = plt.figure(figsize=(10,4))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(lsf)
    ax.grid()
    ax.set_title("LSF")
    plt.tight_layout()
    plt.show()

def plot_edge_and_stats(image_for_mtf, pcoefs, slope, offset, angle, idx, patch_shape, centr, rotated):
    fig, ax = plt.subplots()
    im = ax.imshow(image_for_mtf)
    plt.colorbar(im)
    if rotated:
        # ax.plot(patch_shape[1] - centr[idx], patch_shape[0] - idx, '.k', label="centroids")
        print(patch_shape[0])
        print(idx)
        ax.plot(patch_shape[1] - np.polyval([slope, offset], idx), patch_shape[0] - idx, '-', label="linear fit")
        ax.plot(patch_shape[1] - np.polyval(pcoefs, idx), patch_shape[0] - idx, '--', label="quadratic fit")
        ax.set_xlim([0, patch_shape[1]])
        ax.set_ylim([0, patch_shape[0]])
    else:
        ax.plot(centr[idx], idx, '.k', label="centroids")
        ax.plot(np.polyval([slope, offset], idx), idx, '-', label="linear fit")
        ax.plot(np.polyval(pcoefs, idx), idx, '--', label="quadratic fit")
        ax.set_xlim([0, patch_shape[1]])
        ax.set_ylim([0, patch_shape[0]])
    #ax.text("{angle}")
    # ax.set_aspect('equal', 'box')
    ax.legend(loc='best')
    ax.invert_yaxis()

def plot_filter_window_and_stats(hann_win):
    fig = plt.figure(figsize=(10,4))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(hann_win)
    ax.set_title("Hanning Window")
    ax.grid()
    plt.tight_layout()
    plt.show()

def plot_mtf_and_stats(mtf, esf, hann_win, hann_width, lsf, idx, supersampling):
    i1, i2 = idx
    nn = (i2 - i1) // 2
    lsf_sign = np.sign(np.mean(lsf[i1:i2] * hann_win))
    fig = plt.figure(figsize=(10,4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(mtf[:,0], mtf[:,1])
    ax1.grid()

    f = np.arange(0.0, 1.0, 0.01)
    mtf_sinc = np.abs(np.sinc(f))
    ax1.plot(f, mtf_sinc, 'k-', label='sinc')
    ax1.axes.set_ylim(0, 1.2)
    ax1.axes.set_xlim(0, 1.2)
    ax1.set_title("MTF")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(esf[i1:i2], 'b.-', label=f"ESF, s.s.={supersampling:2d}")
    ax2.plot(lsf_sign * lsf[i1:i2], 'r.-', label=f"{'-' if lsf_sign < 0 else ''}LSF")
    ax2.plot(hann_win * ax2.axes.get_ylim()[1] * 1.1, 'g.-', label=f"Hann Win, w={hann_width:d}")
    ax2.set_xlim(0, 2 * nn)
    # ax2 = ax.twinx()
    # ax2.get_yaxis().set_visible(False)
    ax2.grid()
    ax2.legend(loc='upper left')
    ax2.set_xlabel('Bin no.')
    
    plt.tight_layout()
    plt.show()