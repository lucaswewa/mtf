import numpy as np
import matplotlib.pyplot as plt

from quickMTF.sfr_mtf import sfr_mtfcal

cal = sfr_mtfcal()

image = np.load(r"..\quickMTF\sample_vert_edge.npy")
image = np.load(r"..\quickMTF\sample_hori_edge.npy")

oversampling = 6
sample = image
show_plots = 5
mtf_index = 0.5
sequence=0
return_fig = False

mtf, status = cal.calc_sfr(sample, oversampling, show_plots=show_plots, verbose=True,
                                                return_fig=return_fig)

print()

mtf_quadr, status_quadr = cal.calc_sfr(sample, oversampling, show_plots=0, verbose=True,
                                                            return_fig=return_fig, quadratic_fit=True)

print()

angle = status["angle"]
filtered_first_elements = mtf[:, 1]
absolute_diff = np.abs(filtered_first_elements - mtf_index)
closest_index = np.argmin(absolute_diff)
cp_filter = mtf[closest_index, 0]
filtered_first_elements = mtf[:, 0]
absolute_diff = np.abs(filtered_first_elements - 0.5)
closest_index = np.argmin(absolute_diff)
mtf_nyquist = mtf[closest_index, 1] * 100
if show_plots >= 1:
    plt.figure()
    # Set dark mode color scheme
    plt.style.use('classic')
    plt.plot(mtf[:, 0], mtf[:, 1], '.-', label="linear fit to edge")
    plt.plot(mtf_quadr[:, 0], mtf_quadr[:, 1], '.-', label="quadratic fit to edge")
    f = np.arange(0.0, 1.0, 0.01)
    mtf_sinc = np.abs(np.sinc(f))
    plt.plot(f, mtf_sinc, 'k-', label="sinc")
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    textstr = f"Edge angle: {status['angle']:.1f}°"
    props = dict(facecolor='w', alpha=0.5)
    ax = plt.gca()
    plt.text(0.65, 0.60, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
    plt.grid()
    shape = f'{sample.shape[1]:d}x{sample.shape[0]:d} px'
    noise = 'out noise'
    plt.title(
        f'seq:{sequence}:SFR from {shape:s} curved slanted edge\nwith{noise:s}, edge angle={angle:.1f}°')
    plt.ylabel('MTF')
    plt.xlabel('Spatial frequency (cycles/pixel)')
    plt.legend(loc='best')
    plt.show()
cp_filter = cp_filter.ravel()[0]
a, b, c = round(cp_filter, 2), round(angle, 2), round(mtf_nyquist, 2)
print(a, b, c)
