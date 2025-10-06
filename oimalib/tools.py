#!/usr/bin/env python3
"""
Created on Wed Aug  7 16:31:48 2019

@author: asoulain
"""

import math
import pickle
from bisect import bisect_left, insort
from collections import deque
from itertools import islice

import numpy as np
from astropy import constants as cs, units as u
from astropy.io import fits
from matplotlib import pyplot as plt
from munch import munchify
from uncertainties import ufloat, umath, unumpy

BRG_CONT_WINDOWS = 0.1
BRG_CORE_WINDOWS = 0.002


def decompress_pickle(file):
    with open(file, "rb") as pikd:
        return pickle.load(pikd)


def compressed_pickle(title, data):
    with open(f"{title}.dpy", "wb") as pikd:
        pickle.dump(data, pikd, protocol=4)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(x, y)
    phi_deg = 360 + np.rad2deg(phi) if np.rad2deg(phi) < 0 else np.rad2deg(phi)
    return (rho, phi_deg)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def norm(tab):
    """Normalize the tab array by the maximum."""
    return tab / np.max(tab)


def rad2mas(rad):
    """Convert angle in radians to milli arc-sec."""
    mas = rad * (3600.0 * 180 / np.pi) * 10.0**3
    return mas


def mas2rad(mas):
    """Convert angle in milli arc-sec to radians."""
    rad = mas * (10 ** (-3)) / (3600 * 180 / np.pi)
    return rad


def rad2arcsec(rad):
    arcsec = rad / (1 / mas2rad(1000))
    return arcsec


def incl2elong(incl):
    elong = 1.0 / umath.cos(incl * np.pi / 180.0)
    try:
        print(f"elong = {elong.nominal_value:2.3f} +/- {elong.std_dev:2.3f}")
    except AttributeError:
        print(f"elong = {elong:2.2f}")
    return elong


def check_hour_obs(list_data):
    mjd0 = list_data[0].info.mjd
    l_hour = []
    for d in list_data:
        l_hour.append((d.info.mjd - mjd0) * 24)
    l_hour = np.array(l_hour)
    print(l_hour)
    return l_hour


def round_sci_digit(number):
    """Rounds a float number with a significant digit number."""
    ff = str(number).split(".")[0]
    d = str(number).split(".")[1]
    d, ff = math.modf(number)
    sig_digit = 1
    if ff == 0:
        res = str(d).split(".")[1]
        for i in range(len(res)):
            if float(res[i]) != 0.0:
                sig_digit = i + 1
                break
    else:
        sig_digit = 1

    return float(np.round(number, sig_digit)), sig_digit


def planck_law(T, wl, norm=False):
    h = cs.h.value
    c = cs.c.value
    k = cs.k_B.value
    sigma = cs.sigma_sb.value
    P = (4 * np.pi**2) * sigma * T**4

    # print(T, wl)
    B = ((2 * h * c**2 * wl**-5) / (np.exp(h * c / (wl * k * T)) - 1)) / 1e6  # W/m2/micron
    return B / P if norm else B  # kW/m2/sr/m


def _running_median(seq, M):
    """
    Purpose: Find the median for the points in a sliding window (odd number in size)
             as it is moved from left to right by one point at a time.
     Inputs:
           seq -- list containing items for which a running median (in a sliding window)
                  is to be calculated
             M -- number of items in window (window size) -- must be an integer > 1
     Otputs:
        medians -- list of medians with size N - M + 1
      Note:
        1. The median of a finite list of numbers is the "center" value when this list
           is sorted in ascending order.
        2. If M is an even number the two elements in the window that
           are close to the center are averaged to give the median (this
           is not by definition)
    """
    seq = iter(seq)
    s = []
    m = M // 2

    # Set up list s (to be sorted) and load deque with first window of seq
    s = [item for item in islice(seq, M)]
    d = deque(s)

    # Simple lambda function to handle even/odd window sizes
    def median():
        return s[m] if bool(M & 1) else (s[m - 1] + s[m]) * 0.5

    # Sort it in increasing order and extract the median ("center" of the sorted window)
    s.sort()
    medians = [median()]

    # Now slide the window by one point to the right for each new position (each pass through
    # the loop). Stop when the item in the right end of the deque contains the last item in seq
    for item in seq:
        old = d.popleft()  # pop oldest from left
        d.append(item)  # push newest in from right
        # locate insertion point and then remove old
        del s[bisect_left(s, old)]
        # insert newest such that new sort is not required
        insort(s, item)
        medians.append(median())
    return np.array(medians)


def nan_interp(yall):
    """
    Interpolate nan from non-nan values.
    Along the last dimension if several of them.
    """
    if len(yall.shape) > 1:
        for y in yall:
            nan_interp(y)
    else:
        nans, x = np.isnan(yall), lambda z: z.nonzero()[0]
        yall[nans] = np.interp(x(nans), x(~nans), yall[~nans])


def normalize_continuum(yall, wave, in_cont, *, degree=3, phase=False, plot=False):
    """
    data                          shape: [nbase,nwave]
    wave in [um]                  shape: [nwave]
    inCont array of True/False    shape: [nwave]

    Along the last dimension if several of them.
    Operation in done in-place.
    """
    if len(yall.shape) > 1:
        # Deal with multiple dimensions
        for y in yall:
            normalize_continuum(y, wave, in_cont, degree=degree, phase=phase)
    else:
        # Re-center the x array, to stabilize fit numerically
        x = wave - np.mean(wave)
        # Fit continuum and remove fit
        if phase:
            if plot:
                plt.figure()
                plt.plot(x[in_cont], yall[in_cont], "r.", alpha=0.5)
                plt.plot(x, yall, color="tab:blue", alpha=0.5)
                continuum = np.polyval(np.polyfit(x[in_cont], yall[in_cont], degree), x)
                plt.plot(x, continuum)
                plt.plot(x, yall - continuum)
            yall -= np.polyval(np.polyfit(x[in_cont], yall[in_cont], degree), x)
        else:
            yall /= np.polyval(np.polyfit(x[in_cont], yall[in_cont], degree), x)
    return yall


def substract_run_med(spectrum, wave=None, n_box=50, shift_wl=0, div=False):
    """Substract running median from a raw spectrum `f`. The median
    is computed at each points from n_box/2 to -n_box/2+1 in a
    'box' of size `n_box`. The Br gamma line in vaccum and telluric
    lines can be displayed if wavelengths table (`wave`) is specified.
    `shift_wl` can be used to shift wave table to estimate the
    spectral shift w.r.t. the telluric lines.
    """

    lbdBrg = 2.16625
    inCont = (np.abs(wave - lbdBrg) < BRG_CONT_WINDOWS) * (np.abs(wave - lbdBrg) > BRG_CORE_WINDOWS)
    nan_interp(spectrum)
    normalize_continuum(spectrum, wave, in_cont=inCont)
    return spectrum, wave


def hide_xlabel():
    plt.xticks(color="None")
    plt.grid(lw=0.5, alpha=0.5)


def plot_vline(x, color="#eab15d"):
    plt.axvline(x, lw=1, color=color, zorder=-1, alpha=0.5)


def super_gaussian(x, sigma, m, amp=1, x0=0):
    sigma = float(sigma)
    m = float(m)
    return amp * (
        (np.exp(-(2 ** (2 * m - 1)) * np.log(2) * (((x - x0) ** 2) / ((sigma) ** 2)) ** (m))) ** 2
    )


def apply_windowing(img, window=80, m=3):
    isz = len(img)
    xx, yy = np.arange(isz), np.arange(isz)
    xx2 = xx - isz // 2
    yy2 = isz // 2 - yy
    # Distance map
    distance = np.sqrt(xx2**2 + yy2[:, np.newaxis] ** 2)

    # Super-gaussian windowing
    window = super_gaussian(distance, sigma=window * 2, m=m)

    # Apply the windowing
    img_apod = img * window
    return img_apod


def wtmn(values, weights, axis=0, cons=False):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    values[np.isnan(values)] = 0.0
    try:
        weights[np.isnan(values)] = 0.0
    except TypeError:
        weights = np.ones_like(values)

    mn = np.average(values, weights=weights, axis=axis)

    # Fast and numerically precise:
    variance = np.average((values - mn) ** 2, weights=weights, axis=axis)

    std = np.sqrt(variance)
    std_err = std / (np.shape(values)[0]) ** 0.5

    std_unbias = std_err if not cons else std
    return (mn, std_unbias)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def binning_tab(data, nbox=50, force=False, rel_err=0.01, *, cons=False):
    """Compute spectrally binned observables using weigthed averages (based
    on squared uncertainties).

    Parameters:
    -----------

    `data` {class}:
        Data class (see oimalib.load() for details),\n
    `nbox` {int}:
        Size of the box,\n
    `flag` {bool}:
        If True, obs flag are used and avoided by the average,\n
    `force` {bool}:
        If True, force the uncertainties as the relative error `rel_err`,\n
    `rel_err` {float}:
        If `force`, relative uncertainties to be used [%].

    Outputs:
    --------
    `l_wl`, `l_vis2`, `l_e_vis2`, `l_cp`, `l_e_cp` {array}:
        Wavelengths, squared visibilities, V2 errors, closure phases and CP errors.
    """
    vis2 = data.vis2
    e_vis2 = data.e_vis2
    flag_vis2 = data.flag_vis2
    flag_cp = data.flag_cp
    cp = data.cp
    e_cp = data.e_cp
    wave = data.wl
    dvis = data.dvis
    e_dvis = data.e_dvis
    dphi = data.dphi
    e_dphi = data.e_dphi

    nwl = len(wave)
    l_wl, l_vis2, l_cp = [], [], []
    l_dvis, l_dphi, l_e_dvis, l_e_dphi = [], [], [], []
    l_e_cp, l_e_vis2 = [], []
    for i in np.arange(0, nwl, nbox):
        try:
            ind = i + nbox
            i_wl = np.mean(wave[i:ind])
            i_vis2, i_e_vis2 = [], []
            i_cp, i_e_cp = [], []
            i_dvis, i_e_dvis = [], []
            i_dphi, i_e_dphi = [], []
            for j in range(len(vis2)):
                range_vis2 = vis2[j][i:ind]
                range_e_vis2 = e_vis2[j][i:ind]
                range_dvis = dvis[j][i:ind]
                range_e_dvis = e_dvis[j][i:ind]
                range_dphi = dphi[j][i:ind]
                range_e_dphi = e_dphi[j][i:ind]
                cond_flag_vis2 = ~flag_vis2[j][i:ind]
                if len(range_vis2[cond_flag_vis2]) != 0:
                    weigths = 1.0 / range_e_vis2[cond_flag_vis2] ** 2
                    weigth_dvis = 1.0 / range_e_dvis[cond_flag_vis2] ** 2
                    weigth_dphi = 1.0 / range_e_dphi[cond_flag_vis2] ** 2
                    vis2_med, e_vis2_med = wtmn(range_vis2[cond_flag_vis2], weigths, cons=cons)
                    dvis_med, e_dvis_med = wtmn(range_dvis[cond_flag_vis2], weigth_dvis, cons=cons)
                    dphi_med, e_dphi_med = wtmn(range_dphi[cond_flag_vis2], weigth_dphi, cons=cons)
                else:
                    vis2_med, e_vis2_med = np.nan, np.nan
                    dvis_med, e_dvis_med = np.nan, np.nan
                    dphi_med, e_dphi_med = np.nan, np.nan

                if force:
                    e_vis2_med = rel_err * vis2_med

                i_vis2.append(vis2_med)
                i_e_vis2.append(e_vis2_med)
                i_dvis.append(dvis_med)
                i_e_dvis.append(e_dvis_med)
                i_dphi.append(dphi_med)
                i_e_dphi.append(e_dphi_med)

            for k in range(len(cp)):
                range_cp = cp[k][i:ind]
                range_e_cp = e_cp[k][i:ind]
                cond_flag_cp = ~flag_cp[k][i:ind]
                if len(range_cp[cond_flag_cp]) != 0:
                    weigths_cp = 1.0 / range_e_cp[cond_flag_cp] ** 2
                    cp_med, e_cp_med = wtmn(
                        range_cp[cond_flag_cp],
                        weigths_cp,
                        cons=cons,
                    )
                else:
                    cp_med, e_cp_med = np.nan, np.nan
                i_cp.append(cp_med)
                i_e_cp.append(e_cp_med)

            if (np.mean(i_e_vis2) != 0.0) and (len(wave[i:ind]) == int(nbox)):
                l_wl.append(i_wl)
                l_vis2.append(i_vis2)
                l_cp.append(i_cp)
                l_e_cp.append(i_e_cp)
                l_e_vis2.append(i_e_vis2)
                l_dvis.append(i_dvis)
                l_dphi.append(i_dphi)
                l_e_dvis.append(i_e_dvis)
                l_e_dphi.append(i_e_dphi)
        except (IndexError, ZeroDivisionError):
            pass

    l_vis2 = np.array(l_vis2).T
    l_e_vis2 = np.array(l_e_vis2).T
    l_cp = np.array(l_cp).T
    l_e_cp = np.array(l_e_cp).T
    l_wl = np.array(l_wl)
    l_dvis = np.array(l_dvis).T
    l_dphi = np.array(l_dphi).T
    l_e_dvis = np.array(l_e_dvis).T
    l_e_dphi = np.array(l_e_dphi).T
    return l_wl, l_vis2, l_e_vis2, l_cp, l_e_cp, l_dvis, l_dphi, l_e_dvis, l_e_dphi


def computeBinaryRatio(param, wl):
    """Compute binary ratio at the given wavelength according
    a total luminosity ratio of param['L_ratio_star'] (L_WR = X*L_OB).
    """
    T_Wr = param["T_WR"]
    T_ob = param["T_OB"]
    sig_Teff = T_Wr**4 / T_ob**4

    L_ratio = param["L_WR/O"]

    f1_0 = planck_law(T_ob, wl)
    f2_0 = planck_law(T_Wr, wl) * (L_ratio / sig_Teff)
    ftot_0 = f1_0 + f2_0
    p_OB = f1_0 / ftot_0
    p_WR = f2_0 / ftot_0
    return p_OB, p_WR


def get_dvis(data, bounds=None):
    """
    Get differential observables (visibility amplitude and phase). By default,
    the observable are extracted around brg (2.14-2.19 µm).

    Parameters:
    -----------

    `data` {class}:
        Interferometric data from load()\n
    `bounds` {list}:
        Wavelengths range (by default around Br Gamma line 2.166 µm, [2.14, 2.19]),\n
    `line` {float}:
        Vertical line reference to be plotted (by default, Br Gamma line 2.166 µm)\n
    """
    if bounds is None:
        bounds = [2.14, 2.19]

    spectrum = data.flux if len(data.flux.shape) == 1 else data.flux.mean(axis=0)

    wl = data.wl * 1e6

    try:
        flux, wave = substract_run_med(spectrum, wl, div=True)
    except IndexError:
        flux, wave = spectrum, wl

    cond_wl = (wave >= bounds[0]) & (wave <= bounds[1])
    cond_wl2 = (wl >= bounds[0]) & (wl <= bounds[1])

    flux = flux[cond_wl]
    wave = wave[cond_wl]

    dphi = data.dphi
    dvis = data.dvis
    blname = data.blname
    bl = data.bl

    l_dvis, l_dphi, l_blname, l_bl = [], [], [], []
    for i in range(dvis.shape[0]):
        data_dvis = dvis[i][cond_wl2]
        dvis_m = data_dvis[~np.isnan(data_dvis)].mean()
        if not np.isnan(dvis_m):
            l_dvis.append(data_dvis)
            l_blname.append(blname[i])
            l_bl.append(bl[i])

    for i in range(dphi.shape[0]):
        if np.diff(dphi[i][cond_wl2]).mean() != 0:
            l_dphi.append(dphi[i][cond_wl2])

    out = {
        "wl": wave,
        "flux": flux,
        "dwl": wl[cond_wl2],
        "dvis": l_dvis,
        "dphi": l_dphi,
        "blname": l_blname,
        "bl": l_bl,
    }

    return munchify(out)


def compute_oriented_shift(angle, shift):
    """Compute the east/north offset from the fitted angle and shift."""
    north = shift * unumpy.cos(angle * np.pi / 180.0)
    east = shift * unumpy.sin(angle * np.pi / 180.0)
    return east, north


def combine_dphi_aspro_file(d, ibl, n_per_hour=6, hour=3, cons=False):
    l_dphi = []
    l_e_dphi = []
    for i in np.arange(0, int(6 * n_per_hour * hour), 6):
        tmp = d.dphi[i + ibl]
        tmp2 = d.e_dphi[i + ibl]
        l_dphi.append(tmp)
        l_e_dphi.append(tmp2)
    l_dphi = np.array(l_dphi)
    l_e_dphi = np.array(l_e_dphi)
    dphi_aver, e_dphi_aver = wtmn(l_dphi, weights=l_e_dphi, cons=cons)
    return dphi_aver, e_dphi_aver


deg2rad = np.pi / 180.0
rad2deg = 1.0 / deg2rad


def get_mis(inc_in=None, pa_in=None, inc_out=None, pa_out=None):
    """
    calculates the misalignment angle between the inner and outer disk [radian]

    Parameters
    ----------
    inc_in          : float
                      Inclination angle of the inner disk [radian]

    pa_in           : float
                      Position angle of the inner disk [radian]

    inc_out         : float
                      Inclination angle of the outer disk [radian]

    pa_out          : float
                      Position angle of the outer disk [radian]
    """
    cos_mis = np.sin(inc_in) * np.sin(inc_out) * np.cos(pa_in - pa_out) + np.cos(inc_in) * np.cos(
        inc_out
    )
    mis = np.arccos(cos_mis)

    return mis


def get_shadow_pa(inc_in=None, pa_in=None, inc_out=None, pa_out=None):
    """
    calculates the position angleof the shadows

    Parameters
    ----------
    inc_in          : float
                      Inclination angle of the inner disk [radian]

    pa_in           : float
                      Position angle of the inner disk [radian]

    inc_out         : float
                      Inclination angle of the outer disk [radian]

    pa_out          : float
                      Position angle of the outer disk [radian]
    """
    ax = np.sin(inc_in) * np.cos(inc_out) * np.cos(pa_in) - np.cos(inc_in) * np.sin(
        inc_out
    ) * np.cos(pa_out)
    ay = np.sin(inc_in) * np.cos(inc_out) * np.sin(pa_in) - np.cos(inc_in) * np.sin(
        inc_out
    ) * np.sin(pa_out)
    return np.arctan2(ay, ax) + np.pi


def get_shadow_x(inc_in=None, pa_in=None, inc_out=None, pa_out=None, h=None):
    """
    calculates the position angleof the shadows

    Parameters
    ----------
    inc_in          : float
                      Inclination angle of the inner disk [radian]

    pa_in           : float
                      Position angle of the inner disk [radian]

    inc_out         : float
                      Inclination angle of the outer disk [radian]

    pa_out          : float
                      Position angle of the outer disk [radian]
    h               : float
                      Height of the scattering surface [au]
    """
    x = (
        h
        * np.cos(inc_in)
        / (
            np.cos(inc_out) * np.sin(inc_in) * np.sin(pa_in)
            - np.cos(inc_in) * np.sin(inc_out) * np.sin(pa_out)
        )
    )
    return x


def get_misalignment(
    i_in,
    i_out,
    pa_in,
    pa_out,
    *,
    e_i_in=0,
    e_i_out=0,
    e_pa_in=0,
    e_pa_out=0,
    Rout=None,
    h=None,
):
    """Compute the misalignement angle and the line connecting the two
    projected shadow."""
    # Orientation angle of inner and outer disks

    i1 = ufloat(i_in, e_i_in) * np.pi / 180.0
    i2 = ufloat(-i_in, e_i_in) * np.pi / 180.0

    inc_in = np.array([i1, i2])
    inc_out = ufloat(i_out, e_i_out) * np.pi / 180.0

    pa_in = ufloat(pa_in, e_pa_in) * np.pi / 180.0
    pa_out = ufloat(pa_out, e_pa_out) * np.pi / 180.0

    cos_mis = unumpy.sin(inc_in) * unumpy.sin(inc_out) * unumpy.cos(pa_in - pa_out) + unumpy.cos(
        inc_in
    ) * unumpy.cos(inc_out)

    mis = unumpy.arccos(cos_mis) * 180 / np.pi
    misa = unumpy.arccos(cos_mis)

    # Line of shadow
    y = unumpy.sin(inc_in) * unumpy.cos(inc_out) * unumpy.sin(pa_in) - unumpy.cos(
        inc_in
    ) * unumpy.sin(inc_out) * unumpy.sin(pa_out)
    x = unumpy.sin(inc_in) * unumpy.cos(inc_out) * unumpy.cos(pa_in) - unumpy.cos(
        inc_in
    ) * unumpy.sin(inc_out) * unumpy.cos(pa_out)
    pa_shadow = unumpy.arctan2(y, x) * 180 / np.pi
    pa_shadow[pa_shadow < 0] = 180 + pa_shadow[pa_shadow < 0]

    # Offset
    offset_dec = None
    if Rout is not None:
        offset_dec = 2 * np.arctan(np.sqrt((np.tan(misa) ** 2 / (h / Rout) ** 2) - 1.0))

    print("\nMisalignment results:")
    print("---------------------")
    print("mis =", mis[0], "deg or", mis[1], " deg (for -i_in)")
    print("pa shadow =", pa_shadow[0], "deg or", pa_shadow[1], "deg (for -i_in)")

    return mis, pa_shadow, offset_dec


def MagToJy(m, band):
    """
    Convert Johnson magnitudes into Jy.

    Parameters:
    -----------
        m (float):
            Johnson magnitude,
        band (str):
            Photometric band name (B, V, R, L, etc.).

    Returns:
    --------
        F (float):
            Converted flux in Jy unit.
    """

    conv_flux = {
        "B": {"wl": 0.44, "F0": 4260},
        "V": {"wl": 0.5556, "F0": 3540},  # Allen's astrophysical quantities
        "R": {"wl": 0.64, "F0": 3080},
        "I": {"wl": 0.79, "F0": 2550},
        "J": {"wl": 1.235, "F0": 1594},  # 2MASS
        "H": {"wl": 1.662, "F0": 1024},  # 2MASS
        "K": {"wl": 2.159, "F0": 666.7},  # 2MASS
        "L": {"wl": 3.547, "F0": 276},
        "M": {"wl": 4.769, "F0": 160},
        # 10.2, 42.7 Johnson N (https://www.gemini.edu/?q=node/11119)
        "N": {"wl": 10.2, "F0": 42.7},
        "Q": {"wl": 20.13, "F0": 9.7},
    }
    try:
        F0 = conv_flux[band]["F0"]
        F = F0 * (10 ** (m / -2.5))
    except Exception:
        F = np.nan
    return F


def fluxToJy(flux, wl, alpha, reverse=False):
    """
    Convert flux Flambda (in different unit) to spectral
    flux density Fnu in Jansky or the reverse.

    Parameters :
    ----------

    flux {float}:
        Fλ [unit]
    wl {float}:
        Wavelenght [m]
    unit {str}:
        Unit of Fλ (see tab units)
    reverse {boolean}:
        reverse the formulae if True

    Units :
    -----

    Constant conversion depends on F_lambda unit :

    ==========================   =====   ========
    ``F_lambda measured in``     alpha   beta
    ==========================   =====   ========
    W/m2/m                       0       3x10-6
    W/m2/um                      1       3x10-12
    W/cm2/um                     2       3x10-16
    erg/sec/cm2/um               3       3x10-9
    erg/sec/cm2/Angstrom         4       3x10-13
    ==========================   =====   ========

    References :
    ----------

    [1] Link STSCI http://www.stsci.edu/hst/nicmos/documents/handbooks/current_NEW/Appendix_B.14.3.html
    [2] Wikipedia blackbody https://en.wikipedia.org/wiki/Black_body
    """

    beta_map = {
        0: 3e-6,
        1: 3e-12,
        2: 3e-16,
        3: 3e-9,
        4: 3e-13,
    }

    try:
        beta = beta_map[alpha]
    except KeyError:
        print("Bad unit of flux")
        return None

    wl2 = wl * 1e6  # wl2 in micron
    out = flux * beta / wl2**2 if reverse else flux * wl2**2 / beta
    return out


def compute_yso_carac(
    B,
    rs,
    ms,
    ls,
    *,
    rbrg=5,
    mdot=None,
    P=9,
    magK=5,
    ew=None,
    d=160.3,
    Tsub=1500,
    Q=None,
):
    """Compute the caracteristic radius of YSO. Truncation radius and
    corotation radius.

    Parameters:
    -----------
    `B` {float}: Magnetic field [kG],\n
    `rs` {float}: Stellar radius [r_sun],\n
    `ms` {float}: Stellar mass [m_sun],\n
    `mdot` {float}: Mass accretion rate [m_sun/yr],\n
    `P` {float}: Rotational period [days].

    Outputs:
    --------
    `r_tr` {float}: Truncation radius [with units],\n
    `r_co` {float}: Corotation radius [with units],\n
    """
    if isinstance(B, list):
        B = np.array(B)

    rs / 2.0
    ms / 0.5

    P = P * u.d
    omega = 2 * np.pi / P.to(u.s)
    r_co = (cs.G * ms * cs.M_sun / (omega**2)) ** (1 / 3.0)

    unit_flux = u.W / u.m**2 / u.micron
    fjy = MagToJy(magK, "K")  # Jy
    # alpha = 1 used to convert flux in W/m2/µm
    fk = fluxToJy(fjy, 2.159e-6, alpha=1, reverse=True) * unit_flux

    d = d * u.pc
    ew = ew * u.micron
    Lk = ew * fk * 4 * np.pi * (d.to(u.m) ** 2)
    Lk_sun = Lk.to(u.L_sun)

    # Uncertainties
    uew = ufloat(ew.value, ew.value * 0.01)
    print(f"EW = {uew * 1e-6 * 1e10}")
    err_d = 0.4 * u.pc
    ud = ufloat(d.to(u.m).value, err_d.to(u.m).value)
    ulk = uew * fk.value * 4 * np.pi * ud**2

    ulk_sun = ulk / cs.L_sun.value

    rs = rs * u.R_sun
    ms = ms * u.M_sun
    a = 1.16
    b = 3.6
    Lacc = 10 ** (a * np.log10(Lk_sun.value) + b) * u.L_sun

    new_alcala = True
    if not new_alcala:
        ua = ufloat(1.16, 0.07)  # ufloat(1.19, 0.10) alcala+17
        ub = ufloat(3.6, 0.38)
    else:
        ua = ufloat(1.19, 0.10)  # alcala+17
        ub = ufloat(4.02, 0.51)  # alcala+17

    uLacc = 10 ** (ua * umath.log10(ulk_sun) + ub)
    uLacc_w = uLacc * cs.L_sun

    Macc = 1.25 * Lacc.to(u.W) * rs.to(u.m) / (cs.G * ms.to(u.kg))
    Macc = Macc.to(u.M_sun / u.year)

    urs = ufloat(rs.to(u.m).value, 0.15 * rs.to(u.m).value)
    ums = ufloat(ms.to(u.kg).value, 0.0222 * ms.to(u.kg).value)
    ums2 = ufloat(ms.value, 0.0222 * ms.value)

    uP = ufloat(P.to(u.s).value, 0.00555 * P.to(u.s).value)
    uomega = 2 * np.pi / uP

    ur_co = (cs.G * ums2 * cs.M_sun / uomega**2) ** (1 / 3.0) / cs.au.value

    urbrg = ufloat(rbrg, 1)
    uMacc = (1 - (1 / urbrg)) ** -1 * uLacc_w.value * urs / (cs.G.value * ums)

    uMacc_si = uMacc / cs.M_sun.value * 60 * 60 * 24 * 365.25

    log_umacc = umath.log10(uMacc_si)

    m = round(10 ** (log_umacc.nominal_value) / 1e-8, 1)
    minf = 10 ** (log_umacc.nominal_value - log_umacc.std_dev) / 1e-8
    msup = 10 ** (log_umacc.nominal_value + log_umacc.std_dev) / 1e-8
    err_sup = round(msup - m, 1)
    err_inf = round(m - minf, 1)

    print("\nMass accretion rate:")
    print("--------------------")
    print(f"flux = {fjy:2.3f} Jy ({fk.value:2.1e} W/m2/µm)")
    print(f"Lline = {ulk} W ({ulk_sun * 1e4} Lsun)")
    print(f"Lacc = {uLacc} Lsun")
    print(f"Macc = {uMacc_si} Msun/yr (log(Macc) = {log_umacc})")
    print(f"Macc = {m}^+{err_sup}_-{err_inf} Msun/yr (log(Macc) = {log_umacc})")

    # r_tr = (
    #     12.6 * (B ** (4 / 7) * R2 ** (12.0 / 7)) / (M05 ** (1 / 7) * Mdot8 ** (2 / 7))
    # )

    urs2 = ufloat(rs.value, 0.15 * rs.value)
    ums2 = ufloat(ms.value, 0.1111 * ms.value)

    uB = ufloat(B, 0.0 * B)
    uR2 = urs2 / 2.0
    uM05 = ums2 / 0.5
    uMdot8 = uMacc_si / 1e-8

    ur_tr = 12.6 * (uB ** (4 / 7) * uR2 ** (12.0 / 7)) / (uM05 ** (1 / 7) * uMdot8 ** (2 / 7))

    ur_tr_rstar = ur_tr / urs2
    ur_tr_au = ur_tr * cs.R_sun.to(u.au).value

    r_co = np.round(r_co.to(u.au).value, 2) * u.au

    r_co_star = np.round(r_co / rs.to(u.au), 1)

    urs_au = ufloat(rs.to(u.au).value, 0.15 * rs.to(u.au).value)

    r_co_star = ur_co / urs_au

    if Q is None:
        Q = np.array([1, 4])
    r_sub = np.round(1.1 * np.sqrt(Q) * np.sqrt(ls / 1000.0) * (1500 / Tsub) ** 2, 2)
    r_sub_star = np.round(r_sub / rs.to(u.au).value, 1)
    print("\nCaracteristic radii:")
    print("--------------------")
    print(f"R_tr = {ur_tr_au} = {ur_tr_rstar} R*")
    print(f"R_co = {ur_co.value} = {r_co_star.value} R*")
    print(f"R_sub = {r_sub} AU = {r_sub_star} R*")

    return ur_tr_au, ur_co.value, r_sub, Macc


def convert_ind_data(dataset, wave_lim=None, corr_tellu=False):
    """Correct tellurics and select the wavelenght. To be consistent with
    the temporal averaging format."""
    if wave_lim is None:
        wave_lim = [2.1461, 2.1861]

    wave = dataset.wl * 1e6
    cond_wl = (wave >= wave_lim[0]) & (wave < wave_lim[1])
    wave = wave[cond_wl]

    a = np.mean(dataset.flux, axis=0)[cond_wl]
    tel_tran = np.ones_like(a)
    if corr_tellu:
        filename = dataset.info.filename
        h = fits.open(filename)
        tel_tran = h["TELLURICS"].data["TELL_TRANS"][cond_wl]
    a /= tel_tran
    corr_flux = a / a[0]

    dataset2 = dataset.copy()
    dataset2.flux = corr_flux
    dataset2.wl = wave / 1e6
    dataset2.dphi = dataset.dphi[:, cond_wl]
    dataset2.e_dphi = dataset.e_dphi[:, cond_wl]
    dataset2.dvis = dataset.dvis[:, cond_wl]
    dataset2.e_dvis = dataset.e_dvis[:, cond_wl]
    dataset2.vis2 = dataset.vis2[:, cond_wl]
    dataset2.e_vis2 = dataset.e_vis2[:, cond_wl]
    dataset2.cp = dataset.cp[:, cond_wl]
    dataset2.e_cp = dataset.e_cp[:, cond_wl]
    dataset2.flag_vis2 = dataset.flag_vis2[:, cond_wl]
    return dataset2


def plot_circle(r, center, ax=None, color="k", label=""):
    x0 = center[0]
    y0 = center[1]
    theta_model = np.linspace(0, 2 * np.pi, 100)
    x_star = r * np.sin(theta_model)
    y_star = r * np.cos(theta_model)
    if ax is None:
        ax = plt.gca()
    ax.plot(x0 + x_star, y0 + y_star, color=color, label=label)
