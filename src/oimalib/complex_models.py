"""
Created on Wed Nov  4 13:14:23 2015

@author: asoulain
"""

import contextlib
import sys

import numpy as np
from matplotlib import pyplot as plt
from rich import print as rprint
from scipy import special
from scipy.ndimage import gaussian_filter1d

from oimalib.binary import getBinaryPos, kepler_solve
from oimalib.fourier import shiftFourier
from oimalib.tools import computeBinaryRatio, mas2rad, planck_law, rad2mas

MAX_INCLINATION_DEG = 90
MAX_POSITION_ANGLE_DEG = 180
TOP_COMPONENTS = 3


def norm(x, y):
    return np.sqrt(x**2 + y**2)


def model_acc_mag(x, param):
    u = x[0]
    v = x[1]
    wl = x[2]
    if "diam" in param:
        Y = np.squeeze(abs(visUniformDisk(u, v, wl, param)))
    elif "incl" in param:
        Y = np.squeeze(abs(visEllipticalGaussianDisk2(u, v, wl, param)))
        if (
            (param["incl"] >= MAX_INCLINATION_DEG)
            or (param["incl"] < 0)
            or (param["pa"] >= MAX_POSITION_ANGLE_DEG)
            or (param["pa"] <= 0)
        ):
            Y = np.ones_like(Y)
    elif "fh" in param:
        fhalo = param["fh"]
        fmag = 1 - fhalo
        vis_mag = visGaussianDisk(u, v, wl, param)
        vis_halo = 0.0
        tot_vis = fmag * vis_mag + fhalo * vis_halo
        Y = np.squeeze(abs(tot_vis))
        if (param["fh"] > 1) or (param["fh"] < 0):
            Y = np.ones_like(Y)
    else:
        Y = np.squeeze(abs(visGaussianDisk(u, v, wl, param)))
    return Y


def _elong_gauss_disk(u, v, a=1.0, cosi=1.0, pa=0.0):
    """
    Return the complex visibility of an ellongated gaussian
    of size a cosi (a is the radius),
    position angle PA.
    PA is major-axis, East from North
    """

    # Elongated Gaussian
    rPA = pa - np.deg2rad(90)
    uM = u * np.cos(rPA) - v * np.sin(rPA)
    um = +u * np.sin(rPA) + v * np.cos(rPA)

    # a = rad2mas(a)
    aq2 = (a * uM) ** 2 + (a * cosi * um) ** 2

    # tmp_f = np.exp(-1 * (np.pi * q * a) ** 2 * ((np.cos(psi - theta) * cos_i) ** 2 +
    #                                             (np.sin(psi - theta) ** 2)) / np.log(2))

    return np.exp(-(np.pi**2) * aq2 / (np.log(2))).astype(complex)


def _elong_lorentz_disk(u, v, a, cosi, pa):
    rPA = pa - np.deg2rad(90)
    uM = u * np.cos(rPA) - v * np.sin(rPA)
    um = +u * np.sin(rPA) + v * np.cos(rPA)
    aq = ((a * uM) ** 2 + (a * cosi * um) ** 2) ** 0.5
    return np.exp(-(2 * np.pi * aq) / np.sqrt(3)).astype(complex)


def _elong_ring(u, v, a=1.0, cosi=1.0, pa=0.0, *, c1=0.0, s1=0.0):
    """
    Return the complex visibility of an elongated ring
    of size a cosi,
    position angle PA.
    PA is major-axis, East from North
    """

    # Squeeze and rotation
    rPA = pa - np.deg2rad(90)
    uM = u * np.cos(rPA) - v * np.sin(rPA)
    um = u * np.sin(rPA) + v * np.cos(rPA)

    # Polar coordinates (check angle)
    z = 2.0 * np.pi * a * norm(uM, cosi * um)
    psi = np.arctan2(um, uM)

    # Modulation in polar
    rho1 = norm(c1, s1)
    phi1 = np.arctan2(s1, -c1) + np.pi / 2.0

    rho1 = np.sqrt(c1**2 + s1**2)
    phi1 = np.arctan2(-c1, s1) + np.pi / 2

    mod = 0 if rho1 == 0 else -1j * rho1 * np.cos(psi - phi1) * special.jv(1, z)

    # Visibility
    v = special.jv(0, z) + mod
    return v.astype(complex)


def visPointSource(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a point source.

    Params:
    -------
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    C_centered = np.ones(np.size(Utable))
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, param["x0"], param["y0"])
    return C


def visBinary(Utable, Vtable, Lambda, param):
    sep = mas2rad(param["sep"])
    dm = param["dm"]
    theta = np.deg2rad(90 - param["pa"])

    f1 = 1
    f2 = f1 / (2.512**dm)
    ftot = f1 + f2

    rel_f1 = f1 / ftot
    rel_f2 = f2 / ftot

    p_s1 = {"x0": 0, "y0": 0}
    p_s2 = {"x0": sep * np.cos(theta), "y0": sep * np.sin(theta)}
    s1 = rel_f1 * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = rel_f2 * visPointSource(Utable, Vtable, Lambda, p_s2)
    C_centered = s1 + s2
    return C_centered


def visBinary_res(Utable, Vtable, Lambda, param):
    sep = mas2rad(param["sep"])
    dm = param["dm"]
    theta = np.deg2rad(90 - param["pa"])
    diam = param["diam"]
    if dm < 0:
        return np.array([np.nan] * len(Lambda))
    f1 = 1
    f2 = f1 / (10 ** (0.4 * dm))
    ftot = f1 + f2

    rel_f1 = f1 / ftot
    rel_f2 = f2 / ftot

    p_s1 = {"x0": 0, "y0": 0, "diam": diam}
    p_s2 = {"x0": sep * np.cos(theta), "y0": sep * np.sin(theta)}
    s1 = rel_f1 * visUniformDisk(Utable, Vtable, Lambda, p_s1)
    s2 = rel_f2 * visPointSource(Utable, Vtable, Lambda, p_s2)
    C_centered = s1 + s2
    return C_centered


def visUniformDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an uniform disk

    Params:
    -------
    diam: {float}
        Diameter of the disk [mas],\n
    x0, y0: {float}
        Shift along x and y position [mas].
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    x0 = mas2rad(param.get("x0", 0))
    y0 = mas2rad(param.get("y0", 0))

    diam = mas2rad(param["diam"])

    r = np.sqrt(u**2 + v**2)

    C_centered = 2 * special.j1(np.pi * r * diam) / (np.pi * r * diam)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visEllipticalUniformDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a elliptical uniform disk.

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [mas],\n
    incl: {float}
        Inclination (i.e: minorAxis = cos(incl) * majorAxis),\n
    pa: {float}
        Orientation of the disk [deg],\n
    x0, y0: {float}
        Shift along x and y position [mas].
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    # List of parameter
    elong = np.cos(np.deg2rad(param["incl"]))
    majorAxis = mas2rad(param["majorAxis"])
    minorAxis = elong * majorAxis

    angle = np.deg2rad(param["pa"])
    x0 = mas2rad(param.get("x0", 0))
    y0 = mas2rad(param.get("y0", 0))

    r = np.sqrt(
        ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
        + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
    )

    r[r == 0] = np.nan

    C_centered = 2 * special.j1(np.pi * r) / (np.pi * r)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visGaussianDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a gaussian disk

    Params:
    -------
    fwhm: {float}
        fwhm of the disk [mas],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    fwhm = mas2rad(param["fwhm"])
    x0 = mas2rad(param.get("x0", 0))
    y0 = mas2rad(param.get("y0", 0))

    q = (u**2 + v**2) ** 0.5
    r2 = ((np.pi * q * fwhm) ** 2) / (4 * np.log(2.0))
    C_centered = np.exp(-r2)

    C = C_centered if x0 == y0 == 0 else shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visEllipticalGaussianDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical gaussian disk

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [mas],\n
    incl: {float}
        Inclination of the disk [deg],\n
    pa: {float}
        Orientation of the disk [deg],\n
    x0, y0: {float}
        Shift along x and y position [mas].
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    # List of parameter
    elong = np.cos(np.deg2rad(param["incl"]))
    majorAxis = mas2rad(param["majorAxis"])
    minorAxis = elong * majorAxis
    angle = np.deg2rad(param["pa"])
    x0 = param.get("x0", 0)
    y0 = param.get("y0", 0)

    r2 = (
        (np.pi**2)
        * (
            ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
            + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
        )
        / (4.0 * np.log(2.0))
    )

    C_centered = np.exp(-r2)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visEllipticalGaussianDisk2(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical gaussian disk

    Params:
    -------
    minorAxis: {float}
        Major axis of the disk [mas],\n
    incl: {float}
        Inclination of the disk [deg],\n
    pa: {float}
        Orientation of the disk [deg],\n
    x0, y0: {float}
        Shift along x and y position [mas].
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    # List of parameter
    elong = 1.0 / np.cos(np.deg2rad(param["incl"]))
    minorAxis = mas2rad(param["minorAxis"])
    majorAxis = elong * minorAxis
    angle = np.deg2rad(param["pa"])
    x0 = param.get("x0", 0)
    y0 = param.get("y0", 0)

    r2 = (
        (np.pi**2)
        * (
            ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
            + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
        )
        / (4.0 * np.log(2.0))
    )

    C_centered = np.exp(-r2)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visCont(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility double gaussian model (same as TW Hya for the
    IR continuum).

    Params:
    -------
    `fwhm` {float}:
        Full major axis of the disk [mas],\n
    `incl` {float}:
        Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)) [deg],\n
    `pa` {float}:
        Orientation of the disk (from north to East) [deg],\n
    `fratio` {float}:
        Stellar to total flux ratio (i.e: 1/fratio = f* [%]),\n
    `rstar` {float}:
        Radius of the star [mas],\n
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    pa = np.deg2rad(param["pa"])

    # Define the reduction ratio apply on the fwhm
    # because of the inclination
    incl = np.deg2rad(param["incl"])
    elong = np.cos(incl)
    majorAxis = mas2rad(param["fwhm"])

    # Define stellar flux ratio
    fratio = param["fratio"]
    Fstar = 1.0 / fratio
    Fdisc = 1 - Fstar

    # Stellar radius (resolved disk)
    rstar = 2 * param["rstar"]

    Vdisc = _elong_gauss_disk(u, v, a=majorAxis, cosi=elong, pa=pa)

    p_star = {"fwhm": rstar, "x0": 0, "y0": 0}
    Vstar = visGaussianDisk(Utable, Vtable, Lambda, p_star)

    Vcont = (Fdisc * Vdisc) + (Fstar * Vstar)
    return Vcont


def visYSO(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a YSO model (star + gaussian disk + resolved
    halo). The halo contribution is computed with 1 - fc - fs.

    Params:
    -------
    `hfr` {float}:
        Half major axis of the disk [mas],\n
    `flor` {float}:
        Weighting for radial profile (0 gaussian kernel,
        1 Lorentizian kernel),\n
    `incl` {float}:
        Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)),\n
    `pa` {float}:
        Orientation of the disk (from north to East) [deg],\n
    `fs` {float}:
        Flux contribution of the star [%],\n
    `fc` {float}:
        Flux contribution of the disk [%],\n
    """
    fh = param["fh"]
    fc = param["fc"]
    fs = 1 - fh - fc

    param_disk = {
        "fwhm": 2 * param["hfr"],  # For ellipsoid, fwhm is the radius
        "flor": param.get("flor", 0),
        "pa": param["pa"],
        "incl": param["incl"],
    }

    C = visEllipsoid(Utable, Vtable, Lambda, param_disk)

    p_s1 = {"x0": 0, "y0": 0}
    s1 = fs * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = fc * C
    ftot = fs + fh + fc
    return (s1 + s2) / ftot


def visLazareff(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a Lazareff model (star + thick ring + resolved
    halo). The halo contribution is computed with 1 - fc - fs.

    Params:
    -------
    `la` {float}:
        Half major axis of the disk (log),\n
    `lr` {float}:
        Kernel half light (log),\n
    `flor` {float}:
        Weighting for radial profile (0 gaussian kernel,
        1 Lorentizian kernel),\n
    `incl` {float}:
        Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)) [deg],\n
    `pa` {float}:
        Orientation of the disk (from north to East) [deg],\n
    `fs` {float}:
        Flux contribution of the star [%],\n
    `fc` {float}:
        Flux contribution of the disk [%],\n
    `ks` {float}:
        Spectral index compared to reference wave at 2.2 µm,\n
    `c1`, `s1` {float}:
        Cosine and sine amplitude for the mode 1 (azimutal changes),\n

    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]
    # List of parameter

    elong = np.cos(np.deg2rad(param["incl"]))
    la = param["la"]
    lk = param["lk"]

    kr = 10.0 ** (lk)
    ar = 10**la / (np.sqrt(1 + kr**2))
    ak = ar * kr
    # print(ar, ak)
    # self.a_r = 10 ** self.l_a / np.sqrt(1 + 10 ** (2 * self.l_kr))
    # self.a_k = 10 ** self.l_kr * self.a_r

    ar_rad = mas2rad(ar)
    semi_majorAxis = ar_rad

    pa = np.deg2rad(param["pa"])

    fs = param["fs"]
    fc = param["fc"]
    fh = 1 - fc - fs

    if param["type"] == "smooth":
        param_ker = {
            "pa": param["pa"],
            "incl": param["incl"],
            "fwhm": ak,
            "flor": param["flor"],
        }
        Vkernel = visEllipsoid(Utable, Vtable, Lambda, param_ker)
    elif param["type"] == "uniform":
        param_ker = {
            "diam": ak,
            "x0": 0,
            "y0": 0,
        }
        Vkernel = visUniformDisk(Utable, Vtable, Lambda, param_ker)
    else:
        Vkernel = 1

    try:
        cj = param["cj"]
        sj = param["sj"]
    except Exception:
        cj = 0
        sj = 0

    Vring = _elong_ring(u, v, a=semi_majorAxis, cosi=elong, pa=pa, c1=cj, s1=sj) * Vkernel

    ks = param["ks"]
    kc = param["kc"]
    wl0 = 2.2e-6

    fs_lambda = fs * (wl0 / Lambda) ** ks
    fc_lambda = fc * (wl0 / Lambda) ** kc
    fh_lambda = fh * (wl0 / Lambda) ** ks
    p_s1 = {"x0": 0, "y0": 0}
    s1 = fs_lambda[:, None] * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = fc_lambda[:, None] * Vring
    ftot = fs_lambda + fh_lambda + fc_lambda
    return s1 + s2 / ftot[:, None]


def visLazareff_halo(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a Lazareff model (star + thick ring + resolved
    halo). The star contribution is computed with 1 - fc - fh.

    Params:
    -------
    `la` {float}:
        Logarithmic half-flux extent of the disk (log10),\n
    `lr` {float}:
        Logarithmic ratio between ar (angular radius) and ak (kernel radius),\n
    `flor` {float}:
        Weighting for radial profile (0 gaussian kernel,
        1 Lorentizian kernel),\n
    `incl` {float}:
        Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)),\n
    `pa` {float}:
        Orientation of the disk (from north to East) [rad],\n
    `fh` {float}:
        Flux contribution of the halo [%],\n
    `fc` {float}:
        Flux contribution of the disk [%],\n
    `ks` {float}:
        Spectral index compared to reference wave at 2.2 µm,\n
    `c1`, `s1` {float}:
        Cosine and sine amplitude for the mode 1 (azimutal changes),\n

    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]
    # List of parameter

    elong = np.cos(np.deg2rad(param["incl"]))
    la = param["la"]
    lk = param["lk"]

    kr = 10.0 ** (lk)
    ar = 10**la / (np.sqrt(1 + kr**2))
    ak = ar * kr

    ar_rad = mas2rad(ar)
    semi_majorAxis = ar_rad

    pa = np.deg2rad(param["pa"])

    fh = param["fh"]
    fc = param["fc"]
    fs = 1 - fc - fh

    prf_ring = param.get("type", "smooth")
    if prf_ring == "smooth":
        param_ker = {
            "pa": param["pa"],
            "incl": param["incl"],
            "fwhm": 2 * ak,
            "flor": param.get("flor", 0),
        }
        Vkernel = visEllipsoid(Utable, Vtable, Lambda, param_ker)
    elif prf_ring == "uniform":
        param_ker = {
            "diam": ak,
            "x0": 0,
            "y0": 0,
        }
        Vkernel = visUniformDisk(Utable, Vtable, Lambda, param_ker)
    else:
        Vkernel = 1

    param_ker2 = {
        "diam": param.get("w_out", 0),
        "x0": 0,
        "y0": 0,
    }
    if param_ker2["diam"] != 0:
        Vkernel2 = visUniformDisk(Utable, Vtable, Lambda, param_ker2)

    cj = param.get("cj", 0)
    sj = param.get("sj", 0)
    cj2 = param.get("cj2", 0)
    sj2 = param.get("sj2", 0)

    r_out = mas2rad(ar + param_ker2["diam"] / 2)
    f_out = param.get("fout", 0)

    Vrim = _elong_ring(u, v, a=semi_majorAxis, cosi=elong, pa=pa, c1=cj, s1=sj) * Vkernel

    if f_out == 0:
        Vdisk_out = 0
    elif param_ker2["diam"] != 0:
        Vdisk_out = _elong_ring(u, v, a=r_out, cosi=elong, pa=pa, c1=cj2, s1=sj2) * Vkernel2
    else:
        Vdisk_out = 0

    ks = param.get("ks", 0)
    kc = param.get("kc", 0)
    wl0 = 2.2e-6

    fs_lambda = fs * (wl0 / Lambda) ** ks
    fc_lambda = fc * (wl0 / Lambda) ** kc
    fh_lambda = fh * (wl0 / Lambda) ** ks
    p_s1 = {"x0": param.get("x_star", 0), "y0": param.get("y_star", 0)}
    p_s2 = {
        "x0": mas2rad(param.get("x_clump", 0)),
        "y0": mas2rad(param.get("y_clump", 0)),
    }
    f_clump = param.get("f_clump", 0)

    f_disk_out = f_out * fc_lambda[:, None]
    f_rim = (1 - f_out) * fc_lambda[:, None]
    s1 = fs_lambda[:, None] * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = (f_rim - f_clump) * Vrim
    s3 = (f_disk_out - f_clump) * Vdisk_out
    s4 = f_clump * visPointSource(Utable, Vtable, Lambda, p_s2)
    # print(fs_lambda[0], fh_lambda[0], f_rim[0] - f_clump, f_clump)
    ftot = fs_lambda + fh_lambda + fc_lambda
    return (s1 + s2 + s3 + s4) / ftot[:, None]


def visLazareff_clump(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a Lazareff model (star + thick ring + resolved
    halo + clump). The star contribution is computed with 1 - fc - fh.

    Params:
    -------
    `la` {float}:
        Logarithmic half-flux extent of the disk (log10),\n
    `lr` {float}:
        Logarithmic ratio between ar (angular radius) and ak (kernel radius),\n
    `flor` {float}:
        Weighting for radial profile (0 gaussian kernel,
        1 Lorentizian kernel),\n
    `incl` {float}:
        Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)),\n
    `pa` {float}:
        Orientation of the disk (from north to East) [rad],\n
    `fh` {float}:
        Flux contribution of the halo [%],\n
    `fc` {float}:
        Flux contribution of the disk [%],\n
    `ks` {float}:
        Spectral index compared to reference wave at 2.2 µm,\n
    `c1`, `s1` {float}:
        Cosine and sine amplitude for the mode 1 (azimutal changes),\n

    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]
    # List of parameter

    elong = np.cos(np.deg2rad(param["incl"]))
    la = param["la"]
    lk = param["lk"]

    kr = 10.0 ** (lk)
    ar = 10**la / (np.sqrt(1 + kr**2))
    ak = ar * kr

    ar_rad = mas2rad(ar)
    semi_majorAxis = ar_rad

    pa = np.deg2rad(param["pa"])

    fh = param["fh"]
    fc = param["fc"]
    fs = 1 - fc - fh

    prf_ring = param.get("type", "smooth")
    if prf_ring == "smooth":
        param_ker = {
            "pa": param["pa"],
            "incl": param["incl"],
            "fwhm": 2 * ak,
            "flor": param.get("flor", 0),
        }
        Vkernel = visEllipsoid(Utable, Vtable, Lambda, param_ker)
    elif prf_ring == "uniform":
        param_ker = {
            "diam": ak,
            "x0": 0,
            "y0": 0,
        }
        Vkernel = visUniformDisk(Utable, Vtable, Lambda, param_ker)
    else:
        Vkernel = 1

    Vrim = _elong_ring(u, v, a=semi_majorAxis, cosi=elong, pa=pa, c1=0, s1=0) * Vkernel

    ks = param.get("ks", 0)
    kc = param.get("kc", 0)
    wl0 = 2.2e-6

    fs_lambda = fs * (wl0 / Lambda) ** ks
    fc_lambda = fc * (wl0 / Lambda) ** kc
    fh_lambda = fh * (wl0 / Lambda) ** ks
    p_s1 = {"x0": 0, "y0": 0}

    ratio_clump = param.get("ratio_clump", 0)
    pa_clump = param.get("pa_cl", 0)
    r_clump = mas2rad(param.get("d_cl", 0))

    s1 = fs_lambda[:, None] * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = (1 - ratio_clump) * fc_lambda[:, None] * Vrim

    x0_clump = r_clump * np.sin(np.deg2rad(pa_clump))
    y0_clump = r_clump * np.cos(np.deg2rad(pa_clump))

    p_clump = {"x0": x0_clump, "y0": y0_clump}

    s3 = ratio_clump * fc_lambda[:, None] * visPointSource(Utable, Vtable, Lambda, p_clump)

    ftot = fs_lambda + fh_lambda + fc_lambda
    return (s1 + s2 + s3) / ftot[:, None]


def visLazareff_line(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a Lazareff model (star + thick ring + resolved
    halo). The halo contribution is computed with 1 - fc - fs.

    Params:
    -------
    `la` {float}:
        Half major axis of the disk (log),\n
    `lr` {float}:
        Kernel half light (log),\n
    `flor` {float}:
        Weighting for radial profile (0 gaussian kernel,
        1 Lorentizian kernel),\n
    `incl` {float}:
        Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)),\n
    `pa` {float}:
        Orientation of the disk (from north to East) [rad],\n
    `fs` {float}:
        Flux contribution of the star [%],\n
    `fc` {float}:
        Flux contribution of the disk [%],\n
    `ks` {float}:
        Spectral index compared to reference wave at 2.2 µm,\n
    `c1`, `s1` {float}:
        Cosine and sine amplitude for the mode 1 (azimutal changes),\n

    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]
    # List of parameter

    elong = np.cos(np.deg2rad(param["incl"]))
    la = param["la"]
    lk = param["lk"]

    kr = 10.0 ** (lk)
    ar = 10**la / (np.sqrt(1 + kr**2))
    ak = ar * (10**lk)

    ar_rad = mas2rad(ar)
    majorAxis = ar_rad
    # minorAxis = majorAxis * elong

    pa = np.deg2rad(param["pa"])

    fs = param["fs"]
    fc = param["fc"]
    fh = 1 - fs - fc

    param_ker = {
        "pa": param["pa"],
        "incl": param["incl"],
        "fwhm": ak,
        "flor": param.get("flor", 0),
    }

    Vkernel = visEllipsoid(Utable, Vtable, Lambda, param_ker)

    try:
        cj = param["cj"]
        sj = param["sj"]
    except Exception:
        cj = 0
        sj = 0

    Vring = (_elong_ring(u, v, a=majorAxis, cosi=elong, pa=pa, c1=cj, s1=sj)) * Vkernel

    # Vring = (_elong_ring(u, v, pa, majorAxis, minorAxis) + azimuth_mod) * Vkernel

    ks = param["ks"]
    kc = param["kc"]
    wl0 = 2.2e-6

    lF = param["lF"]

    param_line = {
        "pa": param["lpa"],
        "incl": param["lincl"],
        "fwhm": param["lT"],
        "flor": 0,
    }

    lbdBrg = param["wl_brg"] * 1e-6
    sigBrg = param["sig_brg"] * 1e-6

    # Line emission
    Fl = lF * np.exp(-0.5 * (Lambda - lbdBrg) ** 2 / sigBrg**2)
    Vl = visEllipsoid(Utable, Vtable, Lambda, param_line)

    shift_x = mas2rad(1e-3 * param["shift_x"])
    shift_y = mas2rad(1e-3 * param["shift_y"])

    # Shift of line emission
    Vl = shiftFourier(Utable, Vtable, Lambda, Vl, shift_x, shift_y)

    fs_lambda = fs * (wl0 / Lambda) ** ks
    fc_lambda = fc * (wl0 / Lambda) ** kc
    fh_lambda = fh * (wl0 / Lambda) ** ks
    p_s1 = {"x0": 0, "y0": 0}
    s1 = fs_lambda * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = fc_lambda * Vring
    ftot = fs_lambda + fh_lambda + fc_lambda

    return (s1 + s2 + Fl * Vl) / (ftot + Fl)


def visThickEllipticalRing(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical thick ring.

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [mas],\n
    incl: {float}
        Inclination of the disk [deg],\n
    angle: {float}
        Position angle of the disk [deg],\n
    w: {float}
        Thickness of the ring [mas],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    elong = np.cos(np.deg2rad(param["incl"]))
    majorAxis = mas2rad(param["majorAxis"])
    angle = np.deg2rad(param["pa"])
    kr = 10 ** param["kr"]

    majorAxis = majorAxis / (np.sqrt(1 + kr**2))

    minorAxis = elong * majorAxis
    thickness = majorAxis * kr

    prf = param.get("prf", "gauss")

    # print(param["w"], param["majorAxis"])
    x0 = param.get("x0", 0)
    y0 = param.get("y0", 0)

    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    r = np.sqrt(
        ((u * np.sin(angle) + v * np.cos(angle)) * (majorAxis / 2.0)) ** 2
        + ((u * np.cos(angle) - v * np.sin(angle)) * (minorAxis / 2.0)) ** 2
    )

    C_centered = special.j0(2 * np.pi * r)
    C_shifted = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    if prf == "gauss":
        kernel = visGaussianDisk(Utable, Vtable, Lambda, {"fwhm": rad2mas(thickness)})
    elif prf == "uniform":
        kernel = visUniformDisk(Utable, Vtable, Lambda, {"diam": rad2mas(thickness)})
    C = C_shifted * kernel
    return C


def visEllipticalRing(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical ring (infinitly thin).

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    # List of parameter
    majorAxis = param["majorAxis"]
    minorAxis = param["minorAxis"]
    angle = param["angle"]
    x0 = param["x0"]
    y0 = param["y0"]

    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    r = np.sqrt(
        ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
        + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
    )

    C_centered = special.j0(np.pi * r)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visEllipsoid(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an ellipsoid (as Lazareff+17).

    Params:
    -------
    `fwhm` {float}:
        FWHM of the disk [mas],\n
    `incl` {float}:
        Inclination of the disk [deg],\n
    `pa` {float}:
        Orientation of the disk [deg],\n
    `flor` {float}:
        Hybridation between purely gaussian (flor=0)
        and Lorentzian radial profile (flor=1).
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    pa = np.deg2rad(param["pa"])

    # Define the reduction ratio apply on the fwhm
    # because of the inclination
    incl = np.deg2rad(param["incl"])
    elong = np.cos(incl)
    semi_majorAxis = mas2rad(param["fwhm"]) / 2.0
    # minorAxis = elong * majorAxis

    # majorAxis is the half-radius

    flor = param["flor"]

    Vlor = _elong_lorentz_disk(u, v, a=semi_majorAxis, cosi=elong, pa=pa)
    Vgauss = _elong_gauss_disk(u, v, a=semi_majorAxis, cosi=elong, pa=pa)

    Vc = (1 - flor) * Vgauss + flor * Vlor
    return Vc


def visLorentzDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an Lorentzian disk.

    Params:
    -------
    fwhm: {float}
        Size of the disk [mas],\n
    x0, y0: {float}
        Shift along x and y position [mas].
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    fwhm = mas2rad(param["fwhm"])
    x0 = mas2rad(param["x0"])
    y0 = mas2rad(param["y0"])

    q = (u**2 + v**2) ** 0.5
    r = 2 * np.pi * fwhm * q / np.sqrt(3)
    C_centered = np.exp(-r)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visDebrisDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical thick ring.

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    thickness: {float}
        Thickness of the ring [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    majorAxis = mas2rad(param["majorAxis"])
    elong = np.cos(np.deg2rad(param["incl"]))
    posang = np.deg2rad(param["pa"])
    thickness = param["w"]
    cr_disk = 1e-10  # param["cr"]
    cr_planet = param.get("cr_p", 1e50)
    x0 = param.get("x0", 0)
    y0 = param.get("y0", 0)

    minorAxis = majorAxis * elong

    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    r = np.sqrt(
        ((u * np.sin(posang) + v * np.cos(posang)) * majorAxis) ** 2
        + ((u * np.cos(posang) - v * np.sin(posang)) * minorAxis) ** 2
    )

    C_centered = special.j0(np.pi * r)
    C_shifted = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    C = C_shifted * visGaussianDisk(
        Utable, Vtable, Lambda, {"fwhm": thickness, "x0": 0.0, "y0": 0.0}
    )

    fstar = 1
    fdisk = fstar / cr_disk
    fplanet = fstar / cr_planet
    total_flux = fstar + fdisk + fplanet

    rel_star = fstar / total_flux
    rel_disk = fdisk / total_flux
    rel_planet = fplanet / total_flux

    sep = mas2rad(param.get("sep_planet", 0))
    theta = np.deg2rad(param.get("pa_planet", 0))

    p_s1 = {"x0": x0, "y0": y0}
    p_s2 = {"x0": sep * np.cos(theta), "y0": sep * np.sin(theta)}

    print(
        f" star = {round(rel_star, 4) * 1e2} %, "
        f"disk = {round(rel_disk, 4) * 1e2} %, "
        f"planet = {round(rel_planet, 4) * 1e2} %"
    )
    s1 = rel_star * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = rel_disk * C
    s3 = rel_planet * visPointSource(Utable, Vtable, Lambda, p_s2)
    return s1 + s2 + s3


def visMultipleRing(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical thick ring.

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    thickness: {float}
        Thickness of the ring [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    majorAxis1 = mas2rad(param["majorAxis1"])

    elong = np.cos(np.deg2rad(param["incl"]))
    posang = np.deg2rad(param["pa"])

    thickness1 = param["w1"]
    thickness2 = param.get("w2", thickness1)
    thickness3 = param.get("w3", thickness1)

    x0 = param.get("x0", 0)
    y0 = param.get("y0", 0)

    minorAxis1 = majorAxis1 * elong

    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    r1 = np.sqrt(
        ((u * np.sin(posang) + v * np.cos(posang)) * majorAxis1) ** 2
        + ((u * np.cos(posang) - v * np.sin(posang)) * minorAxis1) ** 2
    )

    C_centered1 = special.j0(np.pi * r1)
    C_shifted1 = shiftFourier(Utable, Vtable, Lambda, C_centered1, x0, y0)
    C1 = C_shifted1 * visGaussianDisk(
        Utable, Vtable, Lambda, {"fwhm": thickness1, "x0": 0.0, "y0": 0.0}
    )

    C2 = visGaussianDisk(Utable, Vtable, Lambda, {"fwhm": thickness2, "x0": 0.0, "y0": 0.0})

    C3 = visGaussianDisk(Utable, Vtable, Lambda, {"fwhm": thickness3, "x0": 0.0, "y0": 0.0})

    f1 = 1
    f2 = param.get("f2", 0)
    f3 = param.get("f3", 0)

    fstar = param.get("fstar", 0)
    fplanet = param.get("fplanet", 0)

    radius_planet = mas2rad(150)
    xp = radius_planet * np.cos(posang)
    yp = radius_planet * np.sin(posang)
    p_s1 = {"x0": x0, "y0": y0}
    p_s2 = {"x0": x0 + xp, "y0": y0 + yp}

    c = fstar * visPointSource(Utable, Vtable, Lambda, p_s1)
    p = fplanet * visPointSource(Utable, Vtable, Lambda, p_s2)

    f1 = 1 - f2 - f3 - fstar - fplanet
    s1 = f1 * C1
    s2 = f2 * C2
    s3 = f3 * C3
    return c + p + s1 + s2 + s3


def visClumpDebrisDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical thick ring.

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    thickness: {float}
        Thickness of the ring [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    majorAxis = mas2rad(param["majorAxis"])
    elong = np.cos(np.deg2rad(param["incl"]))
    posang = np.deg2rad(param["pa"])
    thickness = param["w"]
    fs = param["fs"] / 100.0
    x0 = param["x0"]
    y0 = param["y0"]

    minorAxis = majorAxis * elong

    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    r = np.sqrt(
        ((u * np.sin(posang) + v * np.cos(posang)) * majorAxis) ** 2
        + ((u * np.cos(posang) - v * np.sin(posang)) * minorAxis) ** 2
    )

    d_clump = param["d_clump"]
    pa_clump = -np.deg2rad(param["pa_clump"])
    fc = param["fc"] / 100.0

    x1 = 0
    y1 = majorAxis * elong
    x_clump = (x1 * np.cos(pa_clump) - y1 * np.sin(pa_clump)) / 2.0
    y_clump = (x1 * np.sin(pa_clump) + y1 * np.cos(pa_clump)) / 2.0

    p_clump = {"fwhm": d_clump, "x0": rad2mas(x_clump), "y0": rad2mas(y_clump)}

    C_clump = visGaussianDisk(Utable, Vtable, Lambda, p_clump)

    C_centered = special.j0(np.pi * r)
    C_shifted = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    C = C_shifted * visGaussianDisk(
        Utable, Vtable, Lambda, {"fwhm": thickness, "x0": 0.0, "y0": 0.0}
    )

    fd = 1 - fs - fc

    p_s1 = {"x0": x0, "y0": y0}
    c_star = fs * visPointSource(Utable, Vtable, Lambda, p_s1)
    c_ring = fd * C
    c_clump = fc * C_clump
    return c_star + c_ring + c_clump


def visOrionDisk(Utable, Vtable, Lambda, param):
    param_disk = {
        "pa": param["pa"] + 90,
        "incl": param["i1"],
        "majorAxis": 2 * param["r1"],
    }

    param_cocoon = {
        "pa": param["pa"] + 90,
        "incl": param["i2"],
        "fwhm": 2 * param["r2"],
        "flor": param.get("flor", 0),
    }

    param_blob = {
        "pa": param["pa"] + 90,
        "incl": param["i3"],
        "fwhm": 2 * param["r3"],
        "flor": param.get("flor", 0),
    }

    pa0 = np.deg2rad(param["pa_blob"] - (param["pa"] + 90) + 180)

    x0 = mas2rad(param["d"]) * np.cos(pa0)
    y0 = mas2rad(param["d"]) * np.sin(pa0)
    V_disk = visEllipticalUniformDisk(Utable, Vtable, Lambda, param_disk)
    V_cocoon = visEllipsoid(Utable, Vtable, Lambda, param_cocoon)
    V_blob_c = visEllipsoid(Utable, Vtable, Lambda, param_blob)

    V_blob = shiftFourier(Utable, Vtable, Lambda, V_blob_c, x0, y0)

    f_disk = param["fd"]
    f_blob = param["fb"]
    f_cocoon = 1 - f_disk - f_blob  # param["fc"]

    V_tot = f_disk * V_disk + f_cocoon * V_cocoon + f_blob * V_blob
    return V_tot


def _compute_param_elts(
    ruse,
    tetuse,
    alpha,
    thick,
    incl,
    angleSky,
    angle_0,
    step,
    rounds,
    rnuc=0,
    proj=True,
    limit_speed=False,
    display=False,
    verbose=False,
):
    angle0 = tetuse + angle_0
    x0 = ruse * np.cos(angle0)
    y0 = ruse * np.sin(angle0)
    fwhmy0 = thick
    x1 = x0 * np.cos(incl)
    y1 = y0
    angle1 = np.arctan2(x1, y1)
    fwhmx1 = fwhmy0 * np.cos(angle0) * np.sin(incl)
    fwhmy1 = fwhmy0
    angle2 = np.transpose(angle1 + angleSky)
    x2 = np.transpose(x1 * np.cos(angleSky) + y1 * np.sin(angleSky))
    y2 = np.transpose(-x1 * np.sin(angleSky) + y1 * np.cos(angleSky))
    fwhmx2 = np.transpose(fwhmx1)
    fwhmy2 = np.transpose(fwhmy1)

    if proj:
        proj_fact = np.cos(alpha / 2.0)
        if verbose:
            print(f"Projection factor θ ({np.rad2deg(alpha):2.1f}) = {proj_fact:2.2f}")
    else:
        proj_fact = 1

    decx = rad2mas(fwhmy2 / 2.0) * np.cos(angle2) * proj_fact
    decy = rad2mas(fwhmy2 / 2.0) * np.sin(angle2) * proj_fact
    px0, py0 = -rad2mas(x2) * proj_fact, -rad2mas(y2) * proj_fact
    px1, py1 = px0 - decx, py0 + decy
    px2, py2 = px0 + decx, py0 - decy
    dwall1 = (px1**2 + py1**2) ** 0.5
    dwall2 = (px1**2 + py1**2) ** 0.5

    lim = rounds * rad2mas(step)
    limit_speed_cond = (dwall1 <= lim) & (dwall2 <= lim) if limit_speed else [True] * len(dwall1)

    if display:
        tmp = np.linspace(0, 2 * np.pi, 300)
        xrnuc, yrnuc = rnuc * np.cos(tmp), rnuc * np.sin(tmp)
        plt.figure(figsize=(5, 5))
        plt.plot(
            px0[limit_speed_cond],
            py0[limit_speed_cond],
            ".",
            color="#0d4c36",
            label="Archimedean spiral",
        )
        for px1_i, px2_i, py1_i, py2_i in zip(
            px1[limit_speed_cond],
            px2[limit_speed_cond],
            py1[limit_speed_cond],
            py2[limit_speed_cond],
            strict=False,
        ):
            plt.plot(
                [px1_i, px2_i],
                [py1_i, py2_i],
                "-",
                color="#00b08b",
                alpha=0.5,
                lw=1,
            )
        plt.plot(
            px1[limit_speed_cond],
            py1[limit_speed_cond],
            "--",
            color="#ce0058",
            label="Spiral wall",
            lw=0.5,
        )
        plt.plot(px2[limit_speed_cond], py2[limit_speed_cond], "--", color="#ce0058", lw=0.5)
        for j in range(int(rounds)):
            radius = (j + 1) * rad2mas(step)
            prop_limx, prop_limy = radius * np.cos(tmp), radius * np.sin(tmp)
            plt.plot(prop_limx, prop_limy, "k-", lw=1)  # , label='S%i')
        plt.plot(xrnuc, yrnuc, "r:", lw=1)
        plt.plot(0, 0, "rx", label="WR star")
        plt.legend(fontsize=7, loc=1)
        plt.axis(np.array([-lim, lim, lim, -lim]) * 2.0)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("RA [mas]")
        plt.ylabel("DEC [mas]")
        plt.tight_layout()
        plt.show(block=False)
    return (
        x0 * proj_fact,
        y0 * proj_fact,
        x2 * proj_fact,
        y2 * proj_fact,
        fwhmx2 * proj_fact,
        fwhmy2 * proj_fact,
        angle2,
    )


def _compute_param_elts_spec(mjd, param, verbose=True, display=True):
    """Compute only once the elements parameters fo the spiral."""
    # Pinwheel parameters
    rounds = float(param["rounds"])

    # Convert appropriate units
    alpha = np.deg2rad(param["opening_angle"])
    r_nuc = mas2rad(param["r_nuc"])
    step = mas2rad(param["step"])

    # angle_0 = omega (orientation of the binary @ PA (0=north) counted counter-clockwise
    angle_0 = np.deg2rad(float(param["angle_0"]))
    incl = np.deg2rad(float(param["incl"]))
    angleSky = np.deg2rad(float(param["angleSky"]))

    totalSize = param["rounds"] * step

    # rarely changed
    power = 1.0  # float(param["power"])
    minThick = 0.0  # float(param["minThick"])

    # Set opening angle
    maxThick = 2 * np.tan(alpha / 2.0) * (rounds * step)

    # Ring type of the pinwheel
    types = param["compo"]  # "r2"

    # fillFactor = param["fillFactor"]
    thick = np.mean(minThick + maxThick) / 2.0

    N = int(param["nelts"] * rounds)

    # Full angular coordinates unaffected by the ecc.
    theta = np.linspace(0, rounds * 2 * np.pi, N)

    offst = mjd - param["mjd0"]

    # Use kepler solver solution to compute the eccentric angular coordinates
    time = np.linspace(0, rounds * param["P"], N) - offst
    theta_ecc, E = kepler_solve(time, param["P"], param["e"])

    a_bin = param["a"]  # [AU]
    sep_bin = a_bin * (1 - param["e"] * np.cos(E))

    fact = totalSize * rounds / (((rounds * 2.0 * np.pi) / (2.0 * np.pi)) ** power)
    r = (((theta) / (2.0 * np.pi)) ** power) * (fact / rounds)

    # Dust production if the binary separation is close enought (sep_bin < 's_prod')
    try:
        cond_prod = sep_bin <= param["s_prod"]
    except KeyError:
        if verbose:
            rprint(
                "[red]Warning: s_prod is not given, dust production is set to constant.",
                file=sys.stderr,
            )
        cond_prod = np.array([True] * len(sep_bin))

    dst = r * theta

    proj_fact = np.cos(alpha / 2.0)
    step1 = np.array([True] * N)
    # Add the dust sublimation radius
    if r_nuc != 0:
        step1[(r <= r_nuc / proj_fact)] = False

    step1[~cond_prod] = False
    step2 = np.array(list(step1))
    step3 = np.transpose(step2)

    pr2 = np.array(np.where((abs(dst) == max(abs(dst))) | (abs(dst) == min(abs(dst))) | step3)[0])
    pr2 = pr2[1:]

    N2 = len(pr2)
    if verbose:
        print(f"Number of ring in the pinwheel N = {N2:2.1f}")

    # Use only selected rings position (sublimation and production limits applied)
    ruse = r[pr2]
    tetuse = theta_ecc[pr2]

    typei = [types] * N2

    thick = minThick + ruse / (max(r) + (max(r) == 0)) * maxThick

    proj = True  # param["proj"]
    tab_orient = _compute_param_elts(
        ruse,
        tetuse,
        alpha,
        thick,
        incl,
        angleSky,
        angle_0,
        step,
        rounds,
        param["r_nuc"],
        proj=proj,
        limit_speed=False,
        display=display,
    )
    tab_faceon = _compute_param_elts(
        ruse,
        tetuse,
        alpha,
        thick,
        0,
        0,
        angle_0,
        step=step,
        rounds=rounds,
        rnuc=param["r_nuc"],
        proj=proj,
        limit_speed=False,
        display=False,
        verbose=verbose,
    )
    return tab_orient, tab_faceon, typei, N2, r_nuc, step, alpha


def sed_pwhl(wl, mjd, param, verbose=True, display=True):
    if "a" not in param:
        tab = getBinaryPos(mjd, param, mjd0=param["mjd0"], revol=1, v=2, au=True, display=False)
        param["a"] = tab["a"]

    _tab_orient, tab_faceon, _typei, N2, r_nuc, step, alpha = _compute_param_elts_spec(
        mjd, param, verbose=verbose, display=display
    )

    dmas = rad2mas(np.sqrt(tab_faceon[2] ** 2 + tab_faceon[3] ** 2))
    Tin = param["T_sub"]
    q = param["q"]
    Tr = Tin * (dmas / rad2mas(r_nuc)) ** (-q)

    gap_dist = param.get("gap_dist", 1)

    beta = 3e-12
    spec = []
    for xx in wl:
        gap_factor2 = param.get("gap_factor2", 1)
        T_smoother = param.get("T_smoother", 1)
        l_Tr = Tr.copy()
        l_Tr[dmas >= gap_dist * rad2mas(step)] /= param["gap_factor"]
        l_Tr = gaussian_filter1d(l_Tr, T_smoother)
        l_Tr[dmas >= gap_dist * 2 * rad2mas(step)] /= gap_factor2
        l_Tr = gaussian_filter1d(l_Tr, T_smoother / 2.0)
        i_flux = planck_law(l_Tr, xx)
        i_flux_jy = (i_flux * xx**2) / beta
        spectrumi = param["f_scale_pwhl"] * i_flux_jy
        spectrumi[0] *= 15
        spec.append(spectrumi)

    wl_sed = np.logspace(-7, -3.5, 1000)
    spec_all, spec_all1, spec_all2 = [], [], []
    for temperature, dist in zip(l_Tr, dmas, strict=False):
        i_flux = planck_law(temperature, wl_sed)
        i_flux_jy = (i_flux * wl_sed**2) / beta
        spectrum_r = param["f_scale_pwhl"] * i_flux_jy
        spec_all.append(spectrum_r)
        if dist >= np.cos(alpha / 2.0) * rad2mas(step):
            spec_all2.append(spectrum_r)
        else:
            spec_all1.append(spectrum_r)

    spec_all, spec_all1, spec_all2 = (
        np.array(spec_all),
        np.array(spec_all1),
        np.array(spec_all2),
    )

    total_sed = np.sum(spec_all, axis=0)
    wl_peak = wl_sed[np.argmax(total_sed)] * 1e6
    T_wien = 3000.0 / wl_peak

    spec = np.array(spec)
    flux0 = np.sum(spec[0, :])
    if display:
        plt.figure()
        plt.loglog(wl_sed * 1e6, spec_all1.T, color="grey")
        with contextlib.suppress(ValueError):
            plt.loglog(wl_sed * 1e6, spec_all2.T, color="lightgrey")
        plt.loglog(
            wl_sed * 1e6,
            total_sed,
            color="#008080",
            lw=3,
            alpha=0.8,
            label=rf"Total SED (T$_{{wien}}$ = {T_wien:.0f} K)",
        )
        plt.plot(-1, -1, "-", color="grey", lw=3, label="Illuminated dust")
        plt.plot(-1, -1, "-", color="lightgrey", lw=3, label="Shadowed dust")
        plt.ylim(total_sed.max() / 1e6, total_sed.max() * 1e1)
        plt.legend(loc=2, fontsize=9)
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("Blackbodies flux [arbitrary unit]")

    if verbose:
        print(f"Temperature law: r0 = {dmas[0]:2.2f} mas, T0 = {Tr[0]:.0f} K")

    return (
        np.array(spec) / N2,
        total_sed / N2,
        wl_sed * 1e6,
        T_wien,
        spec_all1.T / N2,
        spec_all2.T / N2,
        flux0 / N2,
        dmas,
        l_Tr,
    )


def visSpiralTemp(
    Utable,
    Vtable,
    Lambda,
    mjd,
    param,
    spec=None,
    verbose=True,
    display=True,
):
    """
    Compute complex visibility of an empty spiral.

    Params:
    -------
    rounds: {float}
        Number of turns,\n
    minThick: {float}
        Size of the smallest ring (given an opening angle) [rad],\n
    d_choc: {float}
        Dust formation radius [rad],\n
    anglePhi, angleSky, incl: {float}
        Orientaiton angle along the 3 axis [rad],\n
    opening_angle: {float}
        Opening angle [rad],\n
    compo: {str}
        Composition of the spiral ('r2': ring, 'g': , etc.),\n
    d_first_turn: {float}:
        Spiral step [rad],\n
    fillFactor: {int}
        Scale parameter to set the number of rings inside the spiral,\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    # Start using the '_compute_param_elts_spec' function to determine the elements
    # parameters composing the spiral. Used to easely get these parameters and
    # determine the SED of the actual elements of the spiral.
    tab_orient, tab_faceon, typei, N2, r_nuc, _step, _alpha = _compute_param_elts_spec(
        mjd, param, verbose=verbose, display=display
    )

    x0, y0, x2, y2, fwhmx2, fwhmy2, angle2 = tab_orient
    list_param = []

    if (len(x2)) == 0:
        if verbose:
            rprint(
                "[red]Warning:No dust in the pinwheel given the parameters fov/rnuc/rs.",
                file=sys.stderr,
            )

        C = np.zeros(len(Utable))
        return C

    if len(Lambda) != len(x2):
        spiral_params = zip(x2, y2, fwhmx2, fwhmy2, angle2, strict=False)
        for x0_val, y0_val, major, minor, angle_val in spiral_params:
            list_param.append(
                {
                    "Lambda": Lambda[0],
                    "x0": x0_val,
                    "y0": y0_val,
                    "majorAxis": major,
                    "minorAxis": minor,
                    "angle": angle_val,
                }
            )
    else:
        spiral_params = zip(
            Lambda,
            x2,
            y2,
            fwhmx2,
            fwhmy2,
            angle2,
            strict=False,
        )
        for lambda_val, x0_val, y0_val, major, minor, angle_val in spiral_params:
            list_param.append(
                {
                    "Lambda": lambda_val,
                    "x0": x0_val,
                    "y0": y0_val,
                    "majorAxis": major,
                    "minorAxis": minor,
                    "angle": angle_val,
                }
            )

    if (param["q"] == 1.0) or (spec is None):
        spectrumi = list(np.linspace(1, 0, N2))
    else:
        dmas = rad2mas(np.sqrt(tab_faceon[2] ** 2 + tab_faceon[3] ** 2))
        Tin = param["T_sub"]
        q = param["q"]
        Tr = Tin * (dmas / rad2mas(r_nuc)) ** (-q)
        if verbose:
            print(f"Temperature law: r0 = {dmas[0]:2.2f} mas, T0 = {Tr[0]:.0f} K")
        spectrumi = spec

    C_centered = visMultipleResolved(
        Utable,
        Vtable,
        Lambda,
        typei,
        spec=spectrumi,
        list_param=list_param,
    )
    # print(C_centered.shape)
    x0, y0 = mas2rad(param["x0"]), mas2rad(param["y0"])
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visMultipleResolved(Utable, Vtable, Lambda, typei, *, spec, list_param):
    """Compute the complex visibility of a multi-component object."""
    n_obj = len(typei)
    nbl = 1 if np.isscalar(Utable) else len(Utable)
    nwl = 1 if np.isscalar(Lambda) else len(Lambda)
    corrFluxTmp = np.zeros([n_obj, nwl, nbl], dtype=complex)

    for i in range(n_obj):
        if typei[i] == "r":
            Ci = visEllipticalRing(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "t_r":
            Ci = visThickEllipticalRing(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "ud":
            Ci = visUniformDisk(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "e_ud":
            Ci = visEllipticalUniformDisk(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "gd":
            Ci = visGaussianDisk(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "e_gd":
            Ci = visEllipticalGaussianDisk(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "star":
            Ci = visPointSource(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "pwhl":
            Ci = visSpiralTemp(Utable, Vtable, Lambda, list_param[i])
        else:
            print("Model not yet in VisModels")
        spec2 = spec[i]
        ampli = 1
        if i < TOP_COMPONENTS:
            spec2 *= ampli

        corrFluxTmp[i, :, :] = spec2 * Ci

    corrFlux = corrFluxTmp.sum(axis=0)
    flux = np.sum(spec, 0)
    try:
        vis = corrFlux / flux
    except Exception:
        print("Error : flux = 0")
        vis = None
    return vis


def visPwhl(Utable, Vtable, Lambda, param, verbose=False, expert_plot=False):
    """
    Compute a multi-component model of a pinwheel nebula with:
    - Pinwheel: multiple rings, uniform disks, or gaussian components selected
      via `compo`. Geometry follows `step`, `opening_angle`, etc. Fluxes come
      from a blackbody law based on `T_sub`, `r_nuc`, and `q`.
    - Binary: obtained with `binary_integrator` using `M1`, `M2`, `e`, and `dpc`,
      or with a manual separation `s_bin`. Relative flux (`contrib_star` in [%])
      is evaluated at 1 µm from blackbodies `T_WR`, `T_OB`, and the pinwheel SED.
    - Halo (optional): fully resolved environment (cf. Lazareff et al. 2017)
      set by `contrib_halo` [%] and drawing flux from the spiral contribution.
    """

    # param = param2.copy()
    mjd = param["mjd"]  # 50000.0
    phase = (mjd - param["mjd0"]) / param["P"] % 1
    if verbose:
        s = "Model pinwheel S = {:2.1f} mas, phase = {:1.2f} @ {:2.2f} µm:".format(
            param["step"],
            phase,
            Lambda[0] * 1e6,
        )
        rprint(f"[cyan]{s}.")
        rprint(f"[cyan]{'-' * len(s)}.")

    angle_0 = param["angle_0"]
    angle_0_bis = (angle_0 - 0) * -1

    # -= 90  # Switch reference to the NORTH
    param["angle_0"] = angle_0_bis

    param["incl"] = param["incl"] + 180
    # Binary point source
    # --------------------------------------------------------------------------
    if ("M1" in param) & ("M2" in param):
        tab = getBinaryPos(
            mjd, param, mjd0=param["mjd0"], revol=1, v=2, au=True, display=expert_plot
        )
        param["a"] = tab["a"]
        param_star_WR = {
            "x0": mas2rad(tab["star1"]["x"]),
            "y0": mas2rad(tab["star1"]["y"]),
        }
        param_star_O = {
            "x0": mas2rad(tab["star2"]["x"]),
            "y0": mas2rad(tab["star2"]["y"]),
        }
    else:
        param["a"] = param["sep_bin"] * param["dpc"]
        param_star_WR = {
            "x0": 0,
            "y0": 0,
        }
        param_star_O = {
            "x0": 0,
            "y0": 0,
        }

    # Flux contribution of the different components (binary star, pinwheel and
    # resolved environment)
    # --------------------------------------------------------------------------
    wl_0 = 1e-6  # Wavelength 0 for the ratio

    # Contribution of each star in the binary system
    p_OB, p_WR = computeBinaryRatio(param, Lambda)
    p_OB = np.mean(p_OB)
    p_WR = np.mean(p_WR)
    if verbose:
        print(f"Binary relative fluxes: WR = {p_WR * 100:2.2f} %, OB = {p_OB * 100.0:2.2f} %")

    contrib_star = param["contrib_star"] / 100.0

    wl = np.array([Lambda]) if isinstance(Lambda, float) else Lambda

    wl_sed = np.logspace(-7, -3.5, 1000)
    if param["r_nuc"] != 0:
        input_wl = [wl_0, *list(wl)]
        tab_dust_fluxes = sed_pwhl(input_wl, mjd, param, display=False, verbose=False)
        full_sed = tab_dust_fluxes[1]
        wl_sed = tab_dust_fluxes[2]
        f_pinwheel_wl0 = tab_dust_fluxes[6]

        f_binary_wl0 = p_OB * planck_law(param["T_OB"], wl_0) + p_WR * planck_law(
            param["T_WR"], wl_0
        )

        f_binary_wl = p_OB * planck_law(param["T_OB"], wl_sed / 1e6) + p_WR * planck_law(
            param["T_WR"], wl_sed / 1e6
        )

        f_binary_obs = p_OB * planck_law(param["T_OB"], wl) + p_WR * planck_law(param["T_WR"], wl)

    if param["r_nuc"] != 0:
        sed_pwhl_wl = tab_dust_fluxes[0][1, :]
        P_dust = np.sum(sed_pwhl_wl)
        n_elts = len(sed_pwhl_wl)

        if contrib_star != 1:
            scale_star = (f_pinwheel_wl0 / f_binary_wl0) * (contrib_star / (1.0 - contrib_star))
        else:
            scale_star = 1e6
    else:
        P_dust = 1 - param["contrib_star"] / 100.0
        n_elts = param["nelts"]
        sed_pwhl_wl = None

    wl_m = wl * 1e6
    l_wl = [wl_m] * n_elts

    if param["r_nuc"] != 0:
        binary_sed = scale_star * f_binary_wl
        P_star = scale_star * f_binary_obs
        dmas = tab_dust_fluxes[7]
        Tr = tab_dust_fluxes[8]
        Twien = tab_dust_fluxes[3]
    else:
        P_star = param["contrib_star"] / 100.0

    if (expert_plot) & (param["r_nuc"] != 0):
        plt.figure()
        plt.plot(dmas, Tr, label=f"r0 = {dmas[0]:2.1f} mas, T0 = {Tr[0]:2.1f} K")
        plt.grid(alpha=0.2)
        plt.xlabel("Distance [mas]")
        plt.ylabel("Temperature [K]")
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.loglog(
            wl_sed,
            full_sed,
            color="#af6d04",
            lw=3,
            alpha=0.8,
            label=rf"Pinwheel (T$_{{wien}}$ = {Twien:.0f} K)",
        )
        plt.loglog(
            wl_sed,
            binary_sed,
            color="#008080",
            lw=3,
            alpha=0.8,
            label="Binary",
            zorder=3,
        )
        plt.loglog(l_wl, sed_pwhl_wl, ".", ms=3, color="#222223", zorder=10)
        plt.loglog(wl_sed, tab_dust_fluxes[4], "-", color="grey")
        with contextlib.suppress(ValueError):
            plt.loglog(wl_sed, tab_dust_fluxes[5], "-", color="lightgrey")

        max_plot = full_sed.max()
        plt.ylim(max_plot / 1e6, max_plot * 1e2)
        plt.plot(-1, -1, "-", color="grey", lw=3, label="Illuminated dust")  # legend
        plt.plot(-1, -1, "-", color="lightgrey", lw=3, label="Shadowed dust")  # legend
        plt.loglog(
            wl_m,
            P_dust,
            "H",
            color="#71490a",
            zorder=5,
            ms=5,
            markeredgecolor="k",
            markeredgewidth=0.5,
            # label=f"$\Sigma F_{{{wl_m:2.1f}µm}}$ = {P_dust:2.1f} Jy",
        )
        plt.loglog(
            wl_m,
            P_star,
            "H",
            color="#009ace",
            zorder=5,
            ms=5,
            markeredgecolor="k",
            markeredgewidth=0.5,
            # label=f"$\Sigma F_{{*, {wl_m:2.1f}µm}}$ = {P_star:2.1f} Jy",
        )
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("SED [Jy]")
        plt.legend(loc=1, fontsize=8)
        plt.grid(alpha=0.1, which="both")
        plt.tight_layout()

    if contrib_star == 1:
        P_star = 1
        P_dust = 0

    P_tot = P_star + P_dust

    # # Different contributions
    Fstar = P_star / P_tot
    Fpwhl = P_dust / P_tot

    # halo background
    # --------------------------------------------------------------------------
    if "contrib_halo" in param:
        Fhalo = param["contrib_halo"] / 100.0
        Fpwhl -= Fhalo

    if verbose:
        print(
            f"Relative component fluxes: Fstar = {100 * Fstar:2.3f} %; "
            f"Fpwhl = {100 * Fpwhl:2.3f} %, "
            f"Fenv = {100 * Fhalo:2.3f} %"
        )

    # # Visibility
    # --------------------------------------------------------------------------
    param["x0"] = rad2mas(param_star_O["x0"]) * (2 / 3.0)
    param["y0"] = rad2mas(param_star_O["y0"]) * (2 / 3.0)

    thickness = param["thickness"]
    Vpwhl = visSpiralTemp(
        Utable,
        Vtable,
        wl,
        mjd,
        param,
        spec=sed_pwhl_wl,
        verbose=verbose,
        display=expert_plot,
    ) * visGaussianDisk(Utable, Vtable, Lambda, {"fwhm": thickness, "x0": 0.0, "y0": 0.0})

    vis_OB = p_OB * Fstar[:, None] * visPointSource(Utable, Vtable, wl, param_star_O)
    vis_WR = p_WR * Fstar[:, None] * visPointSource(Utable, Vtable, wl, param_star_WR)

    ftot = Fstar + Fpwhl + Fhalo

    vis = (Fpwhl[:, None] * Vpwhl + vis_OB + vis_WR) / ftot[:, None]

    return vis
