import sys

import numpy as np
import scipy

if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources


def gravity_wave():
    from astropy.io import fits

    datadir = importlib_resources.files("magneto") / "internal_data"
    grav = datadir / "gravity_example.fits"

    with fits.open(grav) as fd:
        grawav = fd["OI_WAVELENGTH", 10].data.field("EFF_WAVE") * 1e6
    return grawav


def shiftcomp(angle, shift):
    if isinstance(angle, np.ndarray) is False and isinstance(shift, list) is False:
        if angle < 0:
            angle = 360 - abs(angle)

        while angle > 360:
            angle = angle - 360

        while angle < 0:
            angle = angle + 360

        if angle >= 0 and angle <= 90:
            north = shift * np.cos(np.deg2rad(angle))
            east = shift * np.sin(np.deg2rad(angle))
        if angle > 90 and angle <= 180:
            north = -shift * np.cos(np.deg2rad(180 - angle))
            east = shift * np.sin(np.deg2rad(180 - angle))

        if angle > 180 and angle <= 270:
            north = -shift * np.sin(np.deg2rad(270 - angle))
            east = -shift * np.cos(np.deg2rad(270 - angle))

        if angle > 270 and angle <= 360:
            north = shift * np.cos(np.deg2rad(360 - angle))
            east = -shift * np.sin(np.deg2rad(360 - angle))

    else:
        east = []
        north = []

        angle = [360 - abs(x) if x < 0 else x for x in angle]
        for i in range(len(angle)):
            if angle[i] >= 0 and angle[i] <= 90:
                north.append(shift[i] * np.cos(np.deg2rad(angle[i])))
                east.append(shift[i] * np.sin(np.deg2rad(angle[i])))
            if angle[i] > 90 and angle[i] <= 180:
                north.append(-shift[i] * np.cos(np.deg2rad(180 - angle[i])))
                east.append(shift[i] * np.sin(np.deg2rad(180 - angle[i])))

            if angle[i] > 180 and angle[i] <= 270:
                north.append(-shift[i] * np.sin(np.deg2rad(270 - angle[i])))
                east.append(-shift[i] * np.cos(np.deg2rad(270 - angle[i])))

            if angle[i] > 270 and angle[i] <= 360:
                north.append(shift[i] * np.cos(np.deg2rad(360 - angle[i])))
                east.append(-shift[i] * np.sin(np.deg2rad(360 - angle[i])))

    return east, north


def masinangular(x):  # mas -> rad
    val = x / 206264806.2471
    return val


def ellipse(size, inc, angle, ori):
    angle = 90 - angle
    ori = 90 - ori
    # print(angle+ori)
    x = size * np.cos(np.deg2rad(angle)) * np.cos(np.deg2rad(ori)) - size * np.sin(
        np.deg2rad(angle)
    ) * np.sin(np.deg2rad(ori)) * np.cos(np.deg2rad(inc))
    y = size * np.cos(np.deg2rad(angle)) * np.sin(np.deg2rad(ori)) + size * np.sin(
        np.deg2rad(angle)
    ) * np.cos(np.deg2rad(ori)) * np.cos(np.deg2rad(inc))

    proj = np.sqrt(x**2 + y**2)

    return masinangular(proj)


def ellipseshift(angle, shift):
    x = 0
    y = 0
    east_shifted, north_shifted = shiftcomp(angle, shift)
    x = x + east_shifted
    y = y + north_shifted
    return x, y


def contvis(size, bangle, inc, ori, wave, blength):
    # bangle=bangle-(ori-90)
    return projvis2(ellipse(size, inc, bangle, ori), wave, blength)


def totflux(flux, irex):
    return (flux + irex) / (1 + irex)


def projvis2(size, wave, baseline):
    bess_arg = 2 * np.pi * size * baseline / wave
    vis = scipy.special.jv(0, bess_arg) * np.exp(
        -((2 * size * baseline / wave) ** 2) * np.pi**2 / (4 * np.log(2))
    )

    return vis


def totvis(irex, cont, flux, vis, DS):
    return (DS * (irex + 1) * cont + flux * vis) / (irex + flux)


def totphase(flux, vis, irex, DS, contvis, phase):
    phase = np.deg2rad(phase)
    return np.rad2deg(
        np.arcsin(flux * vis / (DS * (irex + 1) * contvis + flux * vis) * np.sin(phase))
    )


def plvis(vis, vcont, FLC):
    erg = (vis * FLC - vcont) / (FLC - 1)
    return erg


def pldp(dp, vis, plvis, FLC):
    dp = np.deg2rad(dp)

    pdp = np.arcsin(np.sin(dp) * vis / plvis * FLC / (FLC - 1))

    return np.rad2deg(pdp)
