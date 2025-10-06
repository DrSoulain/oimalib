import importlib.resources as importlib_resources

import numpy as np
import scipy
from astropy.io import fits
from scipy.interpolate import interp1d


def _find_nearest(array, value):
    """Find the nearest index of the value in the array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def mas2rad(mas):
    """Convert angle in milli arc-sec to radians."""
    rad = mas * (10 ** (-3)) / (3600 * 180 / np.pi)
    return rad


def gravity_wave():
    """Get the GRAVITY wavelenghts table from internal package dataset."""
    datadir = importlib_resources.files("oimalib") / "internal_data"
    grav = datadir / "gravity_example.fits"

    with fits.open(grav) as fd:
        grawav = fd["OI_WAVELENGTH", 10].data.field("EFF_WAVE") * 1e6
    return grawav


def shiftcomp(angle, shift):
    east = []
    north = []

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


def masinangular(x):
    val = x / 206264806.2471
    return val


def ellipse(size, inc, angle, ori):
    angle = 90 - angle
    ori = 90 - ori

    x = size * np.cos(np.deg2rad(angle)) * np.cos(np.deg2rad(ori)) - size * np.sin(
        np.deg2rad(angle)
    ) * np.sin(np.deg2rad(ori)) * np.cos(np.deg2rad(inc))
    y = size * np.cos(np.deg2rad(angle)) * np.sin(np.deg2rad(ori)) + size * np.sin(
        np.deg2rad(angle)
    ) * np.cos(np.deg2rad(ori)) * np.cos(np.deg2rad(inc))

    proj = np.sqrt(x**2 + y**2)

    return masinangular(proj)


def ellipseshift(angle, shift):
    east_shifted, north_shifted = shiftcomp(angle, shift)
    return east_shifted, north_shifted


def contvis(size, bangle, inc, ori, wave, blength):
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
    inp = np.sin(dp) * vis / plvis * FLC / (FLC - 1)

    # Check for unpermitted values in arcsin (-1:1)
    inp = min(inp, 1)
    inp = max(inp, -1)

    pdp = np.arcsin(inp)
    return np.rad2deg(pdp)


def pcshift(waves, dp, blength):
    waves = waves * 10 ** (-6)

    shift = -dp * waves / (2 * np.pi * blength)

    return shift


def shiftfit(angle, totalshift, totalangle):
    MAX_ANGLE = 180.0
    if totalangle - angle.all() > MAX_ANGLE:
        return -totalshift * np.cos(np.deg2rad(totalangle - (angle - 180)))

    else:
        return totalshift * np.cos(np.deg2rad(totalangle - angle))


def cube_interpolator(Model, wl=None):
    """
    Interpolate the data cube across the wavelength range and provide a new
    interpolated cube based on the input wavelength table `wl`. The default
    setting calculates the interpolated grid using the pure line wave model
    `Model.plwl`. For more information, refer to the `get_pureline()` function.
    """

    if wl is None:
        wl = Model.plwl

    cube = Model.cube
    fct_cube_interp = np.array(
        [
            [
                [
                    interp1d(Model.waves, cube[n, :, x, y], kind="nearest")
                    for y in range(len(cube[0, 0, 0, :]))
                ]
                for x in range(len(cube[0, 0, :, 0]))
            ]
            for n in range(len(cube[:, 0, 0, 0]))
        ]
    )

    cube_interpolated = np.array(
        [
            [
                [fct_cube_interp[n, x, y](wl) for y in range(len(cube[0, 0, 0, :]))]
                for x in range(len(cube[0, 0, :, 0]))
            ]
            for n in range(len(cube[:, 0, 0, 0]))
        ]
    )
    return np.squeeze(cube_interpolated)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(x, y)
    phi_deg = 360 + np.rad2deg(phi) if np.rad2deg(phi) < 0 else np.rad2deg(phi)
    return (rho, phi_deg)
