import os

import numpy as np
import pandas
from astropy import constants as cs, units as u
from munch import munchify

from magneto.alex import _find_nearest

brg_void = 2.1661178  # Value from Alana (SPIRou reference)


def _get_model_params_from_key(key, file_header="keymod.txt"):
    """ """
    import linecache

    return list(
        map(float, linecache.getline(file_header, key + 3).split("/")[1].split())
    )


def _azim_to_phase(z, reverse=False, clock=False, verbose=False):
    """Transform the azim (`z`) into orbital phase (phase = 0 is when the
    spot/shock face the observer). If reverse == True, computes the azimuth
    position from the given orbital phase (`z`)."""

    azim = np.arange(-90, 270, 5)
    phase = np.roll((azim - 90) / 360.0, len(azim) // 2)
    phase[phase < 0] = phase[phase < 0] + 1

    azim_roll = np.roll(azim, len(azim) // 2)

    if not reverse:
        if z == 270:
            z = -90
        ind = _find_nearest(azim_roll, z)
        if verbose:
            print("Closest value in the grid is %i (%i) deg" % (azim_roll[ind], z))
        out = phase[ind]
    else:
        if clock:
            if z != 0:
                z = 1 - z
        ind = _find_nearest(phase, z)
        out = azim_roll[ind]
        if verbose:
            print(
                "\nClosest phase in the grid is %2.3f (%2.3f) -> z = %i deg"
                % (phase[ind], z, out)
            )
    return out


def get_spectrum(hdu, fs=1, void=True, norm=True):
    """Compute and normalize (if `norm`==True) the spectrum. If `fs` < 1, the
    spectrum is diluted to take into account the disk contribution."""
    data = np.squeeze(hdu[0].data)
    wave = hdu[1].data / 1e3

    spectrum = np.array([x.sum() for x in data])

    rest = hdu[0].header["LAMBDA0"] / 1e3
    vel = ((wave - rest) / (rest)) * cs.c / 1e3

    if norm:
        left_cont, left_wl = spectrum[0], wave[0]
        right_cont, right_wl = spectrum[-1], wave[-1]
        m, b = np.polyfit([left_wl, right_wl], [left_cont, right_cont], 1)
        norm = m * wave + b
        spectrum /= norm

    spectrum = spectrum * fs + (1 - fs)

    offset = brg_void - rest
    if void:
        wave += offset
    return wave, vel, spectrum


def find_model_key(masterdir, tilt=None, Tmax=None, frac=None, width=None):
    """Retrieve the different models available within a given directory
    `masterdir`. You can specify a magnetic obliquity (`tilt`), a maximum
    temperature (`Tmax`) and the size of the magnetosphere (`frac`). `frac` need
    to be in %, fraction of the co-rotation radius."""
    pandas.options.mode.chained_assignment = None

    header = ["key", "", "Rco", "Racc", "Mdot", "tilt", "Tmax", "ri", "ro"]
    print(masterdir)
    log = pandas.read_csv(
        masterdir + "log.txt", skiprows=3, delimiter=r"\s+", names=header
    )

    log["frac"] = (1e2 * round(log["ro"] / log["Rco"], 2)).astype(int)
    log["width"] = (1e2 * round((log["ro"] - log["ri"]) / log["ro"], 2)).astype(int)

    c1 = log["tilt"] > 0
    if tilt is not None:
        c1 = log["tilt"] == tilt
    c2 = log["Tmax"] > 0
    if Tmax is not None:
        c2 = log["Tmax"] == Tmax
    c3 = log["frac"] > 0
    if frac is not None:
        c3 = log["frac"] == frac
    c4 = log["width"] > 0
    if width is not None:
        c4 = log["width"] == width

    pd = log.loc[(c1) & (c2) & (c3) & (c4)]

    select_key = list(pd.key)

    good = 0
    l_good = []
    for k in select_key:
        params = _get_model_params_from_key(k, masterdir + "log.txt")

        m_acc = 10 ** params[2]
        mod_id = (
            "%i_" % k
            + "b%d" % params[3]
            + "m%.2e" % m_acc
            + "ri%d" % (10 * params[5])
            + "ro%d" % (10 * params[6])
            + "t%d" % int(params[4])
            + "P%d" % (9)
            + "Tps%d/" % (int(1e4))
        )
        modeldir = masterdir + mod_id
        if os.path.exists(modeldir):
            l_good.append(True)
            good += 1
        else:
            l_good.append(False)

    pd["Done"] = l_good

    print("\n# Requested parameters have %i models (%i done)" % (len(select_key), good))
    print(pd)
    return select_key, pd


def get_model_param(key, phase, incl=55, P=9, rs=2, clock=True, md=None, verbose=False):
    """Open hdu and retrieve parameters of the mcfost models.

    Parameters:
    -----------
    `key` {int}:
       Key of the model (run find_model_key() to check for existing models),\n
    `z` {int}:
       Azimuth (rotational phase) of the image (check azim_to_phase()),\n
    `incl` {int}:
       Inclination  of the model [deg].\n

    Returns:
    --------
    `filename`:
        Name of the model file,\n
    `dparam` {dict}:
        Dictionnary with the parameters of the model.
    """
    if md is None:
        masterdir = (
            "/Volumes/PowerStone/Models_MCFOST/"
            + "grid_citau_nr64_nt64_np100_order2_mcTrue/"
        )
    else:
        masterdir = md

    list_line = {"brg": 2.1661, "bra": 4.0523, "pab": 1.2822}

    params = _get_model_params_from_key(key, masterdir + "log.txt")

    m_acc = 10 ** params[2]
    mod_id = (
        "%i_" % key
        + "b%d" % params[3]
        + "m%.2e" % m_acc
        + "ri%d" % (10 * params[5])
        + "ro%d" % (10 * params[6])
        + "t%d" % int(params[4])
        + "P%d" % (int(P))
        + "Tps%d/" % (int(1e4))
    )

    modeldir = masterdir + mod_id

    z = _azim_to_phase(phase, reverse=True, clock=clock, verbose=verbose)
    if (z < 100.0) & (z > 5):
        s_z = " %2.1f" % z
    elif (z <= 5) & (z > -10):
        s_z = "  %2.1f" % z
    else:
        s_z = "%2.1f" % z

    mfile = "H_line_{}_i= {:2.1f}a={}.fits.gz".format(list_line["brg"], incl, s_z)

    rstar = (rs * cs.R_sun).to(u.au).value

    dparam = munchify(
        {
            "key": key,
            "m_acc": m_acc,
            "P": P,
            "tilt": int(params[3]),
            "Tmax": params[4],
            "ri": params[5],
            "ro": params[6],
            "rco": params[0],
            "phase": phase,
            "frac": round(params[6] / params[0], 2),
            "width": (params[6] - params[5]) / params[6],
            "azim": z,
            "rs": rstar,
            "incl": incl,
        }
    )
    return (
        modeldir + mfile,
        dparam,
    )
