"""
@author: Anthony Soulain (University of Sydney)
-----------------------------------------------------------------
OIMALIB: Optical Interferometry Modelisation and Analysis Library
-----------------------------------------------------------------

OIFITS related function.
-----------------------------------------------------------------
"""

import contextlib
import os
import sys
from glob import glob

import numpy as np
from astropy.io import fits
from munch import munchify as dict2class

from oimalib.plotting import dic_color


def _compute_dic_index(index_ref, teles_ref):
    dic_index = {}
    for i in range(len(index_ref)):
        ind = index_ref[i]
        tel = teles_ref[i]
        if ind not in dic_index:
            dic_index[ind] = tel
    return dic_index


def _compute_bl_name(index, index_ref, teles_ref):
    """Compute baseline name and check if the appropriate color
    is already associated (for the VLTI)."""
    dic_index = _compute_dic_index(index_ref, teles_ref)

    list_bl_name = []
    nbl = len(index)
    for i in range(nbl):
        base = f"{dic_index[index[i][0]]}-{dic_index[index[i][1]]}"
        base2 = f"{dic_index[index[i][1]]}-{dic_index[index[i][0]]}"
        if base in list(dic_color.keys()):
            baseline_name = base
        elif base2 in list(dic_color.keys()):
            baseline_name = base2
        else:
            baseline_name = base
        list_bl_name.append(baseline_name)
    list_bl_name = np.array(list_bl_name)
    return list_bl_name


def _compute_cp_name(index_cp, index_ref, teles_ref):
    """Compute triplet name and check if the appropriate color
    is already associated (for the VLTI)."""
    ncp = len(index_cp)
    dic_index = _compute_dic_index(index_ref, teles_ref)

    list_cp_name = []
    for i in range(ncp):
        b1 = dic_index[index_cp[i][0]]
        b2 = dic_index[index_cp[i][1]]
        b3 = dic_index[index_cp[i][2]]
        triplet = f"{b1}-{b2}-{b3}"
        list_cp_name.append(triplet)
    list_cp_name = np.array(list_cp_name)
    return list_cp_name


def dir2data(filedir, ext="fits"):
    """
    Format all data from different oifits files in filedir to the list usable
    by the other functions.
    """
    listfile = glob(os.path.join(filedir, f"*.{ext}"))
    tab = []
    for f in listfile:
        data = load(f, cam="SC")
        tab.append(data)
    return tab


def get_condition(list_data):
    """Plot the weather conditions (seeing, tau0) for a list of oifits files."""
    l_seeing = []
    l_tau = []
    l_mjd = []
    for i, _ in enumerate(list_data):
        filename = list_data[i].info.filename
        hdu = fits.open(filename)
        hdr = hdu[0].header
        hdu.close()
        seeing1 = hdr["HIERARCH ESO ISS AMBI FWHM START"]
        seeing2 = hdr["HIERARCH ESO ISS AMBI FWHM END"]
        tau1 = hdr["HIERARCH ESO ISS AMBI TAU0 START"]
        tau2 = hdr["HIERARCH ESO ISS AMBI TAU0 END"]

        mjd = hdr["MJD-OBS"]
        if i == 0:
            mjd0 = mjd
        l_mjd.append(mjd)

        tau = np.mean([tau1, tau2]) * 1e3
        seeing = np.mean([seeing1, seeing2])
        l_seeing.append(seeing)
        l_tau.append(tau)

    output = {
        "tau": np.array(l_tau),
        "seeing": np.array(l_seeing),
        "mjd": np.array(l_mjd),
        "mjd0": mjd0,
    }

    return dict2class(output)


def load(namefile, cam="SC", split=False, simu=False, pola=None):
    fitsHandler = fits.open(namefile)

    # OI_TARGET table
    target = fitsHandler["OI_TARGET"].data.field("TARGET")

    # OI_WAVELENGTH table
    ins = fitsHandler["OI_WAVELENGTH"].header["INSNAME"]

    try:
        mjd = fitsHandler[0].header["MJD-OBS"]
    except KeyError:
        mjd = np.nan

    if "GRAVITY" in ins:
        index_cam = 10
        if split:
            index_cam += pola

        if cam == "FT":
            index_cam = 20
            if split:
                index_cam += pola
    else:
        index_cam = 1

    if simu:
        index_cam = None

    try:
        wave = fitsHandler["OI_WAVELENGTH", index_cam].data.field("EFF_WAVE")
    except KeyError:
        wave = np.zeros(1)

    # OI_FLUX table
    try:
        spectre = fitsHandler["OI_FLUX", index_cam].data.field("FLUXDATA")
        sta_index = fitsHandler["OI_FLUX", index_cam].data.field("STA_INDEX")
    except KeyError:
        try:
            spectre = fitsHandler["OI_FLUX", index_cam].data.field("FLUX")
            sta_index = fitsHandler["OI_FLUX", index_cam].data.field("STA_INDEX")
        except KeyError:
            spectre = np.zeros(1)
            sta_index = np.zeros(1)

    nspec = spectre.shape[0]

    # OI_ARRAY table
    index_ref = fitsHandler["OI_ARRAY"].data.field("STA_INDEX")
    teles_ref = fitsHandler["OI_ARRAY"].data.field("STA_NAME")
    array = fitsHandler["OI_ARRAY"].header["ARRNAME"]

    dic_index = _compute_dic_index(index_ref, teles_ref)

    tel = []
    for i in range(nspec):
        with contextlib.suppress(KeyError):
            tel.append(dic_index[sta_index[i]])
    tel = np.array(tel)

    # OI_T3 table
    try:
        cp = fitsHandler["OI_T3", index_cam].data.field("T3PHI")
    except KeyError:
        test_cp = isinstance(fitsHandler["OI_T3", None].data.field("T3PHI"), np.ndarray)
        fitsHandler.close()
        if test_cp:
            print(
                "Your dataset seems to be a simulation (from aspro2), you should"
                + "add simu=True.",
                file=sys.stderr,
            )
        else:
            print(
                "Your dataset have not OI_T3 with the supported index (%i)"
                + ", try another dataset.",
                file=sys.stderr,
            )
        return None

    e_cp = fitsHandler["OI_T3", index_cam].data.field("T3PHIERR")
    index_cp = fitsHandler["OI_T3", index_cam].data.field("STA_INDEX")
    flag_cp = fitsHandler["OI_T3", index_cam].data.field("FLAG")
    u1 = fitsHandler["OI_T3", index_cam].data.field("U1COORD")
    u2 = fitsHandler["OI_T3", index_cam].data.field("U2COORD")
    v1 = fitsHandler["OI_T3", index_cam].data.field("V1COORD")
    v2 = fitsHandler["OI_T3", index_cam].data.field("V2COORD")
    u3 = -(u1 + u2)
    v3 = -(v1 + v2)

    # OI_VIS2 table
    vis2 = fitsHandler["OI_VIS2", index_cam].data.field("VIS2DATA")
    e_vis2 = fitsHandler["OI_VIS2", index_cam].data.field("VIS2ERR")
    index_vis2 = fitsHandler["OI_VIS2", index_cam].data.field("STA_INDEX")
    flag_vis2 = fitsHandler["OI_VIS2", index_cam].data.field("FLAG")
    u = fitsHandler["OI_VIS2", index_cam].data.field("UCOORD")
    v = fitsHandler["OI_VIS2", index_cam].data.field("VCOORD")
    B = np.sqrt(u**2 + v**2)

    try:
        # OI_VIS table
        dvis = fitsHandler["OI_VIS", index_cam].data.field("VISAMP")
        e_dvis = fitsHandler["OI_VIS", index_cam].data.field("VISAMPERR")
        dphi = fitsHandler["OI_VIS", index_cam].data.field("VISPHI")
        e_dphi = fitsHandler["OI_VIS", index_cam].data.field("VISPHIERR")
        flag_dvis = fitsHandler["OI_VIS", index_cam].data.field("FLAG")
    except KeyError:
        dvis = np.zeros_like(vis2)
        e_dvis = np.zeros_like(vis2)
        dphi = np.zeros_like(vis2)
        e_dphi = np.zeros_like(vis2)
        flag_dvis = flag_vis2

    try:
        dat = fitsHandler[0].header["DATE-OBS"]
    except KeyError:
        dat = np.nan

    fitsHandler.close()

    info = {
        "Ins": ins,
        "Index": index_ref,
        "Config": teles_ref,
        "Target": target,
        "Bmax": B.max(),
        "Array": array,
        "nbl": len(u),
        "ncp": len(u1),
        "mjd": mjd,
        "Date": dat,
        "filename": namefile,
    }

    # Compute freq, blname
    freq_cp, freq_vis2, bl_cp = [], [], []

    for i in range(len(u1)):
        B1 = np.sqrt(u1[i] ** 2 + v1[i] ** 2)
        B2 = np.sqrt(u2[i] ** 2 + v2[i] ** 2)
        B3 = np.sqrt(u3[i] ** 2 + v3[i] ** 2)

        Bmax = np.max([B1, B2, B3])
        bl_cp.append(Bmax)
        freq_cp.append(Bmax / wave / 206264.806247)  # convert to arcsec-1

    for i in range(len(u)):
        freq_vis2.append(B[i] / wave / 206264.806247)  # convert to arcsec-1

    freq_cp = np.array(freq_cp)
    freq_vis2 = np.array(freq_vis2)
    bl_cp = np.array(bl_cp)

    blname = _compute_bl_name(index_vis2, index_ref, teles_ref)
    cpname = _compute_cp_name(index_cp, index_ref, teles_ref)

    dic_output = {
        "flux": spectre,
        "vis2": vis2,
        "e_vis2": e_vis2,
        "cp": cp,
        "e_cp": e_cp,
        "dvis": dvis,
        "e_dvis": e_dvis,
        "dphi": dphi,
        "e_dphi": e_dphi,
        "wl": wave,
        "u": u,
        "v": v,
        "u1": u1,
        "u2": u2,
        "u3": u3,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "cpname": cpname,
        "teles_ref": teles_ref,
        "index_ref": index_ref,
        "blname": blname,
        "bl": B,
        "bl_cp": bl_cp,
        "index": index_vis2,
        "index_cp": index_cp,
        "freq_cp": freq_cp,
        "freq_vis2": freq_vis2,
        "tel": tel,
        "flag_vis2": flag_vis2,
        "flag_cp": flag_cp,
        "flag_dvis": flag_dvis,
        "info": info,
    }

    output = dict2class(dic_output)
    return output
