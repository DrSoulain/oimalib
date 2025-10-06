"""
@author: Anthony Soulain (Grenoble Alpes University, IPAG)
-----------------------------------------------------------------
oimalib: optical interferometry modelisation and analysis library
-----------------------------------------------------------------

Set of function to perform data selection.
-----------------------------------------------------------------
"""

from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from munch import munchify
from uncertainties import ufloat

from oimalib.fitting import (
    leastsqFit,
    model_pcshift,
    perform_fit_dphi,
    perform_fit_dvis,
    select_model,
)
from oimalib.tools import (
    binning_tab,
    cart2pol,
    compute_oriented_shift,
    nan_interp,
    normalize_continuum,
    rad2mas,
    wtmn,
)


def _check_bl_same(list_data):
    """Check if list_data contains only same VLTI configuration."""
    blname_ref = list_data[0].blname
    diff_bl = []
    for i in range(len(list_data) - 1):
        blname = list_data[i + 1].blname
        n_diff = len(set(blname_ref == blname))
        if n_diff != 1:
            diff_bl.append([i, blname])
    ckeck = True
    if len(diff_bl) != 0:
        print(f"Different BL found compared to the reference ({blname_ref!s})")
        print(diff_bl)
        ckeck = False
    return ckeck


def _check_good_tel(list_data, verbose=True):
    """Check if one telescope if down in the list of dataset."""
    data_ref = list_data[0]
    blname = data_ref.blname
    nbl = len(blname)

    l_bad = []
    for i in range(nbl):
        dvis = data_ref.dvis[i]
        cond_nan = np.isnan(dvis)
        d_vis_sel = dvis[~cond_nan]
        if len(d_vis_sel) == 0:
            l_bad.append(blname[i])

    if len(l_bad) != 0 and verbose:
        print("\n## Warning: only nan detected in baselines:", l_bad)

    i_bl_bad = np.zeros(len(list_data[0].tel))
    l_tel = list(set(list_data[0].tel))

    if len(l_bad) != 0:
        for bad in l_bad:
            for i, tel in enumerate(l_tel):
                if tel in bad:
                    i_bl_bad[i] += 1

    nbad = len(i_bl_bad)
    max_bad = np.max(i_bl_bad)

    exclude_tel = []
    if len(l_bad) != 0:
        exclude_tel = [l_tel[i] for i in range(nbad) if i_bl_bad[i] == max_bad]

    if len(l_bad) != 0 and verbose:
        print(
            "-> so telescopes seem to be down and are automaticaly excluded:",
            exclude_tel,
        )
    return exclude_tel, l_tel


def _select_bl(list_data, blname, exclude_tel):
    """Compute the bl index and name if some need to be excluded (e.g.: down telescope)."""
    if len(exclude_tel) == 0:
        good_bl = range(len(list_data[0].blname))
        good_bl_name = blname
    else:
        good_bl, good_bl_name = [], []
        for i in range(len(blname)):
            for tel in exclude_tel:
                if tel not in blname[i]:
                    good_bl.append(i)
                    good_bl_name.append(blname[i])
    return good_bl, good_bl_name


def _select_data_v2(
    i,
    data,
    use_flag=False,
    cond_uncer=False,
    cond_wl=False,
    wl_bounds=None,
    rel_max=None,
):
    """Select data V2 using different criteria (errors, wavelenght, flag)."""
    nwl = len(data.wl)
    sel_flag = sel_err = sel_wl = np.array([False] * nwl)

    if use_flag:
        sel_flag = data.flag_vis2[i]

    vis2 = data.vis2[i, :]
    e_vis2 = data.e_vis2[i]
    if cond_uncer:
        rel_err = e_vis2 / vis2
        sel_err = np.invert(rel_err <= rel_max * 1e-2)

    if cond_wl:
        try:
            sel_wl = np.invert((data.wl >= wl_bounds[0] * 1e-6) & (data.wl < wl_bounds[1] * 1e-6))
        except TypeError:
            print("wl_bounds is None, please give wavelength limits (e.g.: [2, 3])")
    cond_v2 = sel_flag | sel_err | sel_wl
    return cond_v2


def _select_data_cp(
    i,
    data,
    use_flag=False,
    cond_uncer=False,
    cond_wl=False,
    wl_bounds=None,
):
    """Select data CP using different criteria (errors, wavelenght, flag)."""
    nwl = len(data.wl)

    sel_flag = sel_err = sel_wl = np.array([False] * nwl)

    if use_flag:
        sel_flag = data.flag_cp[i]

    e_cp = data.e_cp[i]
    if cond_uncer:
        sel_err = e_cp < 0

    if cond_wl:
        try:
            sel_wl = np.invert((data.wl >= wl_bounds[0] * 1e-6) & (data.wl < wl_bounds[1] * 1e-6))
        except TypeError:
            print("wl_bounds is None, please give wavelength limits (e.g.: [2, 3])")

    cond_cp = sel_flag | sel_err | sel_wl

    return cond_cp


def _find_max_freq(list_data):
    l_fmin, l_fmax = [], []
    for data in list_data:
        tfmax = data.freq_vis2.flatten().max()
        tfmin = data.freq_vis2.flatten().min()
        l_fmax.append(tfmax)
        l_fmin.append(tfmin)

    fmin = np.array(l_fmin).min()
    fmax = np.array(l_fmax).max()
    return fmin, fmax


def select_data(
    list_data,
    use_flag=True,
    cond_uncer=False,
    rel_max=None,
    cond_wl=False,
    wave_lim=None,
    extra_error_v2=0,
    extra_error_cp=0,
    err_scale_v2=1,
    err_scale_cp=1,
    replace_err=False,
    seuil_v2=None,
    seuil_cp=None,
):
    """
    Perform data selection base on uncertaintities (`cond_uncer`, `rel_max`),
    wavelength (`cond_wl`, `wl_bounds`) and data flagging (`use_flag`).
    Additionnal arguments are used to scale the uncertainties of the data (added
    and scaled).

    Parameters:
    -----------
    `list_data` {class/list}:
        Data or list of data from `oimalib.load()`,\n
    `use_flag` {boolean}:
        If True, use flag from the original oifits file,\n
    `cond_uncer` {boolean}:
        If True, select the best data according their relative
        uncertainties (`rel_max`),\n
    `rel_max` {float}:
        if `cond_uncer`=True, maximum sigma uncertainties allowed [%],\n
    `cond_wl` {boolean}:
        If True, apply wavelenght restriction between wl_min and wl_max,\n
    `wl_bounds` {array}:
        if `cond_wl`=True, limits of the wavelength domain [µm],\n
    `extra_error_v2`: {float}
        Additional uncertainty of the V2 (added quadraticaly),\n
    `extra_error_cp`: {float}
        Additional uncertainty of the CP (added quadraticaly),\n
    `err_scale_v2`: {float}
        Scaling factor applied on the V2 uncertainties,\n
    `err_scale_cp`: {float}
        Scaling factor applied on the CP uncertainties usualy used to
        include the non-independant CP correlation,\n
    """
    if not isinstance(list_data, list):
        list_data = [list_data]
    nfile = len(list_data)
    fmin, fmax = _find_max_freq(list_data)

    param_select = {
        "use_flag": use_flag,
        "cond_uncer": cond_uncer,
        "cond_wl": cond_wl,
        "wl_bounds": wave_lim,
    }

    new_list = deepcopy(list_data)

    list_data_sel = []
    for data_i in new_list:
        data = data_i.copy()
        nbl = data.vis2.shape[0]
        ncp = data.cp.shape[0]
        for i in range(nbl):
            new_flag_v2 = _select_data_v2(i, data, rel_max=rel_max, **param_select)
            data.flag_vis2[i] = new_flag_v2
            old_err = data.e_vis2[i]
            if replace_err:
                e_vis2 = extra_error_v2
            else:
                e_vis2 = np.sqrt(old_err**2 + extra_error_v2**2) * err_scale_v2
                if seuil_v2 is not None:
                    e_vis2[e_vis2 <= seuil_v2] = seuil_v2
            data.e_vis2[i] = e_vis2

        for j in range(ncp):
            new_flag_cp = _select_data_cp(j, data, **param_select)
            data.flag_cp[j] = new_flag_cp
            old_err = data.e_cp[j]
            if replace_err:
                e_cp = extra_error_cp
            else:
                e_cp = np.sqrt(old_err**2 + extra_error_cp**2) * err_scale_cp
                if seuil_cp is not None:
                    e_cp[e_cp <= seuil_cp] = seuil_cp
            data.e_cp[j] = e_cp

        data.info["fmax"] = fmax
        data.info["fmin"] = fmin
        list_data_sel.append(data)

    output = list_data_sel
    if nfile == 1:
        output = list_data_sel[0]
    return output


def spectral_bin_data(list_data, nbox=50, force=False, rel_err=0.01, wave_lim=None):
    """Compute spectrally binned observables using weigthed averages (based
    on squared uncertainties).

    Parameters:
    -----------

    `list_data` {list}:
        List of data class (see oimalib.load() for details),\n
    `nbox` {int}:
        Size of the box,\n
    `force` {bool}:
        If True, force the uncertainties as the relative error `rel_err`,\n
    `rel_err` {float}:
        If `force`, relative uncertainties to be used [%].

    Outputs:
    --------
    `output` {list}:
        Same as input `list_data` but spectrally binned.
    """

    if not isinstance(list_data, list | np.ndarray):
        list_data = [list_data]
    nfile = len(list_data)

    list_data_bin = []
    for data in list_data:
        (
            l_wl,
            l_vis2,
            l_e_vis2,
            l_cp,
            l_e_cp,
            l_dvis,
            l_dphi,
            l_e_dvis,
            l_e_dphi,
        ) = binning_tab(data, nbox=nbox, force=force, rel_err=rel_err)

        data_bin = data.copy()
        data_bin.wl = l_wl
        data_bin.vis2 = l_vis2
        data_bin.e_vis2 = l_e_vis2
        data_bin.flag_vis2 = np.zeros_like(l_vis2) != 0
        data_bin.cp = l_cp
        data_bin.e_cp = l_e_cp
        data_bin.flag_cp = np.zeros_like(l_cp) != 0
        data_bin.dvis = l_dvis
        data_bin.e_dvis = l_e_dvis
        data_bin.dphi = l_dphi
        data_bin.e_dphi = l_e_dphi

        freq_cp, freq_vis2, bl_cp = [], [], []
        for i in range(len(data_bin.u1)):
            B1 = np.sqrt(data_bin.u1[i] ** 2 + data_bin.v1[i] ** 2)
            B2 = np.sqrt(data_bin.u2[i] ** 2 + data_bin.v2[i] ** 2)
            B3 = np.sqrt(data_bin.u3[i] ** 2 + data_bin.v3[i] ** 2)
            Bmax = np.max([B1, B2, B3])
            bl_cp.append(Bmax)
            freq_cp.append(Bmax / l_wl / 206264.806247)  # convert to arcsec-1
        freq_cp = np.array(freq_cp)
        for i in range(len(data_bin.u)):
            freq_vis2.append(data_bin.bl[i] / l_wl / 206264.806247)  # convert to arcsec-1
        freq_vis2 = np.array(freq_vis2)

        data_bin.freq_vis2 = freq_vis2
        data_bin.freq_cp = freq_cp

        if wave_lim is not None:
            data_bin = select_data(data_bin, cond_wl=True, wave_lim=wave_lim)

        list_data_bin.append(data_bin)

    output = list_data_bin
    if nfile == 1:
        output = list_data_bin[0]
    return output


def temporal_bin_data(
    list_data,
    wave_lim=None,
    time_lim=None,
    custom_hour=None,
    verbose=False,
    corr_tellu=False,
):
    """Temporal bin between data observed during the same night. Can specify
    wavelength limits `wave_lim` (should be not used with spectrally binned data) and
    hour range `time_lim` to average the data according their observing time
    compared to the first obs (should be within an hour).

    Parameters:
    -----------
    `list_data` {list}:
        List of data class (see oimalib.load() for details),\n
    `wave_lim` {list, n=2}:
        Wavelength range to be exctracted [µm] (e.g.: around BrG line, [2.146, 2.186]),\n
    `time_lim` {list, n=2}:
        Time range to compute averaged obserbables [hour] (e.g.: [0, 1] for the first hour),\n
    `corr_tellu` {boolean}:
        If True, the tellurics correction is applied. Need pmoired `tellcorr.gravity()`
          to be runned first.

    """
    from astropy.io import fits

    blname = list_data[0].blname
    cpname = list_data[0].cpname
    if not _check_bl_same(list_data):
        return None

    oi = list_data[0].info.filename
    with fits.open(oi) as fo:
        mjd0 = fo["OI_VIS2", None].data.field("MJD")[0]

    l_hour, l_mjd = [], []
    for d in list_data:
        tmp_oi = d.info.filename
        with fits.open(tmp_oi) as fo:
            mjd = fo["OI_VIS2", None].data.field("MJD")[0]
        l_hour.append(round((mjd - mjd0) * 24, 1))
        l_mjd.append(mjd)
    mjd_master = np.mean(l_mjd)

    l_hour = np.array(l_hour)
    l_hour = l_hour[~np.isnan(l_hour)]
    if len(l_hour) == 0:
        if custom_hour is not None:
            l_hour = custom_hour
        else:
            print(
                "WARNING: Data come from aspro (without mjd), you need to add a",
                "custom_hour table to be able to merge files\n",
            )
            return None

    if verbose:
        print("Observation time of the listed dataset:\n", l_hour)
        print()
    if wave_lim is None:
        wave_lim = [0, 20]

    try:
        exclude_tel, _l_tel = _check_good_tel(list_data, verbose=verbose)
    except ValueError:
        exclude_tel = []

    good_bl, good_bl_name = _select_bl(list_data, blname, exclude_tel)

    if len(exclude_tel) != 0:
        good_cp = [i for i, x in enumerate(cpname) if exclude_tel[0] not in x]
        good_cp_name = [x for x in cpname if exclude_tel[0] not in x]
    else:
        good_cp = range(len(cpname))
        good_cp_name = cpname

    master_tel = list_data[0].tel
    n_bl = len(good_bl)
    n_data = len(list_data)

    if time_lim is None:
        n_data = len(list_data)
        file_to_be_combined = list(range(n_data))
    else:
        t0 = time_lim[0]
        t1 = time_lim[1]
        file_to_be_combined = [
            i for i in range(len(list_data)) if (l_hour[i] >= t0) & (l_hour[i] <= t1)
        ]
        n_data = len(file_to_be_combined)

    wave = list_data[0].wl * 1e6
    cond_wl = (wave >= wave_lim[0]) & (wave < wave_lim[1])
    wave = wave[cond_wl]

    n_wave = len(wave)

    if n_wave == 0:
        wave = list_data[0].wl * 1e6
        n_wave = len(wave)
        cond_wl = [True] * n_wave

    tab_dvis = np.zeros([n_data, n_bl, n_wave])
    tab_e_dvis = np.zeros([n_data, n_bl, n_wave])
    tab_dphi = np.zeros([n_data, n_bl, n_wave])
    tab_e_dphi = np.zeros([n_data, n_bl, n_wave])

    tab_vis2 = np.zeros([n_data, n_bl, n_wave])
    tab_e_vis2 = np.zeros([n_data, n_bl, n_wave])

    n_cp = len(good_cp_name)
    tab_cp = np.zeros([n_data, n_cp, n_wave])
    tab_e_cp = np.zeros([n_data, n_cp, n_wave])
    tab_u1 = np.zeros([n_data, n_cp])
    tab_u2 = np.zeros([n_data, n_cp])
    tab_v1 = np.zeros([n_data, n_cp])
    tab_v2 = np.zeros([n_data, n_cp])

    all_u, all_v = [], []
    for i, ind_file in enumerate(file_to_be_combined):
        d = list_data[ind_file].copy()
        tmp_u, tmp_v = [], []

        for k, gdcp in enumerate(good_cp):
            cp = d.cp[gdcp][cond_wl]
            e_cp = d.e_cp[gdcp][cond_wl]
            e_cp[e_cp == 0] = np.nan
            tab_cp[i, k], tab_e_cp[i, k] = cp, e_cp
            tab_u1[i, k], tab_u2[i, k] = d.u1[gdcp], d.u2[gdcp]
            tab_v1[i, k], tab_v2[i, k] = d.v1[gdcp], d.v2[gdcp]

        for j, gd in enumerate(good_bl):
            tmp_u.append(d.u[gd])
            tmp_v.append(d.v[gd])

            vis2 = d.vis2[gd][cond_wl]
            e_vis2 = d.e_vis2[gd][cond_wl]
            e_vis2[e_vis2 == 0] = np.nan

            tab_vis2[i, j], tab_e_vis2[i, j] = vis2, e_vis2

            dvis = d.dvis[gd][cond_wl]
            dphi = d.dphi[gd][cond_wl]
            # flag = d.flag_dvis[gd][cond_wl]
            # dphi[flag] = np.nan
            import oimalib

            oimalib.tools.nan_interp(dphi)
            wl = d.wl[cond_wl] * 1e6
            lbdBrg = lbdBrg = 2.165930
            inCont = (np.abs(wl - lbdBrg) < 0.1) * (np.abs(wl - lbdBrg) > 0.004)
            if False:
                if len(wl) > 10:
                    normalize_continuum(dphi, d.wl[cond_wl], inCont, degree=1, phase=True)
            e_dvis = d.e_dvis[gd][cond_wl]
            e_dphi = d.e_dphi[gd][cond_wl]
            e_dvis[e_dvis == 0] = np.nan
            e_dphi[e_dphi == 0] = np.nan
            tab_dvis[i, j], tab_dphi[i, j] = dvis, dphi
            tab_e_dvis[i, j], tab_e_dphi[i, j] = e_dvis, e_dphi
        all_u.append(tmp_u)
        all_v.append(tmp_v)
    all_u, all_v = np.array(all_u), np.array(all_v)
    master_u = np.mean(all_u, axis=0)
    master_v = np.mean(all_v, axis=0)
    B = np.sqrt(master_u**2 + master_v**2)

    master_u1 = np.mean(tab_u1, axis=0)
    master_u2 = np.mean(tab_u2, axis=0)
    master_v1 = np.mean(tab_v1, axis=0)
    master_v2 = np.mean(tab_v2, axis=0)
    master_u3 = -(master_u1 + master_u2)
    master_v3 = -(master_v1 + master_v2)

    # Compute freq, blname
    freq_cp, freq_vis2, bl_cp = [], [], []

    wave /= 1e6
    for i in range(len(master_u1)):
        B1 = np.sqrt(master_u1[i] ** 2 + master_v1[i] ** 2)
        B2 = np.sqrt(master_u2[i] ** 2 + master_v2[i] ** 2)
        B3 = np.sqrt(master_u3[i] ** 2 + master_v3[i] ** 2)

        Bmax = np.max([B1, B2, B3])
        bl_cp.append(Bmax)
        freq_cp.append(Bmax / wave / 206264.806247)  # convert to arcsec-1

    for i in range(len(master_u)):
        freq_vis2.append(B[i] / wave / 206264.806247)  # convert to arcsec-1

    freq_cp = np.array(freq_cp)
    freq_vis2 = np.array(freq_vis2)
    bl_cp = np.array(bl_cp)

    weight_dvis, weight_dphi = 1.0 / tab_e_dvis**2, 1.0 / tab_e_dphi**2
    weight_vis2 = 1.0 / tab_e_vis2**2
    weight_cp = 1.0 / tab_e_cp**2

    tab_dvis[np.isnan(tab_dvis)] = 0
    weight_dvis[np.isnan(weight_dvis)] = 1e-50
    tab_dphi[np.isnan(tab_dphi)] = 0
    weight_dphi[np.isnan(weight_dphi)] = 1e-50

    dvis_m, e_dvis_m = wtmn(tab_dvis, weights=weight_dvis, cons=False)
    dphi_m, e_dphi_m = wtmn(tab_dphi, weights=weight_dphi, cons=False)

    tab_vis2[np.isnan(tab_vis2)] = 0
    weight_vis2[np.isnan(weight_vis2)] = 1e-50

    vis2_m, e_vis2_m = wtmn(tab_vis2, weights=weight_vis2, cons=False)

    tab_cp[np.isnan(tab_cp)] = 0
    weight_cp[np.isnan(weight_cp)] = 1e-50

    cp_m, e_cp_m = wtmn(tab_cp, weights=weight_cp)

    cond_flux = [True] * len(list_data[0].flux)
    if len(exclude_tel) != 0:
        cond_flux = ~(master_tel == exclude_tel[0])

    tab_flux = np.zeros([len(file_to_be_combined), len(wave)])
    try:
        for i, ind_file in enumerate(file_to_be_combined):
            d = list_data[ind_file].copy()

            a = np.mean(d.flux[cond_flux], axis=0)[cond_wl]
            tel_tran = np.ones_like(a)
            if corr_tellu:
                filename = d.info.filename
                h = fits.open(filename)
                tel_tran = h["TELLURICS"].data["TELL_TRANS"][cond_wl]
            a /= tel_tran
            tab_flux[i] = a / a[0]
        master_flux = np.mean(tab_flux, axis=0)
        e_master_flux = np.std(tab_flux, axis=0)
        rel_err_flux = e_master_flux / master_flux
        # plt.figure()
        # plt.plot(tab_flux.T)
        # plt.plot(master_flux, "k")
        # plt.plot(master_flux - e_master_flux.mean(), "k--")
        # plt.plot(master_flux + e_master_flux.mean(), "k--")

    except IndexError:
        master_flux = np.array([[np.nan] * len(wave)] * 4)
        rel_err_flux = master_flux.copy()

    index_cp = []
    for i in good_cp:
        index_cp.append(list_data[0].index_cp[i])
    output = {
        "vis2": vis2_m,
        "e_vis2": e_vis2_m,
        "cp": cp_m,
        "e_cp": e_cp_m,
        "dvis": dvis_m,
        "e_dvis": e_dvis_m,
        "dphi": dphi_m,
        "e_dphi": e_dphi_m,
        "wl": wave,
        "blname": np.array(good_bl_name),
        "flux": master_flux,
        "rel_err_flux": rel_err_flux,
        "u": master_u,
        "v": master_v,
        "info": list_data[0].info,
        "flag_vis2": np.zeros_like(vis2_m) != 0,
        "flag_dvis": np.zeros_like(dvis_m) != 0,
        "flag_cp": np.zeros_like(cp_m) != 0,
        "cpname": np.array(good_cp_name),
        "bl": B,
        "u1": master_u1,
        "v1": master_v1,
        "u2": master_u2,
        "v2": master_v2,
        "u3": master_u3,
        "v3": master_v3,
        "freq_cp": freq_cp,
        "freq_vis2": freq_vis2,
        "teles_ref": list_data[0].teles_ref,
        "index_ref": list_data[0].index_ref,
        "index_cp": index_cp,
        "mjd": mjd_master,
    }

    return munchify(output)


def normalize_dvis_continuum(
    ibl,
    d,
    inCont,
    param_cont=None,
    force_cont=None,
    lbdBrg=2.1664,
    use_vis2=False,
):
    wl = d.wl * 1e6

    dvis = d.dvis[ibl].copy()
    e_dvis = 0.5 * d.e_dvis[ibl].copy()

    if use_vis2:
        vis2 = d.vis2[ibl].copy()
        e_dvis2 = d.e_vis2[ibl].copy()
        new_edvis = vis2**0.5 * 0.5 * (e_dvis2 / vis2)
        dvis = vis2**0.5
        e_dvis = new_edvis.copy()

    u = d.u[ibl]
    v = d.v[ibl]

    # Interpolate the nan values
    nan_interp(dvis)
    nan_interp(e_dvis)
    mean_cont, err_cont = np.mean(dvis[inCont]), np.std(dvis[inCont])
    # Normalize continuum to one
    normalize_continuum(dvis, wl, inCont)

    if param_cont is not None:
        f_model = select_model(param_cont["model"])
        complex_vis = f_model(u, v, wl * 1e-6, param_cont)[inCont]
        X = wl[inCont] - lbdBrg
        Y = abs(complex_vis)
        mean_model = Y.mean()
        cont_mod = np.polyval(np.polyfit(X, Y, 2), wl - lbdBrg)
        print(f"Obs = {mean_cont:2.3f} ± {err_cont:2.3f}, model cont = {mean_model:2.3f}")
    else:
        cont_mod = 1

    if force_cont is None:
        dvis *= cont_mod
    else:
        dvis *= force_cont
    return dvis, e_dvis


def normalize_dphi_continuum(
    ibl, d, param_cont=None, lbdBrg=2.1664, inCont=None, cond_lim=None, degree=3
):
    wl = d.wl * 1e6
    dphi = d.dphi[ibl].copy()
    e_dphi = d.e_dphi[ibl].copy()
    u = d.u[ibl]
    v = d.v[ibl]

    # Select region of the continuum (avoiding the BrG line)
    if inCont is None:
        inCont = (np.abs(wl - lbdBrg) < 0.1) * (np.abs(wl - lbdBrg) > 0.004)

    if cond_lim is None:
        cond_lim = np.array([True] * len(dphi))

    dphi = dphi[cond_lim]
    e_dphi = e_dphi[cond_lim]
    wl = wl[cond_lim]

    inCont = inCont[cond_lim]

    # Interpolate the nan values
    nan_interp(dphi)
    nan_interp(e_dphi)
    # Normalize continuum to one

    normalize_continuum(dphi, wl, inCont, phase=True, degree=degree)

    if param_cont is not None:
        f_model = select_model(param_cont["model"])
        complex_vis = f_model(u, v, wl * 1e-6, param_cont)[inCont]
        X = wl[inCont] - lbdBrg
        Y = np.angle(complex_vis, deg=True)
        cont_mod = np.polyval(np.polyfit(X, Y, 2), wl - lbdBrg)
    else:
        cont_mod = 0

    dphi += cont_mod
    return dphi, e_dphi


# def normalize_dphi_continuum(ibl, d, param_cont=None, lbdBrg=2.1664):
#     wl = d.wl * 1e6
#     dphi = d.dphi[ibl].copy()
#     e_dphi = d.e_dphi[ibl].copy()
#     u = d.u[ibl]
#     v = d.v[ibl]

#     # Select region of the continuum (avoiding the BrG line)
#     inCont = (np.abs(wl - lbdBrg) < 0.1) * (np.abs(wl - lbdBrg) > 0.004)

#     # Interpolate the nan values
#     nan_interp(dphi)
#     nan_interp(e_dphi)
#     # Normalize continuum to one

#     normalize_continuum(dphi, wl, inCont, phase=True, degree=3)

#     if param_cont is not None:
#         f_model = select_model(param_cont["model"])
#         complex_vis = f_model(u, v, wl * 1e-6, param_cont)[inCont]
#         X = wl[inCont] - lbdBrg
#         Y = np.angle(complex_vis, deg=True)
#         cont_mod = np.polyval(np.polyfit(X, Y, 2), wl - lbdBrg)
#     else:
#         cont_mod = 0

#     dphi += cont_mod
#     return dphi, e_dphi


# def compute_pure_line_cvis_v2(
#     ibl,
#     d,
#     flc,
#     d_ft=None,
#     param_dvis=None,
#     param_dphi=None,
#     lbdBrg=2.1661,
#     wBrg=0.0005,
#     use_mod=True,
#     self_norm=False,
#     force_zero_dphi=None,
#     force_zero_dvis=None,
#     force_simple_phase=None,
#     force_simple_vis=None,
#     use_cont_err=True,
#     verbose=False,
#     ref_v2=False,
#     r_brg=2,
#     r_brg2=1,
#     display=False,
# ):
#     if force_zero_dphi is None:
#         force_zero_dphi = []
#     if force_zero_dvis is None:
#         force_zero_dvis = []
#     if force_simple_phase is None:
#         force_simple_phase = []
#     if force_simple_vis is None:
#         force_simple_vis = []

#     # Extract the parameters from flc dict (from compute_flc_spectra())

#     e_flux = flc.get("e_flux", 0.01)  # error on the flux

#     if use_mod:
#         F_lc = flc["F_lc"]  # fit on the normalized spectrum

#     #  Compute region in and outside the line
#     wl = d.wl * 1e6

#     # inCont2 = (np.abs(wl - lbdBrg) < 0.1) * (np.abs(wl - lbdBrg) > r_brg * wBrg)
#     inLine = np.abs(wl - lbdBrg) <= r_brg2 * wBrg
#     # inLine2 = np.abs(wl - lbdBrg) < r_brg2 * wBrg
#     # inCont = ~inLine2

#     inCont = (np.abs(wl - lbdBrg) < 0.1) * (np.abs(wl - lbdBrg) > r_brg * wBrg)
#     # Take the continuum from the FT camera (for GRAVITY)
#     if self_norm:
#         if ref_v2:
#             cvis_in_cont = d.vis2[ibl][inCont].copy() ** 0.5
#         else:
#             cvis_in_cont = d.dvis[ibl][inCont].copy()
#         nan_interp(cvis_in_cont)
#         cont_ft = cvis_in_cont.mean()
#     else:
#         cont_ft = d_ft.vis2[ibl] ** 0.5

#     # cont_ft *= 1.083
#     dvis, e_dvis = normalize_dvis_continuum(
#         ibl, d, inCont=inCont, force_cont=cont_ft, lbdBrg=lbdBrg
#     )
#     dphi, e_dphi = normalize_dphi_continuum_v2(ibl, d, lbdBrg=lbdBrg, inCont=inCont)

#     if use_cont_err:
#         e_dphi = np.ones(len(e_dphi)) * np.std(dphi[inCont])
#         e_dvis = np.ones(len(e_dvis)) * np.std(dvis[inCont])

#     # Fit differential obs
#     double_phase = True
#     double_vis = True
#     if ibl in force_simple_phase:
#         double_phase = False
#     if ibl in force_simple_vis:
#         double_vis = False

#     mod_dvis, fit_dvis = perform_fit_dvis(
#         wl,
#         dvis,
#         e_dvis,
#         inCont=None,
#         param=param_dvis,
#         double=double_vis,
#         display=display,
#     )

#     mod_dphi, fit_dphi = perform_fit_dphi(
#         wl, dphi, e_dphi, param_dphi, double=double_phase, display=display,
#     )

#     if ibl in force_zero_dphi:
#         mod_dphi = np.zeros_like(mod_dphi)
#     if ibl in force_zero_dvis:
#         mod_dvis = np.ones_like(mod_dvis) * fit_dvis["best"]["C"]

#     if verbose:
#         print("Results fit differentials:")
#         print("--------------------------")
#         print(
#             "chi2: vis={:2.2f}, phi={:2.2f}".format(fit_dvis["chi2"], fit_dphi["chi2"])
#         )

#     # Compute pure line vis and phi
#     if use_mod:
#         V_tot = mod_dvis[inLine]
#         dphi_inline = np.deg2rad(mod_dphi[inLine])
#         F_tot = F_lc[inLine]
#     else:
#         V_tot = dvis[inLine]
#         dphi_inline = np.deg2rad(dphi[inLine])
#         try:
#             F_tot = d.flux[inLine]
#         except IndexError:
#             F_tot = flc["flux"][inLine]

#     e_dvis_inline = np.mean(e_dvis[inLine])
#     e_dphi_inline = np.mean(np.deg2rad(e_dphi[inLine]))

#     from uncertainties import ufloat, unumpy

#     if self_norm:
#         V_cont = dvis[inCont].mean()
#         e_V_cont = dvis[inCont].std()
#     else:
#         V_cont = d_ft.vis2[ibl] ** 0.5
#         e_V_cont = d_ft.e_dvis[ibl]

#     n = len(V_tot)
#     u_V_tot = np.array([ufloat(V_tot[i], e_dvis_inline) for i in range(n)])
#     u_F_tot = np.array([ufloat(F_tot[i], e_flux) for i in range(n)])
#     u_V_cont = np.array([ufloat(V_cont, e_V_cont) for i in range(n)])
#     u_dphi_inline = np.array([ufloat(dphi_inline[i], e_dphi_inline) for i in range(n)])

#     F_cont = 1
#     u_F_line = u_F_tot - F_cont

#     # u_nominator = (
#     #     abs(u_V_tot * u_F_tot) ** 2
#     #     + abs(u_V_cont * F_cont) ** 2
#     #     - (2 * u_V_tot * u_F_tot * u_V_cont * F_cont * unumpy.cos(u_dphi_inline))
#     # )
#     # u_nominator = (
#     #     abs(u_V_tot * (u_F_tot / (u_F_tot - 1))) ** 2
#     #     + abs(u_V_cont * (1 / (u_F_tot - 1))) ** 2
#     #     - (
#     #         2
#     #         * (u_F_tot / (u_F_tot - 1) ** 2)
#     #         * u_V_tot
#     #         * u_V_cont
#     #         * unumpy.cos(u_dphi_inline)
#     #     )
#     # )
#     # u_dvis_pure = u_nominator ** 0.5 / u_F_line

#     u_dvis_pure = (u_F_tot * u_V_tot - u_V_cont) / (u_F_tot - 1)

#     dvis_pure = unumpy.nominal_values(u_dvis_pure)
#     e_dvis_pure = unumpy.std_devs(u_dvis_pure)

#     wl_inline = wl[inLine]

#     X = (u_F_tot * u_V_tot * unumpy.sin(u_dphi_inline)) / (u_F_line * u_dvis_pure)
#     all_err = []
#     for i in range(len(X)):
#         all_err.append(X[i].std_dev)
#     aver_err = np.mean(all_err)

#     for i in range(len(X)):
#         if X[i].std_dev >= 1:
#             a = ufloat(X[i].nominal_value, aver_err)
#             X[i] = a

#     u_dphi_pure = 180 * (X) / np.pi

#     dphi_pure = unumpy.nominal_values(u_dphi_pure)
#     e_dphi_pure = unumpy.std_devs(u_dphi_pure)
#     wl_inline = wl[inLine]

#     # Compute photocenter shifts
#     ucoord = d.u[ibl]
#     vcoord = d.v[ibl]
#     bl_length, bl_pa = cart2pol(ucoord, vcoord)

#     n_wl_inline = len(wl_inline)
#     pco = np.zeros(n_wl_inline)
#     e_pco = np.zeros(n_wl_inline)
#     for iwl in range(n_wl_inline):
#         pi = rad2mas(
#             (-np.deg2rad(dphi_pure[iwl]) / (2 * np.pi))
#             * ((wl_inline[iwl] * 1e-6) / (bl_length))
#         )
#         e_pi = rad2mas(
#             (-np.deg2rad(e_dphi_pure[iwl]) / (2 * np.pi))
#             * ((wl_inline[iwl] * 1e-6) / (bl_length))
#         )
#         pco[iwl] = pi
#         e_pco[iwl] = abs(e_pi)

#     # Uncertainties computation (to be done)
#     output = {
#         "dvis": dvis,
#         "e_dvis": e_dvis,
#         "dphi": dphi,
#         "e_dphi": e_dphi,
#         "pco": pco,
#         "e_pco": e_pco,
#         "bl_pa": bl_pa,
#         "bl_length": bl_length,
#         "dvis_pure": dvis_pure,
#         "e_dvis_pure": e_dvis_pure,
#         "dphi_pure": dphi_pure,
#         "e_dphi_pure": e_dphi_pure,
#         "wl_line": wl_inline,
#         "inLine": inLine,
#         "mod_dvis": mod_dvis,
#         "e_mod_dvis": e_dvis_inline,
#         "mod_dphi": mod_dphi,
#         "e_mod_dphi": np.rad2deg(e_dphi_inline),
#         "e_flux": e_flux,
#         "blname": d.blname[ibl],
#         "param_dphi": param_dphi,
#         "param_dvis": param_dvis,
#         "F_tot": F_tot,
#     }
#     return munchify(output)


def compute_pure_line_cvis(
    ibl,
    d,
    flc,
    *,
    d_ft=None,
    lbdBrg=2.1661,
    wBrg=0.0005,
    use_mod=True,
    self_norm=False,
    force_zero_dphi=None,
    force_zero_dvis=None,
    force_simple_phase=None,
    force_simple_vis=None,
    use_cont_err=True,
    verbose=False,
    A=0.5,
    ref_v2=False,
    param_dphi=None,
    param_dvis=None,
):
    if force_zero_dphi is None:
        force_zero_dphi = []
    if force_zero_dvis is None:
        force_zero_dvis = []
    if force_simple_phase is None:
        force_simple_phase = []
    if force_simple_vis is None:
        force_simple_vis = []

    # Extract the parameters from flc dict (from compute_flc_spectra())
    inLine = flc["inLine"]
    e_flux = flc["e_flux"]  # error on the flux
    F_lc = flc["F_lc"]  # fit on the normalized spectrum

    flc["widthline"]

    #  Compute region in and outside the line
    wl = d.wl * 1e6

    # try:
    inCont = flc["inCont"]
    # inCont = (np.abs(wl - lbdBrg) < 0.1) * (np.abs(wl - lbdBrg) > 0.004)
    # inLine = np.abs(wl - lbdBrg) <= 2.355 * wBrg

    # Take the continuum from the FT camera (for GRAVITY)
    if self_norm:
        cvis_in_cont = d.vis2[ibl][inCont].copy() ** 0.5 if ref_v2 else d.dvis[ibl][inCont].copy()
        nan_interp(cvis_in_cont)
        cont_ft = cvis_in_cont.mean()
    else:
        # cont_ft = d_ft.dvis[ibl]
        cont_ft = d_ft.vis2[ibl] ** 0.5

    dvis, e_dvis = normalize_dvis_continuum(
        ibl, d, inCont=inCont, force_cont=cont_ft, lbdBrg=lbdBrg
    )
    dphi, e_dphi = normalize_dphi_continuum(ibl, d, lbdBrg=lbdBrg)

    if use_cont_err:
        e_dphi = np.ones(len(e_dphi)) * np.std(dphi[inCont])
        e_dvis = np.ones(len(e_dvis)) * np.std(dvis[inCont])

    # Fit differential obs
    # param_dvis = {"A": 0.02, "B": 0.88, "sigma": 0.0005, "pos": lbdBrg}

    # param_dvis = {
    #     "A": 0.01,
    #     "B": 0.01,
    #     "C": 0.9,
    #     "sigmaA": 0.0005 / 2.355,
    #     "sigmaB": 0.0005 / 2.355,
    #     "pos": lbdBrg,
    #     "dp": wBrg * 2,
    # }

    # param_dphi = {
    #     "A": -A,
    #     "B": A,
    #     "sigmaA": 0.0005 / 2.355,
    #     "sigmaB": 0.0005 / 2.355,
    #     "pos": lbdBrg,
    #     "dp": wBrg * 2,
    # }

    double_phase = True
    double_vis = True
    if ibl in force_simple_phase:
        double_phase = False
    if ibl in force_simple_vis:
        double_vis = False

    try:
        mod_dvis, fit_dvis = perform_fit_dvis(
            wl, dvis, e_dvis, param_dvis, inCont=inCont, double=double_vis
        )
    except UnboundLocalError:
        mod_dvis = []

    mod_dphi, fit_dphi = perform_fit_dphi(wl, dphi, e_dphi, param_dphi, double=double_phase)

    if ibl in force_zero_dphi:
        mod_dphi = np.zeros_like(mod_dphi)
    if ibl in force_zero_dvis:
        mod_dvis = np.ones_like(mod_dvis) * fit_dvis["best"]["C"]

    if verbose:
        print("Results fit differentials:")
        print("--------------------------")
        print("chi2: vis={:2.2f}, phi={:2.2f}".format(fit_dvis["chi2"], fit_dphi["chi2"]))

    # Compute pure line vis and phi
    if use_mod:
        V_tot = mod_dvis[inLine]
        dphi_inline = np.deg2rad(mod_dphi[inLine])
        F_tot = F_lc[inLine]
    else:
        V_tot = dvis[inLine]
        dphi_inline = np.deg2rad(dphi[inLine])
        try:
            F_tot = d.flux[inLine]
        except IndexError:
            F_tot = flc["flux"][inLine]
        # F_tot = d.flux[inLine]

    e_dvis_inline = np.mean(e_dvis[inLine])
    e_dphi_inline = np.mean(np.deg2rad(e_dphi[inLine]))

    from uncertainties import ufloat, unumpy

    if self_norm:
        V_cont = dvis[inCont].mean()
        e_V_cont = dvis[inCont].std()  # 0.001 * V_cont
    else:
        V_cont = d_ft.dvis[ibl]
        e_V_cont = d_ft.e_dvis[ibl]

    # print("Continuum amp = ", ufloat(V_cont, e_V_cont))

    n = len(V_tot)
    u_V_tot = np.array([ufloat(V_tot[i], e_dvis_inline) for i in range(n)])
    u_F_tot = np.array([ufloat(F_tot[i], e_flux) for i in range(n)])
    u_V_cont = np.array([ufloat(V_cont, e_V_cont) for i in range(n)])
    u_dphi_inline = np.array(
        [ufloat(dphi_inline[i], np.deg2rad(e_dphi[inLine][i])) for i in range(n)]
    )

    F_cont = 1
    u_F_line = u_F_tot - F_cont

    u_nominator = (
        abs(u_V_tot * u_F_tot) ** 2
        + abs(u_V_cont * F_cont) ** 2
        - (2 * u_V_tot * u_F_tot * u_V_cont * F_cont * unumpy.cos(u_dphi_inline))
    )

    u_dvis_pure = u_nominator**0.5 / u_F_line
    # u_dvis_pure = (u_F_tot * u_V_tot - u_V_cont) / (u_F_tot - 1)

    # for i in range(len(u_V_tot)):
    #     print(u_V_tot[i], u_dvis_pure[i], u_F_tot[i])

    dvis_pure = unumpy.nominal_values(u_dvis_pure)
    e_dvis_pure = unumpy.std_devs(u_dvis_pure)

    u_dphi_pure = (
        180
        * (
            unumpy.arcsin(
                (u_F_tot * u_V_tot * unumpy.sin(u_dphi_inline)) / (u_F_line * u_dvis_pure)
            )
        )
        / np.pi
    )

    dphi_pure = unumpy.nominal_values(u_dphi_pure)
    e_dphi_pure = unumpy.std_devs(u_dphi_pure)
    wl_inline = wl[inLine]

    # Compute photocenter shifts
    ucoord = d.u[ibl]
    vcoord = d.v[ibl]
    bl_length, bl_pa = cart2pol(ucoord, vcoord)

    n_wl_inline = len(wl_inline)
    pco = np.zeros(n_wl_inline)
    e_pco = np.zeros(n_wl_inline)
    for iwl in range(n_wl_inline):
        pi = rad2mas(
            (-np.deg2rad(dphi_pure[iwl]) / (2 * np.pi)) * ((wl_inline[iwl] * 1e-6) / (bl_length))
        )
        e_pi = rad2mas(
            (-np.deg2rad(e_dphi_pure[iwl]) / (2 * np.pi)) * ((wl_inline[iwl] * 1e-6) / (bl_length))
        )
        pco[iwl] = pi
        e_pco[iwl] = abs(e_pi)

    # Uncertainties computation (to be done)
    output = {
        "dvis": dvis,
        "e_dvis": e_dvis,
        "dphi": dphi,
        "e_dphi": e_dphi,
        "pco": pco,
        "e_pco": e_pco,
        "bl_pa": bl_pa,
        "bl_length": bl_length,
        "dvis_pure": dvis_pure,
        "e_dvis_pure": e_dvis_pure,
        "dphi_pure": dphi_pure,
        "e_dphi_pure": e_dphi_pure,
        "wl_line": wl_inline,
        "mod_dvis": mod_dvis,
        "e_mod_dvis": e_dvis_inline,
        "mod_dphi": mod_dphi,
        "e_mod_dphi": np.rad2deg(e_dphi_inline),
        "e_flux": e_flux,
        "blname": d.blname[ibl],
        "param_dphi": param_dphi,
        "param_dvis": param_dvis,
        "inLine": inLine,
        "F_tot": F_tot,
    }
    return munchify(output)


# def compute_pure_line_cvis_v2(
#     ibl,
#     d,
#     lcr,
#     *,
#     inLine=None,
#     inCont=None,
#     e_flux=None,
#     d_ft=None,
#     lbdBrg=2.1661,
#     wBrg=0.0005,
#     use_mod=True,
#     self_norm=False,
#     force_zero_dphi=None,
#     force_zero_dvis=None,
#     force_simple_phase=None,
#     force_simple_vis=None,
#     use_cont_err=True,
#     verbose=False,
#     A=0.5,
#     ref_v2=False,
#     param_dphi=None,
#     param_dvis=None,
# ):
#     if force_zero_dphi is None:
#         force_zero_dphi = []
#     if force_zero_dvis is None:
#         force_zero_dvis = []
#     if force_simple_phase is None:
#         force_simple_phase = []
#     if force_simple_vis is None:
#         force_simple_vis = []

#     # Extract the parameters from flc dict (from compute_flc_spectra())
#     # inLine = flc["inLine"]
#     # e_flux = flc["e_flux"]  # error on the flux
#     F_lc = lcr  # fit on the normalized spectrum

#     # #  Compute region in and outside the line
#     # wl = d.wl * 1e6
#     # inCont = (np.abs(wl - lbdBrg) < 0.1) * (np.abs(wl - lbdBrg) > 0.004)
#     # # inLine = np.abs(wl - lbdBrg) <= 2.355 * wBrg

#     # Take the continuum from the FT camera (for GRAVITY)
#     if self_norm:
#         if ref_v2:
#             cvis_in_cont = d.vis2[ibl][inCont].copy() ** 0.5
#         else:
#             cvis_in_cont = d.dvis[ibl][inCont].copy()
#         nan_interp(cvis_in_cont)
#         cont_ft = cvis_in_cont.mean()
#     else:
#         # cont_ft = d_ft.dvis[ibl]
#         cont_ft = d_ft.vis2[ibl] ** 0.5

#     dvis, e_dvis = normalize_dvis_continuum(
#         ibl, d, inCont=inCont, force_cont=cont_ft, lbdBrg=lbdBrg
#     )
#     dphi, e_dphi = normalize_dphi_continuum(ibl, d, lbdBrg=lbdBrg)

#     if use_cont_err:
#         e_dphi = np.ones(len(e_dphi)) * np.std(dphi[inCont])
#         e_dvis = np.ones(len(e_dvis)) * np.std(dvis[inCont])

#     # Fit differential obs
#     param_dvis = {"A": 0.02, "B": 0.88, "sigma": 0.0005, "pos": lbdBrg}

#     param_dvis = {
#         "A": 0.01,
#         "B": 0.01,
#         "C": 0.9,
#         "sigmaA": 0.0005 / 2.355,
#         "sigmaB": 0.0005 / 2.355,
#         "pos": lbdBrg,
#         "dp": wBrg * 2,
#     }

#     param_dphi = {
#         "A": -A,
#         "B": A,
#         "sigmaA": 0.0005 / 2.355,
#         "sigmaB": 0.0005 / 2.355,
#         "pos": lbdBrg,
#         "dp": wBrg * 2,
#     }

#     double_phase = True
#     double_vis = True
#     if ibl in force_simple_phase:
#         double_phase = False
#     if ibl in force_simple_vis:
#         double_vis = False

#     try:
#         mod_dvis, fit_dvis = perform_fit_dvis(
#             wl, dvis, e_dvis, param_dvis, double=double_vis
#         )
#     except UnboundLocalError:
#         mod_dvis = []

#     mod_dphi, fit_dphi = perform_fit_dphi(
#         wl, dphi, e_dphi, param_dphi, double=double_phase
#     )

# if ibl in force_zero_dphi:
#     mod_dphi = np.zeros_like(mod_dphi)
# if ibl in force_zero_dvis:
#     mod_dvis = np.ones_like(mod_dvis) * fit_dvis["best"]["C"]

# if verbose:
#     print("Results fit differentials:")
#     print("--------------------------")
#     print(
#         "chi2: vis={:2.2f}, phi={:2.2f}".format(fit_dvis["chi2"], fit_dphi["chi2"])
#     )

# # Compute pure line vis and phi
# if use_mod:
#     V_tot = mod_dvis[inLine]
#     dphi_inline = np.deg2rad(mod_dphi[inLine])
#     F_tot = F_lc[inLine]
# else:
#     V_tot = dvis[inLine]
#     dphi_inline = np.deg2rad(dphi[inLine])
#     try:
#         F_tot = d.flux[inLine]
#     except IndexError:
#         F_tot = flc["flux"][inLine]
#     # F_tot = d.flux[inLine]

# e_dvis_inline = np.mean(e_dvis[inLine])
# e_dphi_inline = np.mean(np.deg2rad(e_dphi[inLine]))

# from uncertainties import ufloat, unumpy

# if self_norm:
#     V_cont = dvis[inCont].mean()
#     e_V_cont = dvis[inCont].std()  # 0.001 * V_cont
# else:
#     V_cont = d_ft.dvis[ibl]
#     e_V_cont = d_ft.e_dvis[ibl]

# # print("Continuum amp = ", ufloat(V_cont, e_V_cont))

# n = len(V_tot)
# u_V_tot = np.array([ufloat(V_tot[i], e_dvis_inline) for i in range(n)])
# u_F_tot = np.array([ufloat(F_tot[i], e_flux) for i in range(n)])
# u_V_cont = np.array([ufloat(V_cont, e_V_cont) for i in range(n)])
# u_dphi_inline = np.array(
#     [ufloat(dphi_inline[i], np.deg2rad(e_dphi[inLine][i])) for i in range(n)]
# )

# F_cont = 1
# u_F_line = u_F_tot - F_cont

# u_nominator = (
#     abs(u_V_tot * u_F_tot) ** 2
#     + abs(u_V_cont * F_cont) ** 2
#     - (2 * u_V_tot * u_F_tot * u_V_cont * F_cont * unumpy.cos(u_dphi_inline))
# )

# u_dvis_pure = u_nominator ** 0.5 / u_F_line
# # u_dvis_pure = (u_F_tot * u_V_tot - u_V_cont) / (u_F_tot - 1)

# # for i in range(len(u_V_tot)):
# #     print(u_V_tot[i], u_dvis_pure[i], u_F_tot[i])

# dvis_pure = unumpy.nominal_values(u_dvis_pure)
# e_dvis_pure = unumpy.std_devs(u_dvis_pure)

# u_dphi_pure = (
#     180
#     * (
#         unumpy.arcsin(
#             (u_F_tot * u_V_tot * unumpy.sin(u_dphi_inline))
#             / (u_F_line * u_dvis_pure)
#         )
#     )
#     / np.pi
# )

# dphi_pure = unumpy.nominal_values(u_dphi_pure)
# e_dphi_pure = unumpy.std_devs(u_dphi_pure)
# wl_inline = wl[inLine]

# # Compute photocenter shifts
# ucoord = d.u[ibl]
# vcoord = d.v[ibl]
# bl_length, bl_pa = cart2pol(ucoord, vcoord)

# n_wl_inline = len(wl_inline)
# pco = np.zeros(n_wl_inline)
# e_pco = np.zeros(n_wl_inline)
# for iwl in range(n_wl_inline):
#     pi = rad2mas(
#         (-np.deg2rad(dphi_pure[iwl]) / (2 * np.pi))
#         * ((wl_inline[iwl] * 1e-6) / (bl_length))
#     )
#     e_pi = rad2mas(
#         (-np.deg2rad(e_dphi_pure[iwl]) / (2 * np.pi))
#         * ((wl_inline[iwl] * 1e-6) / (bl_length))
#     )
#     pco[iwl] = pi
#     e_pco[iwl] = abs(e_pi)

# # Uncertainties computation (to be done)
# output = {
#     "dvis": dvis,
#     "e_dvis": e_dvis,
#     "dphi": dphi,
#     "e_dphi": e_dphi,
#     "pco": pco,
#     "e_pco": e_pco,
#     "bl_pa": bl_pa,
#     "bl_length": bl_length,
#     "dvis_pure": dvis_pure,
#     "e_dvis_pure": e_dvis_pure,
#     "dphi_pure": dphi_pure,
#     "e_dphi_pure": e_dphi_pure,
#     "wl_line": wl_inline,
#     "mod_dvis": mod_dvis,
#     "e_mod_dvis": e_dvis_inline,
#     "mod_dphi": mod_dphi,
#     "e_mod_dphi": np.rad2deg(e_dphi_inline),
#     "e_flux": e_flux,
#     "blname": d.blname[ibl],
#     "param_dphi": param_dphi,
#     "param_dvis": param_dvis,
#     "inLine": inLine,
#     "F_tot": F_tot,
# }
# return munchify(output)


def compute_pco(data, data_cont, **args):
    """Compute the photocenter offset for the differente baselines and spectral
    channels available in `data`. The continuum is normalized using the
    observables from `data_cont` (usually the FT data from GRAVITY).

    Parameters:
    -----------
    `data` {dict}: class-like data (from oimalib.load),\n
    `data_cont` {dict}: class-like data used as continuum (from oimalib.load),\n
    `flc` {dict}: Fitted flux to extract the line to continuum ratio (from
    compute_flc_spectra()),\n
    `force_zero_dphi` {list}: index of baseline to force the phase to zero,\n
    `force_simple` {list}: index of baseline to force the phase as simple
    gaussian model,\n
    `use_mod` {bool}: If True, the fitted model is used to compute the pure line
    visibility and phase.\n
    Results:
    --------
    `output` {dict}: Computed pc offset with uncertainties ('pco'), baseline
    lengths ('bl_length') and baseline orientations ('bl_pa').
    """
    # Extract the parameters from flc dict (from compute_flc_spectra())
    flc = args["flc"]
    inLine = flc["inLine"]

    nbl = len(data.vis2)
    nwl = len(data.wl[inLine])
    pco = []
    bl_pa, bl_length = np.zeros(nbl), np.zeros(nbl)

    print(args.keys())
    l_pure = []
    for i in range(nbl):
        tmp = compute_pure_line_cvis(i, data, d_ft=data_cont, **args)
        l_pure.append(tmp)
        pco_wl = []
        for j in range(nwl):
            a = tmp.pco[j].astype(float)
            b = tmp.e_pco[j].astype(float)
            pco_wl.append(ufloat(a, b))
        pco.append(pco_wl)
        bl_pa[i] = tmp.bl_pa
        bl_length[i] = tmp.bl_length
    pco = np.array(pco)

    try:
        wl_line = flc["fit"]["best"]["lbdBrg"]
    except KeyError:
        wl_line = flc["fit"]["best"]["p1"]

    output = {
        "pco": pco,
        "bl_length": bl_length,
        "bl_pa": bl_pa,
        "wl": flc["wl"][inLine],
        "nbl": len(data.vis2),
        "blname": data.blname,
        "wl_line": wl_line,
        "nwl_inline": len(flc["wl"][inLine]),
        "pure": np.array(l_pure),
        "restframe": flc["restframe"],
    }
    return output


def pcs_from_aspro(d, lbdBrg=2.1661, wBrg=0.0005, ratio=2.5):
    from oimalib.plotting import err_pts_style

    """d is class-like dict (from oimalib.load) containing data from jmmc
    software (with only ONE u-v pt). Compute the photocenter shift from the
    already normalized phase visibility."""
    wl = d.wl * 1e6
    inLine = np.abs(wl - lbdBrg) <= ratio * wBrg
    dphi_pure = d.dphi[:, inLine]
    wl_inline = wl[inLine]

    nbl = len(d.u)
    bl_length, bl_pa = [], []
    for i in range(nbl):
        ucoord = d.u[i]
        vcoord = d.v[i]
        tmp = cart2pol(ucoord, vcoord)
        bl_length.append(tmp[0])
        bl_pa.append(tmp[1])
    bl_pa = np.array(bl_pa)
    bl_length = np.array(bl_length)

    plt.figure(figsize=(9, 8))
    l_u_x, l_u_y, l_fit = [], [], []
    for j in range(len(wl_inline)):
        pi = rad2mas(
            (-np.deg2rad(dphi_pure[:, j]) / (2 * np.pi)) * ((wl_inline[j] * 1e-6) / (bl_length[:]))
        )
        y_pc = np.concatenate([pi, -pi]) * 1000.0
        x_pc = np.concatenate([bl_pa, bl_pa - 180])
        e_pc = np.ones_like(y_pc) * y_pc.max() * 0.2

        chi2_tmp = 1e50
        for o in np.arange(0, 360, 45):
            param = {"p": 0.05, "offset": o}
            fit_tmp = leastsqFit(
                model_pcshift,
                x_pc,
                param,
                y_pc,
                err=e_pc,
                verbose=False,
                normalizedUncer=True,
            )
            chi2 = fit_tmp["chi2"]
            if chi2 <= chi2_tmp:
                fit_pc = fit_tmp
                chi2_tmp = chi2
        l_fit.append(fit_pc)

        x_model = np.linspace(0, 360, 100)
        plt.subplot(3, 3, j + 1)
        plt.title(f"λ = {wl_inline[j]:2.4f} µm")
        plt.errorbar(x_pc, y_pc, yerr=e_pc, **err_pts_style)
        plt.plot(x_model, model_pcshift(x_model, fit_pc["best"]), "-", lw=1)
        plt.ylim(-60, 60)
        u_pc = ufloat(fit_pc["best"]["p"], fit_pc["uncer"]["p"])
        u_pa = ufloat(fit_pc["best"]["offset"], fit_pc["uncer"]["offset"])
        east, north = compute_oriented_shift(u_pa, u_pc)
        l_u_x.append(east)
        l_u_y.append(north)
    plt.tight_layout()
    pcs_east = np.array(l_u_x)
    pcs_north = np.array(l_u_y)

    pcs = {
        "east": pcs_east,
        "north": pcs_north,
        "fit_param": l_fit,
        "wl": wl_inline,
        "wl_line": lbdBrg,
        "inLine": inLine,
        "restframe": lbdBrg,
    }
    return pcs


def extract_condition(list_data, tau_lim=20, display=False):
    from astropy.io import fits

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

    if display:
        from oimalib.plotting import plot_condition

        plot_condition(l_mjd, l_seeing, l_tau, tau_lim=tau_lim)

    return munchify(output)
