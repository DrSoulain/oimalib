import contextlib
import multiprocessing
import sys

import corner
import emcee
import numpy as np
from matplotlib import pyplot as plt
from munch import munchify as dict2class
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from termcolor import cprint
from tqdm import tqdm
from uncertainties import ufloat

from . import complex_models
from .complex_models import model_acc_mag
from .fit.dpfit import leastsqFit
from .tools import compute_oriented_shift, mas2rad, normalize_continuum, round_sci_digit

if sys.platform == "darwin":
    multiprocessing.set_start_method("fork", force=True)

err_pts_style = {
    "linestyle": "None",
    "capsize": 1,
    "ecolor": "#364f6b",
    "mec": "#364f6b",
    "marker": ".",
    "elinewidth": 0.5,
    "alpha": 1,
    "ms": 14,
}


err_pts_style_f = {
    "linestyle": "None",
    "capsize": 1.5,
    "ecolor": "#364f6b",
    "mec": "#364f6b",
    "marker": ".",
    "elinewidth": 1,
    "alpha": 1,
}


def select_model(name):
    """Select a simple model computed in the Fourier space
    (check model.py)"""
    if name == "disk":
        model = complex_models.visUniformDisk
    elif name == "binary":
        model = complex_models.visBinary
    elif name == "binary_res":
        model = complex_models.visBinary_res
    elif name == "edisk":
        model = complex_models.visEllipticalUniformDisk
    elif name == "debrisDisk":
        model = complex_models.visDebrisDisk
    elif name == "multipleRing":
        model = complex_models.visMultipleRing
    elif name == "clumpyDebrisDisk":
        model = complex_models.visClumpDebrisDisk
    elif name == "gdisk":
        model = complex_models.visGaussianDisk
    elif name == "egdisk":
        model = complex_models.visEllipticalGaussianDisk
    elif name == "egdisk2":
        model = complex_models.visEllipticalGaussianDisk2
    elif name == "ering":
        model = complex_models.visThickEllipticalRing
    elif name == "lor":
        model = complex_models.visLorentzDisk
    elif name == "yso":
        model = complex_models.visYSO
    elif name == "lazareff":
        model = complex_models.visLazareff
    elif name == "lazareff_halo":
        model = complex_models.visLazareff_halo
    elif name == "lazareff_clump":
        model = complex_models.visLazareff_clump
    elif name == "ellipsoid":
        model = complex_models.visEllipsoid
    elif name == "cont":
        model = complex_models.visCont
    elif name == "bohn":
        model = complex_models.get_full_visibility
    elif name == "yso_line":
        model = complex_models.visLazareff_line
    elif name == "pwhl":
        model = complex_models.visPwhl
    elif name == "orion":
        model = complex_models.visOrionDisk
    else:
        model = None
    return model


def check_params_model(param):
    """Check if the user parameters are compatible
    with the model."""
    isValid = True
    log = ""

    prior = param.get("prior", {})
    if param["model"] == "edisk":
        elong = np.cos(np.deg2rad(param["incl"]))
        majorAxis = mas2rad(param["majorAxis"])
        angle = np.deg2rad(param["pa"])
        if (elong > 1) or (angle < 0) or (angle > np.pi) or (majorAxis < 0):
            log = "# elong > 1,\n# minorAxis > 0 mas,\n# 0 < pa < 180 deg."
            isValid = False
    elif param["model"] == "binary":
        dm = param["dm"]
        if dm < 0:
            isValid = False
    elif param["model"] == "egdisk" or param["model"] == "edisk":
        majorAxis = param["majorAxis"]
        incl = param["incl"]
        pa = param["pa"]

        c1 = (incl > 90.0) | (incl < 0)
        c2 = (pa < 0) | (pa > 180)
        if c1 | c2:
            isValid = False
    elif param["model"] == "ering":
        majorAxis = param["majorAxis"]
        incl = param["incl"]
        pa = param["pa"]
        kr = param["kr"]

        c1 = (incl > 90.0) | (incl < 0)
        c2 = (pa < -180) | (pa > 180)
        c3 = (kr < -2) | (kr > 0.0)
        if c1 | c2 | c3:
            isValid = False
    elif param["model"] == "lazareff":
        la = param["la"]
        lk = param["lk"]
        cj = param["cj"]
        sj = param["sj"]
        kc = param["kc"]

        flor = param["flor"]
        incl = param["incl"]
        pa = param["pa"]

        fs = param["fs"]
        fc = param["fc"]
        fh = 1 - fs - fc
        cond_flux = (fh >= 0) & (fc >= 0) & (fs >= 0)

        c1 = fs + fh + fc != 1
        c2 = (incl > 90.0) | (incl < 0)
        c3 = (pa < -180) | (pa > 180)
        c4 = np.invert(cond_flux)
        c5 = (la < -1) | (la > 1.5)
        c6 = (lk < -2) | (lk > 1.0)
        c7 = (cj < -1) | (cj > 1.0)
        c8 = (sj < -1) | (sj > 1.0)
        c9 = kc < 0
        c10 = (flor < 0) | (flor > 1)
        if c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10:
            log = (
                "# fs + fh + fc = 1,\n"
                + "# 0 < incl < 90,\n"
                + "# 0 < pa < 180 deg,\n"
                + "# -1 < la < 1.5,\n"
                + "# -1 < lk < 1.\n"
                + "# -1 < cj, sj < 1.\n"
            )
            isValid = False
    elif param["model"] == "lazareff_halo":
        la = param["la"]
        lk = param["lk"]
        cj = param.get("cj", 0)
        sj = param.get("sj", 0)
        kc = param.get("kc", 0)

        flor = param.get("flor", 0)
        incl = param["incl"]
        pa = param["pa"]

        fh = param["fh"]
        fc = param["fc"]
        fs = 1 - fh - fc
        cond_flux = (fh >= 0) & (fc >= 0) & (fs >= 0)

        c1 = fs + fh + fc != 1
        c2 = (incl > 90.0) | (incl < 0)
        c3 = (pa < -180) | (pa > 180)
        c4 = np.invert(cond_flux)
        c5 = (la < -1) | (la > 1.5)
        c6 = (lk < -1) | (lk > 2.0)
        c7 = (cj < -1) | (cj > 1.0)
        c8 = (sj < -1) | (sj > 1.0)
        c9 = (kc < -6) | (kc > 0)
        c10 = (flor < 0) | (flor > 1)

        if c1 | c2 | c3 | c7 | c8:
            isValid = False

        for p in prior:
            if param[p] < prior[p][0] or (param[p] > prior[p][1]):
                isValid = False

        # if c2 | c3 | c7 | c8 | c9:
        #     log = (
        #         "# fs + fh + fc = 1,\n"
        #         + "# 0 < incl < 90,\n"
        #         + "# 0 < pa < 180 deg,\n"
        #         + "# -1 < la < 1.5,\n"
        #         + "# -1 < lk < 1.\n"
        #         + "# -1 < cj, sj < 1.\n"
        #     )
        #     isValid = False
    elif param["model"] == "lazareff_clump":
        la = param["la"]
        lk = param["lk"]
        cj = param.get("cj", 0)
        sj = param.get("sj", 0)
        kc = param.get("kc", 0)

        ratio_clump = param.get("ratio_clump", 0)
        sj = param.get("sj", 0)
        kc = param.get("kc", 0)

        flor = param.get("flor", 0)
        incl = param["incl"]
        pa = param["pa"]

        fh = param["fh"]
        fc = param["fc"]
        fs = 1 - fh - fc
        cond_flux = (fh >= 0) & (fc >= 0) & (fs >= 0)

        c1 = fs + fh + fc != 1
        c2 = (incl > 90.0) | (incl < 0)
        c3 = (pa < 0) | (pa > 180)
        c4 = np.invert(cond_flux)
        c5 = (la < -1) | (la > 1.5)
        c6 = (lk < -1) | (lk > 2.0)
        c7 = (cj < -1) | (cj > 1.0)
        c8 = (sj < -1) | (sj > 1.0)
        c9 = (kc < -6) | (kc > 0)
        c10 = (flor < 0) | (flor > 1)

        c11 = (ratio_clump < 0.0) | (ratio_clump > 1)
        if c1 | c2 | c3 | c7 | c8 | c11:
            isValid = False

        for p in prior:
            if param[p] < prior[p][0] or (param[p] > prior[p][1]):
                isValid = False

    elif param["model"] == "yso_line":
        la = param["la"]
        lk = param["lk"]

        incl = param["incl"]
        pa = param["pa"]
        fs = param["fs"]
        fc = param["fc"]
        fh = 1 - fs - fc
        cond_flux = (fs >= 0) & (fc >= 0)

        lincl = param["lincl"]
        lpa = param["lpa"]

        c1 = fs + fh + fc != 1
        c2 = (incl > 90.0) | (incl < 0)
        c3 = (pa < 0) | (pa > 180)
        c4 = np.invert(cond_flux)
        c5 = (la < -1) | (la > 1.5)
        c6 = (lk < -1) | (lk > 1.0)
        c7 = (lincl > 90) | (lincl < 0)
        c8 = (lpa < 0) | (lpa > 180)

        if c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8:
            log = (
                "# fs + fh + fc = 1,\n"
                + "# 0 < incl < 90,\n"
                + "# 0 < pa < 180 deg,\n"
                + "# -1 < la < 1.5,\n"
                + "# -1 < lk < 1.\n"
            )
            isValid = False
    elif param["model"] == "yso":
        incl = param["incl"]
        hfr = param["hfr"]
        pa = param["pa"]
        fh = param["fh"]
        fc = param["fc"]
        flor = param.get("flor", 0)
        fs = 1 - fh - fc
        cond_flux = (fs >= 0) & (fc >= 0)

        c1 = fs + fh + fc != 1
        c2 = (incl > 90.0) | (incl < 0)
        c3 = (pa < 0) | (pa > 180)
        c4 = hfr < 0
        c5 = np.invert(cond_flux)
        c6 = (flor < 0) | (flor > 1)
        if c1 | c2 | c3 | c4 | c5 | c6:
            log = "# cosi <= 1,\n# hfr > 0 mas,\n# 0 < pa < 180 deg."
            isValid = False

    return isValid, log


def comput_V2(X, param, model):
    """Compute squared visibility for a given model."""
    u = X[0]
    v = X[1]
    wl = X[2]

    isValid = check_params_model(param)[0]

    if not isValid:
        V2 = np.nan
    else:
        V = model(u, v, wl, param)
        V2 = np.abs(V) ** 2
    return V2


def comput_phi(X, param, model):
    """Compute phase visibility for a given model."""
    u = X[0]
    v = X[1]
    wl = X[2]

    isValid = check_params_model(param)[0]

    if not isValid:
        phi = np.nan
    else:
        V = model(u, v, wl, param)
        phi = np.angle(V, deg=True)
    return phi


def comput_CP(X, param, model):
    """Compute closure phases for a given model."""
    u1 = X[0]
    u2 = X[1]
    u3 = X[2]
    v1 = X[3]
    v2 = X[4]
    v3 = X[5]
    wl = X[6]

    V1 = model(u1, v1, wl, param)
    V2 = model(u2, v2, wl, param)
    V3 = model(u3, v3, wl, param)

    BS = V1 * V2 * V3
    CP = np.rad2deg(np.arctan2(BS.imag, BS.real))
    return CP


def get_stat_data(data, verbose=True):
    npts_init = 0
    n_flag_vis2, n_flag_cp = 0, 0
    n_nan_vis2, n_nan_cp = 0, 0
    n_vis2, n_cp = 0, 0
    for d in data:
        vis2 = d.vis2.flatten()
        cp = d.cp.flatten()
        flag_vis2 = d.flag_vis2.flatten()
        flag_cp = d.flag_cp.flatten()
        n_flag_vis2 += len(vis2[flag_vis2])
        n_flag_cp += len(cp[flag_cp])
        n_nan_vis2 += len(vis2[np.isnan(vis2) & ~flag_vis2])
        n_nan_cp += len(cp[np.isnan(cp) & ~flag_cp])
        n_vis2 += len(d.vis2.flatten())
        n_cp += len(d.cp.flatten())
        npts_init += len(d.vis2.flatten())
        npts_init += len(d.cp.flatten())

    txt = f"\nTotal npts = {npts_init} (vis2 = {n_vis2}, cp = {n_cp})"
    if verbose:
        cprint(txt, "cyan")
        cprint("-" * len(txt), "cyan")
        print(f"V2 flag = {n_flag_vis2} , nan = {n_nan_vis2}")
        print(f"CP flag = {n_flag_cp} , nan = {n_nan_cp}")
    npts_good = npts_init - n_nan_cp - n_nan_vis2 - n_flag_cp - n_flag_vis2
    npts_good_vis2 = n_vis2 - n_flag_vis2 - n_nan_vis2
    npts_good_cp = n_cp - n_flag_cp - n_nan_cp
    if verbose:
        print(f"Good pts = {npts_good} (vis2 = {npts_good_vis2}, cp = {npts_good_cp})")
        cprint("-" * len(txt), "cyan")
        print()
    return npts_good


def model_standard(d, param):
    l_mod_cp, l_mod_cvis, l_mod_dvis = [], [], []
    fitted = param["fitted"]
    for data in d:
        nbl = len(data.u)
        ncp = len(data.cp)
        model_target = select_model(param["model"])
        mod_cvis = []
        for i in range(nbl):
            vis2 = data.vis2[i]
            flag = data.flag_vis2[i]
            if len(vis2[~np.isnan(vis2)]) != 0:
                u, v, wl = data.u[i], data.v[i], data.wl[~flag]
                mod_cvis.append(np.squeeze(model_target(u, v, wl, param)))
        mod_dvis = []
        if "dvis" in fitted:
            for i in range(nbl):
                dvis = data.vis2[i]
                flag = data.flag_dvis[i]
                if len(vis2[~np.isnan(dvis)]) != 0:
                    u, v, wl = data.u[i], data.v[i], data.wl[~flag]
                    mod_dvis.append(model_target(u, v, wl, param))
        mod_cp = []
        for i in range(ncp):
            cp = data.cp[i]
            flag = data.flag_cp[i]
            if len(cp[~np.isnan(cp)]) != 0:
                u1, u2, u3 = data.u1[i], data.u2[i], data.u3[i]
                v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
                wl = data.wl[~flag]
                X = [u1, u2, u3, v1, v2, v3, wl]
                tmp = np.squeeze(comput_CP(X, param, model_target))
                if len(tmp) != 0:
                    mod_cp.append(tmp)
        l_mod_cp.append(np.array(mod_cp, dtype=object))
        l_mod_cvis.append(np.array(mod_cvis, dtype=object))
        l_mod_dvis.append(np.array(mod_dvis, dtype=object))
    l_mod_cp = np.array(l_mod_cp, dtype=object)
    l_mod_cvis = np.array(l_mod_cvis, dtype=object)
    l_mod_dvis = np.array(l_mod_dvis, dtype=object)

    model_fast = []
    for i in range(len(l_mod_cvis)):
        cvis = l_mod_cvis[i]
        cvis_flat = np.hstack(cvis.flatten())
        if "dvis" in fitted:
            dvis = l_mod_dvis[i]
            if len(dvis[0]) > 1:
                dvis_flat = np.hstack(dvis.flatten())
        cp = l_mod_cp[i]
        cp_flat = np.hstack(cp.flatten())
        if "V2" in fitted:
            model_fast += list(np.abs(cvis_flat) ** 2)
        if "dvis" in fitted:
            model_fast += list(np.abs(dvis_flat))
        if "dphi" in fitted:
            model_fast += list(np.angle(dvis_flat, deg=True))
        if "CP" in fitted:
            model_fast += list(cp_flat)
    model_fast = np.array(model_fast)
    npts = len(model_fast)
    isValid = check_params_model(param)[0]
    if not isValid:
        model_fast = [np.nan] * npts
    return model_fast


def model_standard_v2(d, param):
    l_mod_cp, l_mod_cvis = [], []
    fitted = param["fitted"]

    from .modelling import compute_geom_model_fast

    mod = compute_geom_model_fast(d, param, ncore=1, use_flag=True)

    npts_flagged = get_stat_data(d, verbose=False)

    l_mod_cp = []
    l_mod_cvis = []
    for i in range(len(mod)):
        m = mod[i]
        good_cp = m["good_cp"]
        good_bl = m["good_bl"]
        flag_cp = d[i].flag_cp.take(good_cp, axis=0)
        flag_vis2 = d[i].flag_vis2.take(good_bl, axis=0)
        l_mod_cp.append(m["cp"][~flag_cp])
        l_mod_cvis.append(m["cvis"][~flag_vis2])
    l_mod_cp = np.array(l_mod_cp)
    l_mod_cvis = np.array(l_mod_cvis)

    model_fast = []
    for i in range(len(mod)):
        cvis = l_mod_cvis[i]
        cvis_flat = np.hstack(cvis.flatten())
        cp = l_mod_cp[i]
        cp_flat = np.hstack(cp.flatten())
        if "V2" in fitted:
            model_fast += list(np.abs(cvis_flat) ** 2)
        if "dvis" in fitted:
            model_fast += list(np.abs(cvis_flat))
        if "dphi" in fitted:
            model_fast += list(np.angle(cvis_flat, deg=True))
        if "CP" in fitted:
            model_fast += list(cp_flat)
    model_fast = np.array(model_fast)
    isValid = check_params_model(param)[0]
    if not isValid:
        model_fast = [-np.inf] * npts_flagged
    return model_fast


def _normalize_err_obs(obs, verbose=False):
    """Normalize the errorbars to give the same weight for the V2 and CP data"""

    errs = [o[-1] for o in obs]
    techs = [("V2"), ("CP")]
    n = [0, 0, 0, 0]
    for o in obs:
        for j, t in enumerate(techs):
            if any([x in o[1] for x in t]):
                n[j] += 1
    if verbose:
        print("-" * 50)
        print("error bar normalization by observable:")
        for j, t in enumerate(techs):
            print(t, n[j], np.sqrt(float(n[j]) / len(obs) * len(n)))
        print("-" * 50)

    for i, o in enumerate(obs):
        for j, t in enumerate(techs):
            if any([x in o[1] for x in t]):
                errs[i] *= np.sqrt(float(n[j]) / len(obs) * len(n))
    return errs


def compute_chi2_curve(
    data,
    name_param,
    params,
    array_params,
    fitOnly,
    normalizeErrors=False,
    tobefit=None,
    ymin=0,
    ymax=3,
    sigma=1.0,
    reduced=True,
):
    """
    Compute a 1D reduced chi2 curve to determine the pessimistic (fully correlated)
    uncertainties on one parameter (name_param).

    Parameters:
    -----------

    `obs`: {tuple}
        Tuple containing all the selected data from format_obs function,\n
    `name_param` {str}:
        Name of the parameter to compute the chi2 curve,\n
    `params`: {dict}
        Parameters of the model,\n
    `array_params`: {array}
        List of parameters used to computed the chi2 curve,\n
    `fitOnly`: {list}
        fitOnly is a list of keywords to fit. By default, it fits all parameters in `param`,\n
    `normalizeErrors`: {str or boolean}
        If 'techniques', give the same weight for the V2 and CP data (even if only few CP compare to V2),\n
    `fitCP`: {boolean}
        If True, fit the CP data. If not fit only the V2 data.\n

    Returns:
    --------

    `fit` {dict}:
        Contains the results of the initial fit,\n
    `errors_chi2` {float}:
        Computed errors using the chi2 curve at the position of the chi2_r.min() + 1.
    """
    fit = smartfit(
        data,
        params,
        normalizeErrors=normalizeErrors,
        tobefit=tobefit,
        fitOnly=fitOnly,
        ftol=1e-8,
        epsfcn=1e-6,
        verbose=True,
    )
    fit_theta = fit["best"][name_param]
    fit_e_theta = fit["uncer"][name_param]

    fit_theta = max(fit_theta, array_params[0])

    fitOnly.remove(name_param)
    l_chi2r = []
    for pr in tqdm(array_params, desc=f"Chi2 curve ({name_param})", ncols=100):
        params[name_param] = pr
        lfits = smartfit(
            data,
            params,
            normalizeErrors=False,
            tobefit=tobefit,
            fitOnly=fitOnly,
            ftol=1e-8,
            epsfcn=1e-6,
            verbose=False,
        )
        chi2 = lfits["chi2"]
        l_chi2r.append(chi2)

    n_freedom = len(fitOnly)

    n_pts = get_stat_data(data)

    l_chi2r = np.array(l_chi2r)
    l_chi2 = np.array(l_chi2r) * (n_pts - (n_freedom - 1))

    if not reduced:
        l_chi2r = l_chi2

    chi2r_m = l_chi2r.min()
    chi2_m = l_chi2.min()

    fitted_param = array_params[l_chi2 == chi2_m]

    c_left = array_params <= fitted_param
    c_right = array_params >= fitted_param

    try:
        left_curve = interp1d(l_chi2r[c_left], array_params[c_left])
        left_res = left_curve(chi2r_m + sigma)
        dr1_r = abs(fitted_param - left_res)
        dr1_r = round_sci_digit(dr1_r[0])[0]
    except ValueError:
        dr1_r = array_params[0] - fit_theta
        dr1_r = round_sci_digit(dr1_r[0])[0]

    try:
        right_curve = interp1d(l_chi2r[c_right], array_params[c_right])
        right_res = right_curve(chi2r_m + sigma)
        dr2_r = abs(fitted_param - right_res)
        dr2_r = round_sci_digit(dr2_r[0])[0]
    except ValueError:
        dr2_r = array_params[-1] - fit_theta
        dr2_r = round_sci_digit(dr2_r[0])[0]

    bound = np.array([dr1_r, dr2_r])
    bound = bound[~np.isnan(bound)]
    errors_chi2 = bound[0] if len(bound) == 1 else np.mean([dr1_r, dr2_r])

    plt.figure()
    plt.plot(array_params[c_left], l_chi2r[c_left], color="tab:blue", lw=3, alpha=1, zorder=1)
    plt.plot(array_params[c_right], l_chi2r[c_right], color="tab:blue", lw=3, alpha=1)
    plt.plot(
        fit_theta,
        l_chi2r.min(),
        ".",
        color="#fc5185",
        ms=10,
        label=f"fit: {name_param}={fit_theta:2.2f}±{fit_e_theta:2.2f}",
    )
    plt.axvspan(
        fitted_param[0] - dr1_r,
        fitted_param[0] + dr2_r,
        ymin=ymin,
        ymax=ymax,
        color="#dbe4e8",
        label=rf"$\sigma_{{m1}}=$-{dr1_r}/+{dr2_r}",
    )
    plt.axvspan(
        fitted_param[0] - fit_e_theta,
        fitted_param[0] + fit_e_theta,
        ymin=ymin,
        ymax=ymax,
        color="#359ccb",
        # label=r"$\sigma_{m2}$",
        alpha=0.3,
    )
    plt.grid(alpha=0.1, color="grey")
    plt.legend(loc="best", fontsize=9)
    if reduced:
        plt.ylabel(r"$\chi^2_{red}$")
    else:
        plt.ylabel(r"$\chi^2$")

    plt.xlim(array_params.min(), array_params.max())
    plt.tight_layout()
    plt.show(block=False)

    return fit, errors_chi2


def smartfit(
    data,
    first_guess,
    prior=None,
    doNotFit=None,
    fitOnly=None,
    follow=None,
    ftol=1e-4,
    epsfcn=1e-7,
    normalizeErrors=False,
    scale_err=1,
    fast=True,
    verbose=False,
    tobefit=None,
):
    """
    Perform the fit the observable in `tobefit` contained in data list (or class).

    Parameters:
    -----------

    data: {list}
        List containing all the selected data from `load()` function.\n
    first_guess: {dict}
        Parameters of the model.\n
    fitOnly: {list}
        fitOnly is a LIST of keywords to fit. By default, it fits all parameters in 'first_guess'.\n
    follow: {list}
        List of parameters to "follow" in the fit, i.e. to print in verbose mode.\n
    normalizeErrors: {boolean}
        If True, give the same weight for each observables (even if only few CP compare to V2).
    """
    if tobefit is None:
        tobefit = ["CP", "V2"]
    if prior is None:
        prior = {}

    first_guess["fitted"] = tobefit
    first_guess["prior"] = prior
    first_guess["fitOnly"] = fitOnly
    # -- avoid fitting string parameters
    tmp = list(filter(lambda x: isinstance(first_guess[x], str), first_guess.keys()))

    if not isinstance(data, list):
        data = [data]

    if len(tmp) > 0:
        if doNotFit is None:
            doNotFit = tmp
        else:
            doNotFit.extend(tmp)
        with contextlib.suppress(Exception):
            fitOnly = list(filter(lambda x: not isinstance(first_guess[x], str), fitOnly))

    obs = np.concatenate([format_obs(x, use_flag=True) for x in data])
    save_obs = obs.copy()
    obs = []
    for o in save_obs:
        if o[1] in tobefit:
            obs.append(o)
    obs = np.array(obs)

    errs = [o[-1] for o in obs]
    if normalizeErrors:
        errs = _normalize_err_obs(obs, verbose=True)

    Y = [o[2] for o in obs]

    npts = len(Y)
    npts_good_check = get_stat_data(data, verbose=False)
    if verbose and npts_good_check != npts:
        print(f"Npts data = {npts} (should be {npts_good_check})")

    fct_model = model_standard_v2 if fast else model_standard

    lfit = leastsqFit(
        fct_model,
        data,
        first_guess,
        Y,
        err=np.array(errs) * scale_err,
        doNotFit=doNotFit,
        fitOnly=fitOnly,
        follow=follow,
        normalizedUncer=False,
        fullOutput=True,
        verbose=verbose,
        ftol=ftol,
        epsfcn=epsfcn,
    )

    p = {}
    for k in fitOnly:
        err = lfit["uncer"][k]
        if err < 0:
            err = np.nan
        p[k] = ufloat(lfit["best"][k], err)
    lfit["p"] = p
    return lfit


def _compute_uvcoord(data):
    nwl = len(data.wl)
    nbl = data.vis2.shape[0]
    ncp = data.cp.shape[0]

    u_data = np.zeros([nbl, nwl])
    v_data = np.zeros_like(u_data)

    u1_data = np.zeros([ncp, nwl])
    u2_data = np.zeros_like(u1_data)
    u3_data = np.zeros_like(u1_data)
    v1_data = np.zeros_like(u1_data)
    v2_data = np.zeros_like(u1_data)
    v3_data = np.zeros_like(u1_data)

    for i in range(nbl):
        u_data[i] = np.array([data.u[i]] * nwl)
        v_data[i] = np.array([data.v[i]] * nwl)
    u_data, v_data = u_data.flatten(), v_data.flatten()

    for i in range(ncp):
        u1_data[i] = np.array([data.u1[i]] * nwl)
        u2_data[i] = np.array([data.u2[i]] * nwl)
        u3_data[i] = np.array([data.u3[i]] * nwl)
        v1_data[i] = np.array([data.v1[i]] * nwl)
        v2_data[i] = np.array([data.v2[i]] * nwl)
        v3_data[i] = np.array([data.v3[i]] * nwl)

    u1_data, v1_data = u1_data.flatten(), v1_data.flatten()
    u2_data, v2_data = u2_data.flatten(), v2_data.flatten()
    u3_data, v3_data = u3_data.flatten(), v3_data.flatten()

    output = dict2class(
        {
            "u": u_data,
            "v": v_data,
            "u1": u1_data,
            "v1": v1_data,
            "u2": u2_data,
            "v2": v2_data,
            "u3": u3_data,
            "v3": v3_data,
        }
    )
    return output


def format_obs(data, use_flag=False, input_rad=False, verbose=False):
    nwl = len(data.wl)
    nbl = data.vis2.shape[0]
    ncp = data.cp.shape[0]

    vis2_data = data.vis2.flatten()
    e_vis2_data = data.e_vis2.flatten()
    flag_V2 = data.flag_vis2.flatten()

    dvis_data = data.dvis.flatten()
    e_dvis_data = data.e_dvis.flatten()
    flag_dvis = data.flag_dvis.flatten()

    dphi_data = data.dphi.flatten()
    e_dphi_data = data.e_dphi.flatten()
    flag_dphi = data.flag_dvis.flatten()

    cp_data = data.cp.flatten()
    e_cp_data = data.e_cp.flatten()
    flag_CP = data.flag_cp.flatten()
    if input_rad:
        cp_data = np.rad2deg(cp_data)
        e_cp_data = np.rad2deg(e_cp_data)

    if not use_flag:
        flag_V2 = [False] * len(vis2_data)
        flag_CP = [False] * len(cp_data)
        flag_dvis = [False] * len(dvis_data)
        flag_dphi = [False] * len(dphi_data)

    uv = _compute_uvcoord(data)

    wl_data = np.array(list(data.wl) * nbl)
    wl_data_cp = np.array(list(data.wl) * ncp)

    obs = []
    for i in range(nbl * nwl):
        if not flag_V2[i]:
            tmp = [uv.u[i], uv.v[i], wl_data[i]]
            typ = "V2"
            obser = vis2_data[i]
            err = e_vis2_data[i]
            if ~np.isnan(obser) | ~np.isnan(err):
                obs.append([tmp, typ, obser, err])
    N_v2 = len(obs)

    for i in range(nbl * nwl):
        if not flag_dvis[i]:
            tmp = [uv.u[i], uv.v[i], wl_data[i]]
            typ = "dvis"
            obser = dvis_data[i]
            err = e_dvis_data[i]
            obs.append([tmp, typ, obser, err])
    N_vis = len(obs) - N_v2

    for i in range(nbl * nwl):
        if not flag_dphi[i]:
            tmp = [uv.u[i], uv.v[i], wl_data[i]]
            typ = "dphi"
            obser = dphi_data[i]
            err = e_dphi_data[i]
            obs.append([tmp, typ, obser, err])
    N_phi = len(obs) - N_v2 - N_vis

    for i in range(ncp * nwl):
        if not flag_CP[i]:
            tmp = [
                uv.u1[i],
                uv.u2[i],
                uv.u3[i],
                uv.v1[i],
                uv.v2[i],
                uv.v3[i],
                wl_data_cp[i],
            ]
            typ = "CP"
            obser = cp_data[i]
            err = e_cp_data[i]
            if ~np.isnan(obser) | ~np.isnan(err):
                obs.append([tmp, typ, obser, err])
    N_cp = len(obs) - N_v2 - N_vis - N_phi
    Obs = np.array(obs, dtype=object)
    if verbose:
        print(f"\nTotal # of data points: {len(Obs)} ({N_v2} V2, {N_cp} CP)")
    return Obs


# MCMC estimation of uncertainties and refined parameter estimates
def _compute_model_mcmc(p_mcmc, data, param, fitOnly, tobefit):
    """Compute model for mcmc purpose changing only the parameters from `fitOnly`.

    Parameters:
    -----------
    `p_mcmc` {list, float}:
        List of parameter values formated for mcmc fitting process (the order is
        the same as `fitOnly` one),\n
    `obs` {array}:
        List of data formated with oimalib.format_obs(). obs[0] is the u, v and
        wl position, obs[1] is the observable type ('V2', 'CP', 'VIS'), obs[2]
        is the observable value and obs[3] is the uncertainty,\n
    `param` {dict}:
        Dictionnary of parameters used with oimalib features,\n
    `fitOnly` {list, str}:
        List of parameters to be fit (smartfit LM or MCMC).

    Results:
    --------
    `model` {array}:
        Computed model values sorted as obs (model.shape = len(observables)).
    """
    for i, p in enumerate(fitOnly):
        param[p] = p_mcmc[i]
    param["fitted"] = tobefit
    model = model_standard_v2(data, param)
    return np.array(model)


def log_prior(param, prior, fitOnly):
    """Return -inf if param[p] is outside prior requirements."""
    for i, p in enumerate(fitOnly):
        if param[i] < prior[p][0] or (param[i] > prior[p][1]):
            return -np.inf
    return 0


def log_likelihood(p_mcmc, data, param, fitOnly, tobefit, obs=None, fd=None, e_fd=None):
    """Compute the likelihood estimation between the model (represented by param
    and only for fitOnly parameters) and the data (obs).

    Parameters:
    -----------
    `p_mcmc` {list, float}:
        List of parameter values formated for mcmc fitting process (the order is
        the same as `fitOnly` one),\n
    `obs` {array}:
        List of data formated with oimalib.format_obs(). obs[0] is the u, v and
        wl position, obs[1] is the observable type ('V2', 'CP', 'VIS'), obs[2]
        is the observable value and obs[3] is the uncertainty,\n
    `param` {dict}:
        Dictionnary of parameters used with oimalib features,\n
    `fitOnly` {list, str}:
        List of parameters to be fit (smartfit LM or MCMC).
    """
    model = _compute_model_mcmc(p_mcmc, data, param, fitOnly, tobefit)

    y = obs[:, 2]
    e_y = obs[:, 3]

    inv_sigma2 = 1.0 / (e_y**2)

    if "fc" in param:
        fc = param["fc"]
        add_cons_sed = (fc - fd) ** 2 / e_fd**2
    else:
        add_cons_sed = 0

    res = -0.5 * (
        np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2.astype(float))) + add_cons_sed
    )
    # res = -.5 * (np.sum(delta_y ** 2 / sigma2 + np.log(2 * np.pi * sigma2)) + add_cons_sed)

    if np.isnan(res):
        return -np.inf
    else:
        return res


def log_probability(p_mcmc, data, param, fitOnly, prior, tobefit, obs=None, fd=None, e_fd=None):
    """Similar to log_probability() but including the prior restrictions.

    Parameters:
    -----------
    `p_mcmc` {list, float}:
        List of parameter values formated for mcmc fitting process (the order is
        the same as `fitOnly` one),\n
    `obs` {array}:
        List of data formated with oimalib.format_obs(). obs[0] is the u, v and
        wl position, obs[1] is the observable type ('V2', 'CP', 'VIS'), obs[2]
        is the observable value and obs[3] is the uncertainty,\n
    `param` {dict}:
        Dictionnary of parameters used with oimalib features,\n
    `fitOnly` {list, str}:
        List of parameters to be fit (smartfit LM or MCMC),\n
    `prior` {dict}:
        Dictionnary with the keys as fitOnly where the param[a] need to be
        between prior[a][0] <= param[a] <= prior[a][1].
    """
    lp = log_prior(p_mcmc, prior, fitOnly)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(p_mcmc, data, param, fitOnly, tobefit, obs=obs, fd=fd, e_fd=e_fd)


def neg_like_prob(*args):
    """Minus log_probability computed to be used with minimize (scipy.optimize)."""
    return -log_probability(*args)


def _compute_initial_dist_mcmc(
    nwalkers,
    fitOnly,
    method="normal",
    w_walker=0.1,
    param=None,
    prior=None,
):
    nparam = len(fitOnly)
    pos = np.zeros([nwalkers, nparam])

    if method == "normal":
        for i in range(nparam):
            pp = param[fitOnly[i]]
            if pp == 0:
                i_p = prior[fitOnly[i]]
                m1, m2 = i_p[0], i_p[1]
                i_range = np.random.uniform(m1, m2, nwalkers)
            else:
                i_range = np.random.normal(pp, w_walker * abs(pp), nwalkers)
            pos[:, i] = i_range
    elif method == "prior":
        for i in range(nparam):
            pp = param[fitOnly[i]]
            i_p = prior[fitOnly[i]]
            m1, m2 = i_p[0], i_p[1]
            i_range = np.random.uniform(m1 * 0.98, 0.98 * m2, nwalkers)
            pos[:, i] = i_range
    elif method == "alex":
        p0 = np.array([param[fitOnly[i]] for i in range(nparam)])
        pos = p0 + 1e-4 * np.random.randn(nwalkers, nparam)
    return pos


def mcmcfit(
    data,
    first_guess,
    nwalkers=20,
    niter=150,
    prior=None,
    truths=None,
    fitOnly=None,
    guess_likehood=False,
    method="normal",
    threads=1,
    progress=True,
    plot_corner=False,
    burnin=50,
    tobefit=None,
    fd=None,
    e_fd=None,
):
    """ """
    ndim = len(fitOnly)
    if tobefit is None:
        tobefit = ["V2", "CP"]

    obs = np.concatenate([format_obs(x, use_flag=True) for x in data])
    save_obs = obs.copy()
    obs = []
    for o in save_obs:
        if o[1] in tobefit:
            obs.append(o)
    obs = np.array(obs)

    args = (data, first_guess, fitOnly, prior, tobefit, obs, fd, e_fd)

    initial_mcmc = [first_guess[p] for p in fitOnly]
    if guess_likehood:
        initial_mcmc = [first_guess[p] for p in fitOnly]
        initial_like = np.array(initial_mcmc) + 0.1 * np.random.randn(len(initial_mcmc))
        soln = minimize(neg_like_prob, initial_like, args=args)
        initial_mcmc = soln.x
        print("\nMaximum likelihood estimates:")
        for i in range(ndim):
            print(f"{fitOnly[i]} = {initial_mcmc[i]:2.3f}")

    pool = multiprocessing.Pool(threads)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=args,
        pool=pool,
    )

    pos = _compute_initial_dist_mcmc(
        nwalkers,
        fitOnly,
        method=method,
        param=first_guess,
        prior=prior,
    )

    cprint("------- MCMC is running ------", "cyan")
    sampler.run_mcmc(
        pos,
        niter,
        progress=progress,
        skip_initial_state_check=True,
    )

    pool.close()
    pool.join()

    if burnin > niter:
        cprint(
            f"## Warning: Burnin = {burnin} should be < {niter} iter (force to zero).",
            "green",
        )
        cprint("-> You should increase the niter value (typically >100)", "green")
        burnin = 0

    flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "{3} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], fitOnly[i])
        print(txt)
        # if mcmc[1] == "la":
        #     print(txt)

    if plot_corner:
        corner.corner(
            flat_samples,
            labels=fitOnly,
            bins=50,
            show_titles=True,
            title_fmt=".3f",
            color="k",
            truths=truths,
        )
    return sampler


def get_mcmc_results(sampler, param, fitOnly, burnin=200):
    flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)

    ndim = len(fitOnly)

    fit_mcmc = {}
    fit_mcmc["best"] = param.copy()
    fit_mcmc["uncer"] = {}
    fit_mcmc["p"] = {}
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "{3} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], fitOnly[i])
        fit_mcmc["best"][fitOnly[i]] = mcmc[1]
        fit_mcmc["uncer"][fitOnly[i] + "_m"] = q[0]
        fit_mcmc["uncer"][fitOnly[i] + "_p"] = q[1]
        fit_mcmc["p"][fitOnly[i]] = ufloat(mcmc[1], np.max([q[0], q[1]]))

    try:
        ind_lk = np.argwhere(np.array(fitOnly) == "lk")[0][0]
        ind_la = np.argwhere(np.array(fitOnly) == "la")[0][0]

        lk = flat_samples[:, ind_lk]
        la = flat_samples[:, ind_la]
        ar = 10**la / (np.sqrt(1 + 10 ** (2 * lk)))
        ak = ar * (10**lk)
        a = (ar**2 + ak**2) ** 0.5
        w = ak / a
        mcmc2 = np.percentile(w, [16, 50, 84])
        q = np.diff(mcmc2)
        txt = "{3} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(w[1], q[0], q[1], "w")
        fit_mcmc["best"]["w"] = mcmc2[1]
        fit_mcmc["uncer"]["w_m"] = q[0]
        fit_mcmc["uncer"]["w_p"] = q[1]
        fit_mcmc["p"]["w"] = ufloat(w[1], np.max([q[0], q[1]]))
    except Exception:
        pass
    return fit_mcmc


def model_flux_blue_excess(x, param):
    "Double Gaussian model used to fit the spectrum (gravity)"
    p1 = param["p1"]
    a1 = param["a1"]
    w1 = param["w1"]
    p2 = param["p2"]
    a2 = param["a2"]
    w2 = param["w2"]
    y1 = a1 * np.exp(-0.5 * (x - p1) ** 2 / w1**2)
    y2 = a2 * np.exp(-0.5 * (x - p2) ** 2 / w2**2)
    y = y1 + y2
    return y


def model_flux_red_abs(x, param):
    "Double Gaussian model used to fit the spectrum (gravity)"
    p1 = param["p1"]
    a1 = param["a1"]
    w1 = param["w1"]
    p2 = param["p2"]
    a2 = param["a2"]
    w2 = param["w2"]
    y1 = a1 * np.exp(-0.5 * (x - p1) ** 2 / w1**2)
    y2 = a2 * np.exp(-0.5 * (x - p2) ** 2 / w2**2)
    y = y1 + y2
    return y


def model_flux(x, param):
    "Gaussian model used to fit the spectrum (gravity)"
    lbdBrg = param["lbdBrg"]
    sigBrg = param["sigBrg"]
    lF = param["lF"]
    y = lF * np.exp(-0.5 * (x - lbdBrg) ** 2 / sigBrg**2)
    return y


def fit_flc_spectra(
    data,
    wl0=2.1661,
    fwhm0=0.0005,
    r_brg=3,
    red_abs=False,
    err_cont=True,
    verbose=False,
    display=False,
    norm=True,
    use_model=True,
    tellu=False,
    force_wBrg=None,
    force_restframe=None,
):
    """
    Fit the spectral line of GRAVITY (`lbdBrg`) and return the fitted line to
    continuum ratio `F_lc` (used for pure line visibility computation).

    Parameters
    ----------

    `data` : {dict}
        Class-like object containing the oifits data from
        oimalib.load(). Generaly, the flc is computed from an averaged dataset using
        oimalib.temporal_bin_data(),\n
    `lbdBrg` : {float}
        Central wavelength position (initial) [µm],\n
    `wBrg` : {float}
        Width of the line (initial) [µm],\n
    `r_brg` : {float}
        Number of `wBrg` used to determine the in-line region,\n
    `err_cont` : {bool}
        If True, the continuum is used as error,\n
    `force_wBrg`: {float}
        If not None, inLine is computed using the sigma line.

    Returns:
    -----------
    `flc`: {dict}
        dictionnary with `flux` (normalized), `e_flux`, wavelength (`wl` in [µm]), the
        gaussian fit of the flux (`F_lc`), the fit parameters (`fit`), boolean array
        containing the continuum region (`inCont`) and in line region (`inLine`).

    """
    flux = data.flux
    wl = data.wl * 1e6  # to be in µm

    # Select region of the continuum (avoiding the BrG line)
    inCont = (np.abs(wl - wl0) < 0.1) * (np.abs(wl - wl0) > 0.003)

    # Normalize the flux to 1
    if norm:
        normalize_continuum(flux, wl, inCont)

    e_flux = np.std(flux[inCont]) if err_cont else 0.001 * flux.max()

    Y = flux - 1  # Shift to zero for gaussian fit
    X = wl

    err = np.ones(len(Y)) * e_flux
    if not red_abs:
        param = {"lbdBrg": wl0, "sigBrg": fwhm0, "lF": 0.4}
        name_model = model_flux
    else:
        param = {
            "p1": wl0,
            "w1": fwhm0 / 2.0,
            "a1": 0.6,
            "p2": wl0 + fwhm0,
            "w2": fwhm0 / 2.0,
            "a2": -0.2,
        }
        name_model = model_flux_red_abs

    fit = leastsqFit(
        name_model,
        X,
        param,
        Y,
        err=err,
        verbose=verbose,
    )
    F_lc = name_model(wl, fit["best"]) + 1 if use_model else flux

    best_param = fit["best"]
    if not red_abs:
        restframe = best_param["lbdBrg"]
        wBrg = 2.355 * best_param["sigBrg"]
        lcr = 1 + best_param["lF"]
    else:
        restframe = best_param["p1"]
        wBrg = 2.355 * best_param["w1"]
        lcr = 1 + best_param["a1"]

    if force_wBrg is not None:
        wBrg = force_wBrg
    if force_restframe is not None:
        restframe = force_restframe

    inLine = np.abs(wl - restframe) < r_brg * wBrg / 2.0
    inCont = (np.abs(wl - restframe) < 0.1) * (np.abs(wl - restframe) > 1.8 * wBrg)

    wl_model = np.linspace(wl[0], wl[-1], 1000)
    flux_model = name_model(wl_model, fit["best"]) + 1

    if display:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            X[inLine],
            Y[inLine] + 1,
            c=X[inLine],
            cmap="coolwarm",
            s=40,
            edgecolors="k",
            linewidth=0.5,
            marker="s",
            zorder=3,
        )
        plt.axhline(lcr, color="#b4c1d0", ls=":", alpha=1)
        plt.text(
            restframe - 2.9 * wBrg,
            lcr - 0.02,
            f"LCR = {lcr:2.2f}",
            ha="left",
            va="top",
            color="#b4c1d0",
            alpha=1,
        )
        plt.errorbar(
            wl[~inCont],
            flux[~inCont],
            yerr=e_flux,
            color="tab:blue",
            ms=9,
            **err_pts_style_f,
        )
        plt.errorbar(
            wl[inCont],
            flux[inCont],
            yerr=e_flux,
            color="tab:red",
            ms=9,
            **err_pts_style_f,
        )
        plt.axhline(1, lw=2, color="gray", alpha=0.2)
        plt.axhline(1 - e_flux, lw=2, color="crimson", alpha=0.5, ls="--")
        plt.axhline(1 + e_flux, lw=2, color="crimson", alpha=0.5, ls="--")
        plt.xlim(restframe - 5 * wBrg, restframe + 5 * wBrg)
        plt.axvspan(
            restframe - wBrg / 2,
            restframe + wBrg / 2.0,
            zorder=1,
            alpha=0.2,
            label=f"$w$={wBrg:2.4f} µm",
        )
        plt.axvline(
            restframe,
            c="tab:blue",
            alpha=0.5,
            lw=1,
            label=rf"$\lambda_{{0}}$={restframe:2.4f} µm",
        )
        tellu_pos = [
            2.15909,
            2.1599,
            2.16105,
            2.163446,
            2.166853,
            2.168645,
            2.17265,
            2.1744571,
        ]
        if tellu:
            for x in tellu_pos:
                plt.axvline(x, color="g", lw=1, alpha=0.1)

        plt.plot(wl_model, flux_model, alpha=0.5)
        plt.legend(loc=1, fontsize=14)
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("Norm. Flux")
        plt.tight_layout(pad=1.01)
        plt.show(block=False)

    flc = {
        "flux": flux,
        "wl": wl,
        "e_flux": e_flux,
        "F_lc": F_lc,
        "fit": fit,
        "inCont": inCont,
        "inLine": inLine,
        "red_abs": red_abs,
        "widthline": wBrg,
        "restframe": restframe,
        "wl_model": wl_model,
        "flux_model": flux_model,
    }
    return flc


def model_pcshift(x, param):
    p = param["p"]
    offset = np.deg2rad(param["offset"])
    bl_pa = np.deg2rad(x)
    pc = p * np.cos(offset - bl_pa)
    if (param["offset"] < 0) or (param["p"] < 0) or (param["offset"] > 360):
        pc = [np.nan] * len(x)
    return pc


def fit_pc_shift(output_pco, p=0.05):
    """Fit the photocenter offset across the different baselines (for all
    wavelength). `p` is the photocenter shift (in µas), `offset` is the initial orientation
    of this photocenter shift (in degree). `output_pco` is the results from
    compute_pco_bl().
    """
    pco = output_pco["pco"]
    bl_pa = output_pco["bl_pa"]
    wl_line = output_pco["wl_line"]
    nbl = pco.shape[0]
    nwl = pco.shape[1]
    l_fit, l_u_x, l_u_y = [], [], []
    for j in range(nwl):
        x_pc, y_pc, e_pc = [], [], []
        for i in range(nbl):
            x_pc.append(bl_pa[i])
            x_pc.append(bl_pa[i] - 180.0)
            y_pc.append(pco[i, j].nominal_value)
            y_pc.append(-pco[i, j].nominal_value)
            e_pc.append(pco[i, j].std_dev)
            e_pc.append(pco[i, j].std_dev)
        x_pc = np.array(x_pc)
        y_pc = np.array(y_pc)
        e_pc = np.array(e_pc)

        chi2_tmp = 1e50
        for o in np.arange(0, 360, 45):
            param = {"p": p, "offset": o}
            fit_tmp = leastsqFit(model_pcshift, x_pc, param, y_pc, err=e_pc, verbose=False)
            chi2 = fit_tmp["chi2"]
            if chi2 <= chi2_tmp:
                fit_pc = fit_tmp
                chi2_tmp = chi2

        l_fit.append(fit_pc)
        u_pc = ufloat(fit_pc["best"]["p"] * 1000, fit_pc["uncer"]["p"] * 1000)
        u_pa = ufloat(fit_pc["best"]["offset"], fit_pc["uncer"]["offset"])
        east, north = compute_oriented_shift(u_pa, u_pc)
        l_u_x.append(east)
        l_u_y.append(north)
    pcs_east = np.array(l_u_x)
    pcs_north = np.array(l_u_y)
    pcs = {
        "east": pcs_east,
        "north": pcs_north,
        "fit_param": l_fit,
        "wl": output_pco["wl"],
        "wl_line": wl_line,
        "restframe": output_pco["restframe"],
        # "cond": output_pco["cond_positivity"],
    }
    return pcs


def model_1dgauss_offset(x, param):
    pos = param["pos"]
    sigma = param["sigma"]
    A = param["A"]
    B = param["B"]
    y = ((A * np.exp(-0.5 * (x - pos) ** 2 / sigma**2)) + 1) - (1 - B)
    return y


def model_1dgauss_offset_double(x, param):
    pos = param["pos"]
    dp = param["dp"]
    sigmaA = param["sigmaA"]
    sigmaB = param["sigmaB"]
    A = param["A"]
    B = param["B"]
    C = param.get("C", 0)
    posA = pos - dp
    posB = pos + dp
    y1 = A * np.exp(-((x - posA) ** 2) / (2 * sigmaA**2))
    y2 = B * np.exp(-((x - posB) ** 2) / (2 * sigmaB**2))
    y = y1 + y2 + C
    # if (sigmaA >= 0.0015 / 2.355) or (sigmaB > 0.0015 / 2.355):
    #     y = [np.nan] * len(y)
    return y


def perform_fit_dvis(wl, dvis, e_dvis, param, double=False, inCont=None, display=False):
    fitOnly = None
    if not double:
        fitOnly = ["A", "sigmaA", "C", "pos"]
        param["B"] = 0

    chi2_tmp = 1e50
    np.array([0.0001, 0.0005, 0.001]) / 2.355

    if inCont is None:
        inCont = np.array([False] * len(wl))

    initial_guess = model_1dgauss_offset_double(wl, param)
    if display:
        plt.figure(figsize=(7, 4))
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("Amp. Vis.")
        plt.errorbar(
            wl[~inCont],
            dvis[~inCont],
            yerr=e_dvis[~inCont],
            label="Data",
            **err_pts_style,
        )
        plt.errorbar(
            wl[inCont],
            dvis[inCont],
            yerr=e_dvis[inCont],
            color="darkorange",
            **err_pts_style,
        )
        plt.plot(wl, initial_guess, lw=1, label="Initial guess")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show(block=False)

    fit = leastsqFit(
        model_1dgauss_offset_double,
        wl[~inCont],
        param,
        dvis[~inCont],
        err=e_dvis[~inCont],
        fitOnly=fitOnly,
        verbose=False,
    )
    chi2 = fit["chi2"]
    if not np.isnan(chi2) and chi2 < chi2_tmp:
        fit_best = fit
        chi2_tmp = chi2

    try:
        mod_dvis = model_1dgauss_offset_double(wl, fit_best["best"])
    except UnboundLocalError:
        mod_dvis = None
        fit_best = None
    return mod_dvis, fit_best


def perform_fit_dphi(
    wl, dphi, e_dphi, param, inCont=None, double=True, display=False, verbose=False
):
    fitOnly = None
    if not double:
        fitOnly = ["A", "sigmaA", "pos"]
        param["B"] = 0
    # fitOnly = ["A", "sigmaA", "pos", "dp"]

    dphi_mod = dphi.copy()
    initial_guess = model_1dgauss_offset_double(wl, param)
    if inCont is None:
        inCont = np.array([True] * len(wl))
    dphi_mod[~inCont] = 0

    if display:
        plt.figure(figsize=(7, 4))
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("Phase Vis.")
        plt.errorbar(wl, dphi, yerr=e_dphi, label="Data", **err_pts_style)
        plt.plot(wl, initial_guess, lw=1, label="Initial guess")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show(block=False)

    fit = leastsqFit(
        model_1dgauss_offset_double,
        wl,
        param,
        dphi_mod,
        err=e_dphi,
        fitOnly=fitOnly,
        verbose=verbose,
    )
    mod_dphi = model_1dgauss_offset_double(wl, fit["best"])
    return mod_dphi, fit


# Fit MCFOST model (from the `Model` class in models.py)


def fit_size(Model, i_wl, dwl=None, param="fwhm", display=True, verbose=True):
    """Fit size from the pure line amplitude. Use only 1 wavelength (index
    `i_wl`) if `dwl` == None (if not, take a range around `i_wl`+`dwl`)."""
    if dwl is None:
        wl = np.array([Model.plwl[i_wl]]) * 1e-6
        plvisamp = np.squeeze(Model.plvisamp)
        Y = plvisamp[:, i_wl]
    else:
        wl = np.array([Model.plwl[i_wl - dwl : i_wl + dwl].mean() * 1e-6])
        plvisamp = np.squeeze(Model.plvisamp)
        Y = np.mean(plvisamp[:, i_wl - dwl : i_wl + dwl], axis=1)

    X = [Model.bx, Model.by, wl]

    t = np.linspace(0, 2 * np.pi, 100)
    chi2_min = 1e50
    for pa in np.arange(0, 180, 30):
        for ii in np.arange(0, 70, 20):
            fit = leastsqFit(
                model_acc_mag,
                X,
                {"minorAxis": 1, "incl": ii, "pa": pa},
                Y,
            )
            chi2 = fit["chi2"]
            if chi2 < chi2_min:
                fit_best = fit
                chi2_min = fit_best["chi2"]
    pa = fit_best["best"]["pa"]
    elong = 1 / np.cos(np.deg2rad(fit_best["best"]["incl"]))

    # Compute ellipse coordinates from model size
    b = 0.5 * fit_best["best"]["minorAxis"] * Model.distance / 1e3
    r_ellipse = b * elong
    t_rot = np.deg2rad(90 - pa)  # rotation angle

    Ell = np.array([r_ellipse * np.cos(t), b * np.sin(t)])
    R_rot = np.array([[np.cos(t_rot), -np.sin(t_rot)], [np.sin(t_rot), np.cos(t_rot)]])
    egauss_model = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        egauss_model[:, i] = np.dot(R_rot, Ell[:, i])

    if verbose:
        print("\n---------------------- Fit size ------------------------")
        print(
            f"R = {r_ellipse:2.5f} au, elong = {elong:2.1f}, pa = {pa:2.1f} deg @ {wl.mean() * 1e6:2.4f} µm (elliptic),"
        )
    # else:
    fit_best_gauss = leastsqFit(
        model_acc_mag,
        X,
        {param: 1},
        Y,
    )
    r_gauss = (fit_best_gauss["best"][param] / 2.0) * Model.distance / 1e3
    gauss_model = np.array([r_gauss * np.cos(t), r_gauss * np.sin(t)])

    if verbose:
        print(f"R = {r_gauss:2.5f} au (gaussian).\n")

    u_model = np.linspace(0, 200, 100)
    v_model = np.zeros_like(u_model)
    y_model = model_acc_mag([u_model, v_model, wl], fit_best_gauss["best"])

    if display:
        from oimalib.plotting import err_pts_style

        plt.figure(figsize=(8, 5))
        plt.title("Fit size @ %2.4f µm" % (wl.mean() * 1e6))
        plt.errorbar(
            Model.bl,
            Y,
            yerr=np.zeros_like(plvisamp[:, i_wl]),
            label="data",
            color="#4c84b8",
            **err_pts_style,
        )
        plt.plot(
            Model.bl,
            fit_best["model"],
            "+",
            color="lime",
            zorder=3,
            label=rf"r$_{{ellipse}}$={r_ellipse:2.3f} au",
        )
        plt.legend()
        plt.plot(u_model, y_model, color="#eba15c", label=rf"r$_{{gauss}}$={r_gauss:2.3f} au")
        plt.legend(loc=3)
        plt.xlabel("Baseline lengths [m]")
        plt.ylabel("Pure line amplitude")
        plt.tight_layout(pad=1.01)

    return r_ellipse, r_gauss, fit_best, fit_best_gauss, egauss_model, gauss_model


def fit_multi_size(Model, param="fwhm", display=False):
    pl_wl = Model.plwl
    pl_size = np.zeros([2, len(pl_wl)])
    l_gauss = []
    for i in range(len(pl_wl)):
        output = fit_size(Model, i, verbose=False, display=display, param=param)

        pl_size[0, i] = output[0]
        pl_size[1, i] = output[1]
        l_gauss.append(output[4])
    return pl_size, l_gauss
