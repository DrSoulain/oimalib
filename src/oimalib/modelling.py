"""
@author: Anthony Soulain (University of Sydney)
-----------------------------------------------------------------
OIMALIB: Optical Interferometry Modelisation and Analysis Library
-----------------------------------------------------------------

Set of function to extract complex visibilities from fits image/cube
or geometrical model.
-----------------------------------------------------------------
"""

import multiprocessing
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import fits
from matplotlib import pyplot as plt
from munch import munchify as dict2class
from scipy import fft
from scipy.interpolate import RegularGridInterpolator as regip, interp1d, interp2d
from scipy.ndimage import gaussian_filter1d, rotate
from termcolor import cprint

import oimalib
from oimalib.fitting import check_params_model, comput_CP, comput_V2, select_model
from oimalib.fourier import UVGrid
from oimalib.tools import apply_windowing, mas2rad, rad2arcsec, rad2mas


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


def _print_info_model(wl_model, modelfile, fov, npix, s, starttime):
    nwl = len(wl_model)
    modeltype = "image" if nwl == 1 else "cube"
    try:
        fname = modelfile.split("/")[-1]
    except AttributeError:
        fname = modelfile.name
    title = f"\nModel grid from {modeltype} ({fname})"
    cprint(title, "cyan")
    cprint("-" * len(title), "cyan")
    pixsize = fov / npix
    print(
        f"fov={fov:2.2f} mas, npix={npix} ({s[2]} padded, "
        f"equivalent {s[2] * pixsize:2.2f} mas), pix={pixsize:2.4f} mas"
    )
    if nwl == 1:
        cprint(f"nwl={nwl} ({np.mean(wl_model) * 1e6:2.1f} µm)", "green")
    else:
        wl1 = wl_model[0] * 1e6
        wl2 = wl_model[-1] * 1e6
        wl_step = np.diff(wl_model)[0] * 1e6
        cprint(
            f"nwl={nwl} (wl0={wl1:2.3f}, wlmax={wl2:2.3f} µm, step={wl_step * 1000.0:2.3f} nm)",
            "green",
        )
    print("Computation time = %2.2f s" % (time.time() - starttime))
    return pixsize, title


def _construct_ft_arr(cube):
    """Open the model cube and perform a series of roll (both axis) to avoid grid artefact
    (negative fft values).

    Parameters:
    -----------
    `cube` {array}: padded model cube.

    Returns:
    --------
    `ft_arr` {array}: complex array of the Fourier transform of the cube,\n
    `n_ps` {int}: Number of frames,\n
    `n_pix` {int}: Dimensions of one frames,\n

    """
    n_pix = cube.shape[1]
    cube = np.roll(cube, (n_pix // 2, n_pix // 2), axis=(1, 2))

    ft_arr = fft.fft2(cube, workers=multiprocessing.cpu_count())

    i_ps = ft_arr.shape
    n_ps = i_ps[0]

    return ft_arr, n_ps, n_pix


def _extract_pix(hdr, pix_user, scale):
    try:
        unit = hdr["CUNIT1"]
    except KeyError:
        unit = None
    try:
        if unit is None:
            unit = "deg" if "deg" in hdr.comments["CDELT1"] else "rad"
        if unit == "rad":
            pix_size = abs(hdr["CDELT1"]) * scale
        elif unit == "deg":
            pix_size = np.deg2rad(abs(hdr["CDELT1"])) * scale
        elif unit == "mas":
            pix_size = mas2rad(abs(hdr["CDELT1"])) * scale
        else:
            print("Wrong unit in CDELT1 header.")
    except KeyError:
        unit = "rad"
        if pix_user is None:
            cprint(
                "Error: Pixel size not found, please give pix_user [mas].",
                "red",
                file=sys.stderr,
            )
            return None
        pix_size = mas2rad(pix_user) * scale
    return unit, pix_size


def _extract_wave(hdu, hdr, wl_user=None, azim=None, incl=None):
    n_axis = len(hdu[0].data.shape)
    if n_axis == 5:
        n_azim = hdu[0].data.shape[0]
        n_incl = hdu[0].data.shape[1]
        if (azim is None) or (incl is None):
            print(f"Model seems to include azimuth ({n_azim}) and incl ({n_incl})")
            print("-> Specify azim and incl index.")
            return None

        wl_line = hdu[0].header["LAMBDA0"] * 1e-9 * 1e6
        wl_model = hdu[1].data * 1e-9 * 1e6
        wBrg = 0.0005
        close_line = np.abs(wl_model - wl_line) < 3 * wBrg
        delta_wl = np.diff(wl_model[close_line]).mean()
    elif n_axis == 3:
        n_wl = hdu[0].data.shape[0]
        wl0 = hdr.get("CRVAL3", 0)
        delta_wl = hdr.get("CDELT3", 0)
        wl_model = np.linspace(wl0, wl0 + delta_wl * (n_wl - 1), n_wl)
    elif n_axis == 2:
        return np.array(wl_user).astype(float), None, None

    n_wl = hdr.get("NAXIS3", 1)
    if wl_user is not None:
        wl0 = wl_user
        if isinstance(wl_user, float | np.float64):
            n_wl = 1
        elif isinstance(wl_user, list):
            n_wl = len(wl_user)
        else:
            cprint("wl_user have a wrong format: need float or list.", "red")
            return 0
    else:
        if n_axis == 3:
            wl0 = hdr.get("CRVAL3", None)
            if wl0 is None:
                wl0 = hdr.get("WLEN0", None)

        if wl0 is None:
            if wl_user is None:
                cprint("Wavelength not found: need wl_user [µm].", "red")
                return None
            else:
                wl0 = wl_user
                print(f"Wavelenght not found: argument wl_user ({wl_user:2.1f}) is used).")

    return wl_model, delta_wl, n_wl


def model2grid(
    modelfile,
    wl_user=None,
    pix_user=None,
    rotation=0,
    scale=1,
    fliplr=False,
    pad_fact=2,
    method="linear",
    i1=0,
    i2=None,
    light=False,
    window=None,
    azim=None,
    incl=None,
    wline=0.1,
    verbose=True,
):
    """Compute grid class from model as fits file cube.

    Parameters
    ----------
    `modelfile` {str}:
        Name of the model (path),\n
    `wl_user` {array}:
        If not found in the header, wavelength array is required [µm],\n
    `rotation` {int};
        Angle to rotate the model [deg], by default 0,\n
    `scale` {int}:
        Scale the model pixel size, by default 1,\n
    `pad_fact` {int}:
        Padding factor, by default 2.\n

    Returns
    -------
    `grid` {class}:
        class like containing model with keys:\n
            - 'real': 3-d interpolated real part of the complex vis,\n
            - 'imag': 3-d interpolated imaginmary part of the complex vis,\n
            - 'wl': Wavelength grid [m],\n
            - 'freq': Frequencies vector [m-1],\n
            - 'fov': Model field of view [mas],\n
            - 'cube': datacube model,\n
            - 'fft': 2-d fft of the cube,\n
            - 'name': file name of the model.\n
    """
    hdu = fits.open(modelfile)

    try:
        hdr = hdu[0].header
        npix = hdr["NAXIS1"]
    except KeyError:
        hdr = hdu["IMAGE"].header
    starttime = time.time()

    npix = hdr["NAXIS1"]

    # Extract the wavelength table or specify user one
    wl_model, _delta_wl, n_wl = _extract_wave(hdu, hdr, wl_user=wl_user, azim=azim, incl=incl)

    # Extract the pixel size or specify user one
    _unit, pix_size = _extract_pix(hdr, pix_user, scale)

    fov = rad2mas(npix * pix_size)
    if n_wl == 1:
        pass

    unit_wl = hdr.get("CUNIT3", None)
    if unit_wl != "m":
        wl_model *= 1e-6

    if wl_user is not None:
        wl_model = wl_user / 1e6

    try:
        restframe = hdu[0].header["LAMBDA0"] * 1e-9 * 1e6
    except Exception:
        restframe = None

    if restframe is not None:
        inLine = abs(wl_model * 1e6 - restframe) < 10 * wline
    else:
        inLine = [True] * len(wl_model)

    # wl_model = wl_model[inLine]
    if len(hdu[0].data.shape) == 3:
        image_input = hdu[0].data[inLine, :, :]
    if len(hdu[0].data.shape) == 5:
        image_input = hdu[0].data[azim, incl, inLine]
        if fliplr:
            image_input = np.array([np.fliplr(x) for x in image_input])
    else:
        image_input = hdu[0].data

    flux = []
    for _ in image_input:
        flux.append(_.sum())
    flux = np.array(flux) / image_input[0].sum()
    # flux = flux[inLine]

    if len(hdu[0].data.shape) >= 3:
        sns.set_context("talk", font_scale=0.9)
        plt.figure(figsize=(6, 4))
        plt.plot(wl_model * 1e6, flux, ".", label=f"nwl={len(wl_model)}")
        if restframe is not None:
            plt.axvline(
                restframe,
                ls="--",
                lw=2,
                color="green",
                alpha=0.5,
                label=rf"$\lambda$={restframe:2.4f} µm",
            )
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("Norm. spectrum")
        plt.legend(fontsize=12)
        plt.tight_layout(pad=1.01)

    axes_rot = (1, 2)
    if len(image_input.shape) == 2:
        image_input = image_input.reshape([1, image_input.shape[0], image_input.shape[1]])
    mod = rotate(image_input, rotation, axes=axes_rot, reshape=False)
    model_aligned = mod.copy()
    # if fliplr:
    #     model_aligned = np.fliplr(model_aligned)

    npix = model_aligned.shape[1]
    pp = npix * (pad_fact - 1) // 2
    pad_width = [(0, 0), (pp, pp), (pp, pp)]
    mod_pad = np.pad(model_aligned, pad_width=pad_width, mode="edge")
    mod_pad = mod_pad / np.max(mod_pad)
    mod_pad[mod_pad < 1e-20] = 1e-50
    s = np.shape(mod_pad)

    tmp = _construct_ft_arr(mod_pad)
    fft2D = tmp[0]
    n_pix = tmp[2]
    fft2D = np.fft.fftshift(fft2D, axes=(1, 2))
    maxi = np.max(np.abs(fft2D), axis=(1, 2))
    fft2D /= maxi[:, None, None]
    freqVect = np.fft.fftshift(np.fft.fftfreq(n_pix, pix_size))
    if verbose:
        pix_size_grid, title = _print_info_model(wl_model, modelfile, fov, npix, s, starttime)
    pix_size_grid = fov / npix

    if verbose:
        print("start interpolation...")
    t2 = time.time()
    fft2d_real = fft2D.real
    fft2d_imag = fft2D.imag
    if window is not None:
        fft2d_real = np.array([apply_windowing(x, window) for x in fft2d_real])
        fft2d_imag = np.array([apply_windowing(x, window) for x in fft2d_imag])

    if n_wl == 1:
        im3d_real = interp2d(freqVect, freqVect, fft2D.real, kind="cubic")
        im3d_imag = interp2d(freqVect, freqVect, fft2D.imag, kind="cubic")
    elif method == "linear":
        im3d_real = regip(
            (wl_model, freqVect, freqVect),
            [x.T for x in fft2d_real],
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        im3d_imag = regip(
            (wl_model, freqVect, freqVect),
            [x.T for x in fft2d_imag],
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
    else:
        print("Not implemented yet.")
        return None

    p = Path(modelfile)
    modelname = p.stem

    try:
        sign = np.sign(hdr["CDELT1"])
    except KeyError:
        sign = 1.0

    if light:
        grid = {
            "real": im3d_real,
            "imag": im3d_imag,
            "sign": sign,
            "pad_fact": pad_fact,
            "flux": flux,
            "wl": wl_model,
            "psize": pix_size_grid,
        }
    else:
        grid = {
            "real": im3d_real,
            "imag": im3d_imag,
            "sign": sign,
            "wl": wl_model,
            "freq": freqVect,
            "fov": fov,
            "cube": model_aligned,
            "fft": fft2D,
            "name": modelname,
            "pad_fact": pad_fact,
            "flux": flux,
            "npix": s[1],
            "psize": pix_size_grid,
        }
    hdu.close()

    if verbose:
        print("Interpolation is done (%2.2f s)" % (time.time() - t2))
        cprint("-" * len(title) + "\n", "cyan")
    return dict2class(grid)


def _compute_grid_model_chromatic(data, grid, verbose=False):
    nbl = len(data.u)
    ncp = len(data.cp)
    nwl = len(data.wl)

    greal, gimag = grid.real, grid.imag

    l_data = [data] if not isinstance(data, list) else data
    start_time = time.time()
    l_mod_v2, l_mod_cp = [], []

    for data in l_data:
        mod_v2 = np.zeros([nbl, nwl])
        for i in range(nbl):
            um, vm = data.u[i], data.v[i]
            for j in range(nwl):
                wl = data.wl[j]
                x = grid.sign * um / wl
                y = vm / wl
                pts = (wl, x, y)
                v2 = abs(greal(pts) + 1j * gimag(pts)) ** 2 if not data.flag_vis2[i][j] else np.nan
                mod_v2[i, j] = v2

        mod_cp = np.zeros([ncp, nwl])
        for i in range(ncp):
            u1, u2, u3 = (
                grid.sign * data.u1[i],
                grid.sign * data.u2[i],
                grid.sign * data.u3[i],
            )
            v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
            for j in range(nwl):
                wl = data.wl[j]
                u1m, u2m, u3m = u1 / wl, u2 / wl, u3 / wl
                v1m, v2m, v3m = v1 / wl, v2 / wl, v3 / wl
                if not data.flag_cp[i][j]:
                    cvis_1 = greal([wl, u1m, v1m]) + 1j * gimag([wl, u1m, v1m])
                    cvis_2 = greal([wl, u2m, v2m]) + 1j * gimag([wl, u2m, v2m])
                    cvis_3 = greal([wl, u3m, v3m]) + 1j * gimag([wl, u3m, v3m])
                    bispec = np.array(cvis_1) * np.array(cvis_2) * np.array(cvis_3)
                    cp = grid.sign * np.rad2deg(np.arctan2(bispec.imag, bispec.real))
                else:
                    cp = [np.nan]
                mod_cp[i, j] = cp[0]
        l_mod_cp.append(mod_cp)
        l_mod_v2.append(mod_v2)

    if verbose:
        print("Execution time compute_grid_model: %2.3f s" % (time.time() - start_time))
    return l_mod_v2, l_mod_cp


def _compute_grid_model_nochromatic(data, grid, verbose=False):
    starttime = time.time()
    nbl = len(data.u)
    ncp = len(data.cp)
    nwl = len(data.wl)

    greal, gimag = grid.real, grid.imag

    l_data = [data] if not isinstance(data, list) else data
    # start_time = time.time()

    l_mod_v2, l_mod_cp = [], []

    for data in l_data:
        mod_v2 = np.zeros([nbl, nwl])
        for i in range(nbl):
            um = grid.sign * data.u[i] / data.wl
            vm = data.v[i] / data.wl
            mod_v2[i] = [
                abs(grid.real(um[j], vm[j])[0] + 1j * grid.imag(um[j], vm[j])[0]) ** 2
                for j in range(nwl)
            ]

        mod_cp = np.zeros([ncp, nwl])
        for i in range(ncp):
            u1, u2, u3 = (
                grid.sign * data.u1[i],
                grid.sign * data.u2[i],
                grid.sign * data.u3[i],
            )
            v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
            u1m, u2m, u3m = u1 / data.wl, u2 / data.wl, u3 / data.wl
            v1m, v2m, v3m = v1 / data.wl, v2 / data.wl, v3 / data.wl
            greal, gimag = grid.real, grid.imag
            cvis_1 = [greal(u1m[i], v1m[i]) + 1j * gimag(u1m[i], v1m[i]) for i in range(nwl)]
            cvis_2 = [greal(u2m[i], v2m[i]) + 1j * gimag(u2m[i], v2m[i]) for i in range(nwl)]
            cvis_3 = [greal(u3m[i], v3m[i]) + 1j * gimag(u3m[i], v3m[i]) for i in range(nwl)]
            bispec = np.array(cvis_1) * np.array(cvis_2) * np.array(cvis_3)
            cp = np.rad2deg(np.arctan2(bispec.imag, bispec.real))
            mod_cp[i] = np.squeeze(cp)
        l_mod_cp.append(mod_cp)
        l_mod_v2.append(mod_v2)
    if verbose:
        print("Execution time compute_grid_model: %2.3f s" % (time.time() - starttime))
    return l_mod_v2, l_mod_cp


def compute_grid_model(data, grid, verbose=False):
    nwl = len(grid.wl)
    if nwl == 1:
        mod_v2, mod_cp = _compute_grid_model_nochromatic(data, grid, verbose=verbose)
    else:
        mod_v2, mod_cp = _compute_grid_model_chromatic(data, grid, verbose=verbose)
    return mod_v2, mod_cp


def compute_geom_model(data, param, verbose=False):
    """Compute interferometric observables baseline per baseline
    and for all wavelengths (slow)."""
    start_time = time.time()
    l_data = [data] if not isinstance(data, list) else data
    start_time = time.time()
    l_mod_v2, l_mod_cp = [], []
    k = 0
    for data in l_data:
        model_target = select_model(param["model"])
        isValid, log = check_params_model(param)
        if not isValid:
            cprint("\nWrong input parameters for {} model:".format(param["model"]), "green")
            print(log)
            cprint(
                "-" * len("Wrong input parameters for {} model.".format(param["model"])) + "\n",
                "green",
            )
            return None, None

        nbl = len(data.u)
        ncp = len(data.cp)
        mod_v2 = np.zeros_like(data.vis2)
        for i in range(nbl):
            u, v, wl = data.u[i], data.v[i], data.wl
            mod = comput_V2([u, v, wl], param, model_target)
            mod_v2[i, :] = np.squeeze(mod)

        mod_cp = np.zeros_like(data.cp)
        for i in range(ncp):
            u1, u2, u3 = data.u1[i], data.u2[i], data.u3[i]
            v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
            wl2 = data.wl
            X = [u1, u2, u3, v1, v2, v3, wl2]
            tmp = comput_CP(X, param, model_target)
            mod_cp[i, :] = np.squeeze(tmp)

        l_mod_cp.append(mod_cp)
        l_mod_v2.append(mod_v2)
        k += 1

    if verbose:
        print("Execution time compute_geom_model: %2.3f s" % (time.time() - start_time))
    return l_mod_v2, l_mod_cp


def _check_prior(param):
    isValid = True
    prior = param.get("prior", {})
    fitOnly = param.get("fitOnly", [])
    for p in fitOnly:
        value = param[p]
        if p in prior:
            min_value = prior[p][0]
            max_value = prior[p][1]
        else:
            min_value = -np.inf
            max_value = np.inf
        if (value < min_value) | (value > max_value):
            isValid = False
    return isValid


def _compute_geom_model_ind(dataset, param, compute_cp=True, use_flag=False, verbose=False):
    """Compute interferometric observables all at once (including all spectral
    channels) by using matrix computation. `dataset` corresponds to an individual
    fits file (from oimalib.load())."""
    startime = time.time()
    Utable = dataset.u
    Vtable = dataset.v
    Lambda = dataset.wl
    nobs = len(Utable) * len(Lambda)
    model_target = select_model(param["model"])
    # isValid = check_params_model(param)[0]

    # Find telescope down or problem with one baseline (only nan)
    try:
        exclude_tel = _check_good_tel([dataset], verbose=False)[0]
    except ValueError:
        exclude_tel = []

    if not use_flag:
        exclude_tel = []

    # Compute index of good bl and cp
    if len(exclude_tel) == 0:
        good_cp = np.arange(len(dataset.u1))
        good_bl = np.arange(len(dataset.u))
    else:
        good_cp = []
        good_bl = []
        cp_name = dataset.cpname
        for i in range(len(cp_name)):
            for bad in exclude_tel:
                if bad not in cp_name[i]:
                    good_cp.append(i)
        for i in range(len(Utable)):
            for bad in exclude_tel:
                if bad not in dataset.blname[i]:
                    good_bl.append(i)

    # Select only for good baselines (deal if one telescope down for instance)
    Utable = Utable.take(good_bl)
    Vtable = Vtable.take(good_bl)

    # Add try to use the visPwhl model that requires verbose
    try:
        cvis = model_target(Utable, Vtable, Lambda, param, verbose=verbose).T
    except TypeError:
        cvis = model_target(Utable, Vtable, Lambda, param).T
    # Compute quantities
    vis2 = np.abs(cvis) ** 2
    vis_amp = np.abs(cvis)
    vis_phi = np.angle(cvis)

    if compute_cp:
        # Compute bispectrum and closure phases (same for .T)
        u1, u2, u3 = dataset.u1, dataset.u2, dataset.u3
        v1, v2, v3 = dataset.v1, dataset.v2, dataset.v3
        V1 = model_target(u1.take(good_cp), v1.take(good_cp), Lambda, param)
        V2 = model_target(u2.take(good_cp), v2.take(good_cp), Lambda, param)
        V3 = model_target(u3.take(good_cp), v3.take(good_cp), Lambda, param)
        bispectrum = V1 * V2 * V3
        cp = np.rad2deg(np.arctan2(bispectrum.imag, bispectrum.real)).T
    else:
        cp = None
    endtime = time.time()

    # isValid = check_params_model(param)[0]

    # valid_prior = _check_prior(param)
    # if not (isValid & valid_prior):
    #     vis2 = np.zeros_like(vis2)
    #     vis2[vis2 == 0] = np.nan
    #     cp = np.zeros_like(cp)
    #     cp[cp == 0] = np.nan
    #     vis_amp = np.zeros_like(vis_amp)
    #     vis_amp[vis_amp == 0] = np.nan
    #     vis_phi = np.zeros_like(vis_phi)
    #     vis_phi[vis_phi == 0] = np.nan
    #     cvis = np.zeros_like(cvis)
    #     cvis[cvis == 0] = np.nan

    if verbose:
        print(f"Time to compute {nobs:d} points = {endtime - startime:2.2f} s")
    mod = {
        "vis2": vis2,
        "cp": cp,
        "dvis": vis_amp,
        "dphi": vis_phi,
        "cvis": cvis,
        "good_bl": good_bl,
        "good_cp": good_cp,
    }
    return mod


def compute_geom_model_fast(data, param, ncore=1, compute_cp=True, use_flag=False, verbose=False):
    """Compute interferometric observables using the matrix method (faster)
    for a list of data (type(data) == list) or only one file (type(data) ==
    dict). The multiple dataset can be computed in parallel if `ncore` > 1."""
    l_data = [data] if not isinstance(data, list) else data
    start_time = time.time()

    if ncore == 1:
        result_list = [
            _compute_geom_model_ind(
                x,
                param=param,
                compute_cp=compute_cp,
                use_flag=use_flag,
            )
            for x in l_data
        ]
    else:
        pool = multiprocessing.Pool(processes=ncore)
        prod = partial(
            _compute_geom_model_ind,
            param=param,
            compute_cp=compute_cp,
            use_flag=use_flag,
        )
        result_list = pool.map(prod, l_data)
        pool.close()
    etime = time.time() - start_time
    if verbose:
        print(f"Execution time compute_geom_model_fast: {etime:2.3f} s")
    return result_list


def decoratortimer(decimal):
    def decoratorfunction(f):
        def wrap(*args, **kwargs):
            time1 = time.monotonic()
            result = f(*args, **kwargs)
            time2 = time.monotonic()
            print(
                "{:s} function took {:.{}f} ms".format(
                    f.__name__, ((time2 - time1) * 1000.0), decimal
                )
            )
            return result

        return wrap

    return decoratorfunction


@decoratortimer(2)
def _compute_dobs_grid_bl(
    ibl,
    d,
    grid,
    obs_wl=True,
    w_line=0.0005,
    p_line=2.166128,
    scale=False,
):
    """Compute the differential observable (dvis, dphi) from the
    pre-computed interpolated grid (from chromatic models).
    """
    # Extract u-v coordinates from data at specific baseline (ibl)
    um = d.u[ibl]
    vm = d.v[ibl]

    wlm = d.wl if obs_wl else np.linspace(d.wl[0], d.wl[-1], 100)

    # Interpolated function (real and imaginary parts)
    greal, gimag = grid.real, grid.imag

    # Extract the specific points (from the data)
    pts = (wlm, um / wlm, vm / wlm)
    cvis = greal(pts) + 1j * gimag(pts)

    # Compute the reference vis and phase from the continuum
    cont = np.abs(wlm * 1e6 - p_line) >= 2 * w_line
    vis_ref = np.mean(abs(cvis[cont & ~np.isnan(cvis)]))
    phi_ref = np.rad2deg(np.mean(np.angle(cvis[cont & ~np.isnan(cvis)])))

    # Compute observables
    dvis = abs(cvis) / vis_ref
    dphi = np.rad2deg(np.angle(cvis)) - phi_ref
    vis2 = abs(cvis) ** 2

    # Compute flux on the data resolution
    res_obs = np.diff(d.wl).mean()
    res_model = np.diff(grid.wl).mean()
    scale_resol = res_obs / res_model
    flux_scale = gaussian_filter1d(grid.flux, sigma=scale_resol) if scale else grid.flux
    fct_flux = interp1d(grid.wl, flux_scale, kind="cubic", bounds_error=False)
    flux_obs = fct_flux(wlm)

    output = {
        "cvis": cvis,
        "dvis": dvis,
        "dphi": dphi,
        "vis2": vis2,
        "cont": cont,
        "wl": wlm * 1e6,
        "p_line": p_line,
        "flux": flux_obs,
        "flux_model": grid.flux,
        "wl_model": grid.wl * 1e6,
        "ibl": ibl,
    }

    return dict2class(output)


def combine_grid_geom_model_image(
    wl,
    grid,
    param,
    ampli_factor=1,
    rotation=0,
    fh=0,
    fc=0,
    fmag=1,
    fov=3,
    npts=256,
    verbose=False,
):
    if isinstance(wl, list):
        wl = np.array(wl)
    elif isinstance(wl, float | np.float64):
        wl = np.array([wl])

    fov = mas2rad(fov)
    bmax = (wl / fov) * npts
    maxX = rad2mas(wl * npts / bmax) / 2.0
    xScales = np.linspace(0, 2 * maxX, npts) - maxX
    pixel_size = rad2mas(fov) / npts
    extent_ima = np.array((xScales.max(), xScales.min(), xScales.min(), xScales.max()))
    pixel_size = rad2mas(fov) / npts  # Pixel size of the image [mas]

    pixel_size_grid = grid.psize

    if verbose:
        print(f"The pixel size need to be > {pixel_size_grid:2.4f} ({pixel_size:2.4f})")
    # # Creat UV coord
    UVTable = UVGrid(bmax, npts) / 2.0  # Factor 2 due to the fft
    Utable = UVTable[:, 0]
    Vtable = UVTable[:, 1]

    model_target = select_model(param["model"])

    vis_disk = model_target(Utable, Vtable, wl, param)

    greal, gimag = grid.real, grid.imag
    # Extract the specific points (from the data)
    pts = (wl, Utable / wl, Vtable / wl)
    vis_mag = greal(pts) + 1j * gimag(pts)

    index_image = np.abs(grid.wl - wl).argmin()

    # Amplify spectral line (mimic temperature increase)

    fs = 1 - fh - fc
    mm = grid.flux.copy() - 1
    mm[mm > 0] = mm[mm > 0] * ampli_factor
    mm += 1
    fmag = fs * mm
    ftot = fmag + fh + fc
    # ftot = gaussian_filter1d(ftot, sigma=23.0 / 2.355)

    fmag_im = fmag[index_image]
    ftot_im = ftot[index_image]

    vis = (fmag_im * vis_mag + fc * vis_disk) / (ftot_im)

    if verbose:
        print(
            f"oimalibsphere contribution = "
            f"{100 * fmag_im / (fmag_im + fh + fc):2.1f} % "
            f"(lcr = {mm[index_image]:2.2f})"
        )
    fwhm_apod = 5e4
    # Apodisation
    x, y = np.meshgrid(range(npts), range(npts))
    freq_max = rad2arcsec(bmax / wl) / 2.0
    pix_vis = 2 * freq_max / npts
    freq_map = np.sqrt((x - (npts / 2.0)) ** 2 + (y - (npts / 2.0)) ** 2) * pix_vis / 4

    x = np.squeeze(np.linspace(0, 1.5 * np.sqrt(freq_max**2 + freq_max**2), npts))
    y = np.squeeze(np.exp(-(x**2) / (2 * (fwhm_apod / 2.355) ** 2)))

    f = interp1d(x, y)
    img_apod = f(freq_map.flat).reshape(freq_map.shape)

    im_vis = vis.reshape(npts, -1) * img_apod
    fftVis = np.fft.ifft2(im_vis)
    image = np.fft.fftshift(abs(fftVis))
    tmp = np.fliplr(image)
    image_orient = tmp / np.sum(tmp)
    image_orient *= ftot_im

    if rotation != 0:
        image_orient = rotate(image, rotation, reshape=False)

    # image_orient = cv.GaussianBlur(image_orient, (7, 7), 0)
    return image_orient, pixel_size, extent_ima, ftot


def compute_pdf_chi2(data, param, fitOnly=None, use_flag=True, normalizeErrors=False):
    """Compute panda dataframe from a dataset (splitted between vis2 and cp).
    Use those pdf to compute the individual chi2 (compared to models stands by
    param) and global.
    Parameters:
    -----------
    `data` {list}:
        List of dataset from load(),\n
    `param` {dict}:
        Parameters for the models,\n
    `fitOnly` {list}:
        List of fitted parameters (if any) to compute the degree of freedom,\n
    `use_flag` {bool}:
        If True, the flags are used.

    Outputs:
    --------
    `df_v2` {pandas}:
        Panda dataframe of the vis2,\n
    `df_cp` {pandas}:
        Panda dataframe of the cp,\n
    `chi2_global` {float}:
        chi2 global for the model,\n
    `chi2_vis2` {float}:
        chi2 for the vis2,\n
    `chi2_cp` {float}:
        chi2 for the cp.
    """
    l_mod = compute_geom_model_fast(data, param, ncore=4)
    mod_cp, mod_v2 = [], []
    for i in range(len(l_mod)):
        mod_cp.append(l_mod[i]["cp"])
    for i in range(len(l_mod)):
        mod_v2.append(l_mod[i]["vis2"])

    if fitOnly is None:
        fitOnly = []
    input_keys_v2 = [
        "vis2",
        "e_vis2",
        "freq_vis2",
        "wl",
        "blname",
        "set",
        "flag_vis2",
    ]
    input_keys_cp = ["cp", "e_cp", "freq_cp", "wl", "cpname", "set", "flag_cp"]

    dict_obs_cp = {}
    for k in input_keys_cp:
        dict_obs_cp[k] = []
    dict_obs_v2 = {}
    for k in input_keys_v2:
        dict_obs_v2[k] = []

    nobs = 0
    for d in data:
        for k in input_keys_cp:
            nbl = d.cp.shape[0]
            nwl = d.cp.shape[1]
            if k == "wl":
                for _ in range(nbl):
                    dict_obs_cp[k].extend(np.round(d[k] * 1e6, 3))
            elif k == "cpname":
                for j in range(nbl):
                    for _ in range(nwl):
                        dict_obs_cp[k].append(d[k][j])
            elif k == "set":
                for _ in range(nbl):
                    for _ in range(nwl):
                        dict_obs_cp[k].append(nobs)
            else:
                dict_obs_cp[k].extend(d[k].flatten())
        for k in input_keys_v2:
            nbl = d.vis2.shape[0]
            nwl = d.vis2.shape[1]
            if k == "wl":
                for _ in range(nbl):
                    dict_obs_v2[k].extend(np.round(d[k] * 1e6, 3))
            elif k == "blname":
                for j in range(nbl):
                    for _ in range(nwl):
                        dict_obs_v2[k].append(d[k][j])
            elif k == "set":
                for _ in range(nbl):
                    for _ in range(nwl):
                        dict_obs_v2[k].append(nobs)
            else:
                dict_obs_v2[k].extend(d[k].flatten())
        nobs += 1

    dict_obs_cp["mod"] = np.array(mod_cp).flatten()
    dict_obs_cp["res"] = (dict_obs_cp["cp"] - dict_obs_cp["mod"]) / dict_obs_cp["e_cp"]
    dict_obs_v2["mod"] = np.array(mod_v2).flatten()
    dict_obs_v2["res"] = (dict_obs_v2["vis2"] - dict_obs_v2["mod"]) / dict_obs_v2["e_vis2"]

    if use_flag:
        flag = np.array(dict_obs_cp["flag_cp"])
        flag_nan = np.isnan(np.array(dict_obs_cp["cp"]))
        for k in dict_obs_cp:
            dict_obs_cp[k] = np.array(dict_obs_cp[k])[~flag & ~flag_nan]
        flag2 = np.array(dict_obs_v2["flag_vis2"])
        flag_nan2 = np.isnan(np.array(dict_obs_v2["vis2"]))
        for k in dict_obs_v2:
            dict_obs_v2[k] = np.array(dict_obs_v2[k])[~flag2 & ~flag_nan2]

    df_v2 = pd.DataFrame(dict_obs_v2)
    df_cp = pd.DataFrame(dict_obs_cp)

    d_freedom = len(fitOnly)
    chi2_vis2_full = np.sum((df_v2["vis2"] - df_v2["mod"]) ** 2 / (df_v2["e_vis2"]) ** 2)
    chi2_vis2 = chi2_vis2_full / (len(df_v2["e_vis2"]) - (d_freedom - 1))

    chi2_cp_full = np.sum((df_cp["cp"] - df_cp["mod"]) ** 2 / (df_cp["e_cp"]) ** 2)
    chi2_cp = chi2_cp_full / (len(df_cp["e_cp"]) - (d_freedom - 1))

    nv2 = len(df_v2["vis2"])
    ncp = len(df_cp["cp"])
    nobs = nv2 + ncp
    obs = np.zeros(nobs)
    e_obs = np.zeros(nobs)
    all_mod = np.zeros(nobs)

    norm_v2, norm_cp = 1, 1
    if normalizeErrors:
        if nv2 > ncp:
            norm_v2 = np.sqrt(nv2 / ncp)
            norm_cp = 1.0
        else:
            norm_v2 = 1
            norm_cp = np.sqrt(ncp / nv2)

    # print(0.02 * norm_v2)
    for i in range(len(df_v2["vis2"])):
        obs[i] = df_v2["vis2"][i]
        e_obs[i] = df_v2["e_vis2"][i] * norm_v2
        all_mod[i] = df_v2["mod"][i]
    for i in range(len(df_cp["cp"])):
        obs[i + nv2] = df_cp["cp"][i]
        e_obs[i + nv2] = df_cp["e_cp"][i] * norm_cp
        all_mod[i + nv2] = df_cp["mod"][i]
    chi2_global = np.sum((obs - all_mod) ** 2 / (e_obs) ** 2) / (nobs - (d_freedom - 1))
    return df_v2, df_cp, chi2_global, chi2_vis2, chi2_cp


def plot_chi2_lk(dataset, fit, *, fitOnly, cons=True, norm=False, verbose=False, display=True):
    """Plot chi2 curve of the ratio a/ak for the lazareff model. The return
    results are the best fitted w and the upper and lower limit. The search of
    minimum chi2 can be:
        - conservative (`cons`=True), the reduced chi2 is used and normalized by
        the sqrt(chi2_r), this approach corresponds to a fully correlated data
        points and retrieves usually over estimated error bars (but the
        normalization allows us to mitigate this effects),
        - optimistic (`cons`=False), the chi2 is used and not normalized. This
        approach is used when the data points are uncorrelated (independant).
       Generally under estimates the error bars."""
    l_chi2 = []
    l_w = []
    l_lk = np.linspace(-1, 1, 100)
    input_param = fit["best"].copy()

    ind_stat = 1
    if not cons:
        ind_stat = 5

    for lk in l_lk:
        input_param["lk"] = lk
        stat = oimalib.plot_residuals(
            dataset,
            input_param,
            fitOnly=fitOnly,
            normalizeErrors=norm,
            verbose=False,
            display=False,
        )

        ar = 10 ** input_param["la"] / (np.sqrt(1 + 10 ** (2 * input_param["lk"])))
        ak = ar * (10 ** input_param["lk"])
        a = (ar**2 + ak**2) ** 0.5
        w = ak / a
        if verbose:
            print(
                f"lk = {lk:2.2f} (chi2 = {stat[ind_stat]:2.2f}): "
                f"w = {w:2.2f}, ar = {ar:2.2f}, ak = {ak:2.2f}"
            )
        l_w.append(w)
        l_chi2.append(stat[ind_stat])
    l_w = np.array(l_w)
    l_chi2 = np.array(l_chi2)
    # l_chi2 -= l_chi2.min()
    # l_chi2 += 1

    cons_w = l_w[l_chi2 <= l_chi2.min() + 1]
    w_best = l_w[np.argmin(l_chi2)] * 1e2

    up_w = cons_w[-1]

    arg_up = np.where(up_w == l_w)[0][0]

    w_limit = l_w[arg_up] * 1e2

    norm_value = 1
    name_norm = ""
    label = r"$\chi^2$"
    if cons:
        name_norm = r" ($\sqrt{\chi^2_r}$)"
        label = r"$\chi^2_r$"
        norm_value = np.sqrt(l_chi2.min())

    w_up = (w_limit - w_best) / norm_value
    w_low = abs(w_best - 10) / norm_value

    if display:
        plt.figure()
        plt.title(
            rf"$w = {w_best:2.0f}^{{+{w_up:2.0f}}}_{{-{abs(w_low):2.0f}}}$ "
            rf"(norm = {norm_value:2.1f}{name_norm})"
        )
        plt.plot(1e2 * l_w, l_chi2)
        plt.plot(
            w_best,
            l_chi2.min(),
            "ro",
            label=f"$w_{{best}}$ = {w_best:2.0f} %",
        )
        plt.plot(
            w_limit,
            l_chi2[arg_up],
            "bo",
            label=f"$w_{{limit}}$ = {w_limit:2.0f} %",
        )
        plt.xlabel("$w$ [%]")
        plt.ylabel(label)
        plt.legend()
        plt.tight_layout()

    return w_best / 1e2, w_up, w_low
