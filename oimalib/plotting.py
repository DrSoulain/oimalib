"""
@author: Anthony Soulain (University of Sydney)
-----------------------------------------------------------------
OIMALIB: Optical Interferometry Modelisation and Analysis Library
-----------------------------------------------------------------

Set of function to plot oi data, u-v plan, models, etc.
-----------------------------------------------------------------
"""
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c as c_light
from scipy.interpolate import interp1d
from scipy.ndimage import center_of_mass
from scipy.ndimage import rotate
from termcolor import cprint
from uncertainties import unumpy

from oimalib.complex_models import visGaussianDisk
from oimalib.fitting import check_params_model
from oimalib.fitting import format_obs
from oimalib.fitting import get_mcmc_results
from oimalib.fitting import model_flux
from oimalib.fitting import model_flux_red_abs
from oimalib.fitting import model_pcshift
from oimalib.fitting import select_model
from oimalib.fourier import UVGrid
from oimalib.modelling import compute_geom_model_fast
from oimalib.tools import find_nearest
from oimalib.tools import hide_xlabel
from oimalib.tools import mas2rad
from oimalib.tools import normalize_continuum
from oimalib.tools import plot_vline
from oimalib.tools import rad2arcsec
from oimalib.tools import rad2mas

dic_color = {
    "A0-B2": "#928a97",  # SB
    "A0-D0": "#7131CC",
    "A0-C1": "#ffc93c",
    "B2-C1": "indianred",
    "B2-D0": "#086972",
    "C1-D0": "#3ec1d3",
    "D0-G2": "#f37735",  # MB
    "D0-J3": "#4b86b4",
    "D0-K0": "#CC9E3D",
    "G2-J3": "#d11141",
    "G2-K0": "#A6DDFF",
    "J3-K0": "#00b159",
    "A0-G1": "#96d47c",  # LB
    "A0-J2": "#f38181",
    "A0-J3": "#1f5f8b",
    "G1-J2": "#a393eb",
    "G1-J3": "#eedf6b",
    "J2-J3": "c",
    "J2-K0": "c",
    "A0-K0": "#8d90a1",
    "G1-K0": "#ffd100",
    "U1-U2": "#82b4bb",
    "U2-U3": "#255e79",
    "U3-U4": "#5ec55e",
    "U2-U4": "#ae3c60",
    "U1-U3": "#e35d5e",
    "U1-U4": "#f1ca7f",
}

err_pts_style = {
    "linestyle": "None",
    "capsize": 1,
    "marker": ".",
    "elinewidth": 0.5,
    "alpha": 1,
    "ms": 5,
}

err_pts_style_pco = {
    "linestyle": "None",
    "capsize": 1,
    "ecolor": "#364f6b",
    "mec": "#364f6b",
    "marker": ".",
    "elinewidth": 0.5,
    "alpha": 1,
    "ms": 11,
}


# Plot data and models (mainly V2 and CP)


def plot_oidata(
    tab,
    use_flag=True,
    mod_v2=None,
    mod_cp=None,
    cp_max=200,
    v2min=0,
    v2max=1.2,
    log=False,
    is_nrm=False,
    color=False,
    plot_amp=True,
    force_freq=None,
    mega=False,
):
    """
    Plot the interferometric data (and the model if required).

    Parameters:
    -----------
    `tab` {list}:
        list of data from amical.load(),\n
    `use_flag` {boolean}:
        If True, use flag from the oifits file (selected if select_data()
        was used before),\n
    `mod_v2`, `mod_cp` {array}:
        V2 and CP model computed from grid (`compute_grid_model()`) or
        the analytical model (`compute_geom_model()`),\n
    `cp_max` {float}:
        Limit maximum along Y-axis of CP data plot,\n
    `v2min`, `v2max` {float}:
        Limits along Y-axis of V2 data plot,\n
    `log` {boolean}:
        If True, display the Y-axis of the V2 plot in log scale,\n
    `is_nrm` {boolean}:
        If True, data come from NRM data,\n
    `plot_amp` {boolean}:
        If True, amplitude visibility are plotted instead of V2,\n
    `force_freq` {list}, optionnal:
        Limits of the spatial frequencies [arcsec-1],\n
    `mega` {boolean}:
        If True, spatial frequencies are set in Mλ.
    """

    if (type(tab) == list) | (type(tab) == np.ndarray):
        data = tab[0]
    else:
        data = tab
        tab = [tab]

    ms_model = 2
    dic_color = _update_color_bl(tab)

    array_name = data.info["Array"]
    l_fmin, l_fmax = [], []

    list_triplet = []
    for _ in tab:
        for i in range(len(_.cpname)):
            list_triplet.append(_.cpname[i])
    list_triplet = np.array(list_triplet)

    for _ in tab:
        tfmax = _.freq_vis2.flatten().max()
        tmp = _.freq_vis2.flatten()
        tfmin = tmp[tmp != 0].min()
        l_fmax.append(tfmax)
        l_fmin.append(tfmin)

    ff = 1
    if mega:
        ff = (1.0 / mas2rad(1000)) / 1e6
    l_fmin = np.array(l_fmin) * ff
    l_fmax = np.array(l_fmax) * ff

    if len(l_fmin) == 1:
        fmin = l_fmin[0]
    else:
        fmin = np.min(l_fmin)
    fmax = l_fmax.max()

    ncp_master = len(set(list_triplet))

    if not plot_amp:
        ylabel = r"V$^2$"
    else:
        ylabel = r"Vis. Amp."

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
    list_bl = []

    # PLOT VIS2 DATA AND MODEL IF ANY (mod_v2)
    # ----------------------------------------
    ndata = len(tab)
    for j in range(ndata):
        data = tab[j]
        nwl = len(data.wl)
        nbl = data.vis2.shape[0]
        for i in range(nbl):
            if use_flag:
                sel_flag = np.invert(data.flag_vis2[i])
            else:
                sel_flag = [True] * nwl

            freq_vis2 = data.freq_vis2[i][sel_flag]
            if not plot_amp:
                vis2 = data.vis2[i][sel_flag]
                e_vis2 = data.e_vis2[i][sel_flag]
            else:
                vis2 = data.dvis[i][sel_flag]
                e_vis2 = data.e_dvis[i][sel_flag]

            base, label = data.blname[i], ""
            wave = data.wl[sel_flag]

            if len(vis2[~np.isnan(vis2)]) == 0:
                continue

            if base not in list_bl:
                label = base
                list_bl.append(base)

            if mega:
                freq_vis2 = freq_vis2 * (1.0 / mas2rad(1000)) / 1e6

            if is_nrm:
                p_color = "tab:blue"
                ax1.errorbar(
                    freq_vis2,
                    vis2,
                    yerr=e_vis2,
                    ecolor="lightgray",
                    color=p_color,
                    marker=".",
                    ms=6,
                    elinewidth=1,
                )
            else:
                p_color = dic_color[base]
                if len(vis2) == 1:
                    ax1.errorbar(
                        freq_vis2,
                        vis2,
                        yerr=e_vis2,
                        color=p_color,
                        label=label,
                        **err_pts_style,
                    )
                else:
                    if color:
                        sc = ax1.scatter(freq_vis2, vis2, c=wave * 1e6, s=3)
                    else:
                        ebar = ax1.plot(
                            freq_vis2, vis2, color=p_color, ls="-", lw=1, label=label
                        )
                        ax1.fill_between(
                            freq_vis2,
                            vis2 - e_vis2,
                            vis2 + e_vis2,
                            color=p_color,
                            alpha=0.3,
                        )

            if mod_v2 is not None:
                if not plot_amp:
                    mod = mod_v2[j][i][sel_flag]
                else:
                    mod = mod_v2[j][i][sel_flag] ** 0.5
                ax1.plot(
                    freq_vis2,
                    mod,
                    marker="x",
                    color="k",
                    alpha=0.7,
                    zorder=100,
                    lw=1,
                    ms=ms_model,
                    ls="",
                )

    if mod_v2 is not None:
        ax1.plot(
            -1,
            -1,
            marker="x",
            color="k",
            alpha=0.7,
            zorder=100,
            lw=1,
            ms=ms_model,
            ls="",
            label="model",
        )

    if log:
        ax1.set_yscale("log")
        ax1.set_ylim(v2min, v2max)
    else:
        ax1.set_ylim(v2min, v2max)

    offset = 0
    if data.info["Array"] == "CHARA":
        offset = 150

    if force_freq is not None:
        fmin, fmax = force_freq[0], force_freq[1]
    ax1.set_xlim(fmin - 2, fmax + 2 + offset)
    if not color:
        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax1.legend(handles, labels, fontsize=7)
    else:
        cax = fig.add_axes(
            [
                ax1.get_position().x1 * 0.83,
                ax1.get_position().y1 * 1.03,
                0.2,
                ax1.get_position().height * 0.04,
            ]
        )
        cb = plt.colorbar(sc, cax=cax, orientation="horizontal")
        cb.ax.set_title(r"$\lambda$ [µm]", fontsize=9)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.grid(alpha=0.3)

    set_cp = 1
    ms_model = 2
    # PLOT CP DATA AND MODEL IF ANY (mod_cp)
    # --------------------------------------
    ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)
    if not is_nrm:
        if array_name == "CHARA":
            fontsize = 5
            ax2.set_prop_cycle("color", plt.cm.turbo(np.linspace(0, 1, ncp_master)))
        elif array_name == "VLTI":
            fontsize = 7
            if set_cp == 1:
                ax2.set_prop_cycle(
                    "color",
                    [
                        "#eabd6f",
                        "#fa9583",
                        "#3a6091",
                        "#79ab8e",
                        "#82b4bb",
                        "#ae3c60",
                        "#eabd6f",
                        "#96d47c",
                    ],
                )
            elif set_cp == 2:
                ax2.set_prop_cycle(
                    "color", ["#79ab8e", "#5c95a8", "#fa9583", "#263a55"]
                )
            else:
                ax2.set_prop_cycle("color", plt.cm.turbo(np.linspace(0, 1, ncp_master)))

        else:
            fontsize = 7
            ax2.set_prop_cycle("color", plt.cm.Set2(np.linspace(0, 1, 8)))
        color_cp = None
    else:
        color_cp = "tab:blue"

    color_cp_dic = {}
    list_triplet = []
    for j in range(len(tab)):
        data = tab[j]
        ncp = data.cp.shape[0]
        color_cp = None
        for i in range(ncp):
            if use_flag:
                sel_flag = np.invert(data.flag_cp[i])
            else:
                sel_flag = np.array([True] * nwl)

            freq_cp = data.freq_cp[i][sel_flag]
            cp = data.cp[i][sel_flag]
            e_cp = data.e_cp[i][sel_flag]
            wave = data.wl[sel_flag]

            if mega:
                freq_cp = freq_cp * (1.0 / mas2rad(1000)) / 1e6

            if len(cp[~np.isnan(cp)]) == 0:
                continue

            dic_index = _create_match_tel(data)
            b1 = dic_index[data.index_cp[i][0]]
            b2 = dic_index[data.index_cp[i][1]]
            b3 = dic_index[data.index_cp[i][2]]
            triplet = f"{b1}-{b2}-{b3}"

            label = ""
            if triplet not in list_triplet:
                label = triplet
                list_triplet.append(triplet)

            if triplet in color_cp_dic.keys():
                color_cp = color_cp_dic[triplet]

            if not color:
                ebar = ax2.errorbar(
                    freq_cp, cp, yerr=e_cp, label=label, color=color_cp, **err_pts_style
                )
                if triplet not in color_cp_dic.keys():
                    color_cp_dic[triplet] = ebar[0].get_color()
            else:
                ax2.scatter(freq_cp, cp, c=wave, s=3)

            if mod_cp is not None:
                mod = mod_cp[j][i][sel_flag]
                ax2.plot(
                    freq_cp,
                    mod,
                    marker="x",
                    ls="",
                    color="k",
                    ms=ms_model,
                    zorder=100,
                    alpha=0.7,
                )

    if (is_nrm) | (not color):
        ax2.legend(fontsize=fontsize)
    ax2.set_ylabel(r"CP [deg]", fontsize=12)
    ax2.set_xlabel(r"Sp. Freq [arcsec$^{-1}$]", fontsize=12)
    ax2.set_ylim(-cp_max, cp_max)
    ax2.set_xlim(fmin - 2, fmax + 2 + offset)
    ax2.grid(alpha=0.2)
    plt.tight_layout()
    return fig


def plot_uv(tab, bmax=150, rotation=0):
    """
    Plot the u-v coverage.

    Parameters:
    -----------

    `tab` {list}:
        list containing of data from OiFile2Class function (size corresponding to the number of files),\n
    `bmax` {float}:
        Limits of the plot [Mlambda],\n
    `rotation` {float}:
        Rotate the u-v plan.
    """

    if (type(tab) == list) or (type(tab) == np.ndarray):
        wl_ref = np.mean(tab[0].wl) * 1e6
    else:
        wl_ref = np.mean(tab.wl) * 1e6

    bmax = bmax / wl_ref

    fig = plt.figure(figsize=(6.5, 6))
    ax = plt.subplot(111)

    ax2 = ax.twinx()
    ax3 = ax.twiny()

    _plot_uvdata_coord(tab, ax=ax, rotation=rotation)

    ax.patch.set_facecolor("#f7f9fc")
    ax.set_xlim([-bmax, bmax])
    ax.set_ylim([-bmax, bmax])
    ax2.set_ylim([-bmax * wl_ref, bmax * wl_ref])
    ax3.set_xlim([-bmax * wl_ref, bmax * wl_ref])
    plt.grid(alpha=0.5, linestyle=":")
    ax.axvline(0, linewidth=1, color="gray", alpha=0.2)
    ax.axhline(0, linewidth=1, color="gray", alpha=0.2)
    ax.set_xlabel(r"U [M$\lambda$]")
    ax.set_ylabel(r"V [M$\lambda$]")
    ax2.set_ylabel("V [m] - East", color="#007a59")
    ax3.set_xlabel("U [m] (%2.2f µm) - North" % wl_ref, color="#007a59")
    ax2.tick_params(axis="y", colors="#007a59")
    ax3.tick_params(axis="x", colors="#007a59")
    ax.legend(fontsize=7)
    plt.subplots_adjust(
        top=0.97, bottom=0.09, left=0.11, right=0.93, hspace=0.2, wspace=0.2
    )
    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_residuals(
    data,
    param,
    fitOnly=None,
    use_flag=True,
    hue=None,
    save_dir=None,
    name=None,
    verbose=True,
):
    """
    Plot the comparison between data vs model and the corresponding residuals
    [in σ] for the V2 and CP.

    Parameters:
    -----------
    `data` {list, dict}:
        data or list of data from amical.load(),\n
    `param` {dict}:
        Parameters of the model,\n
    `fitOnly` {list}:
        List of fitted parameters to compute the degree of freedom,\n
    `use_flag` {boolean}:
        If True, use flag from the oifits file (selected if select_data()
        was used before),\n
    `hue` {str}:
        Key to be displayed in color (e.g.: 'wl'),\n
    `save_dir` {str}, optionnal:
        If set, the figure is saved in `save_dir` as *`name`*.png,\n
    """

    if name is None:
        name = ""
    sns.set_theme(color_codes=True)
    if fitOnly is None:
        print("Warning: FitOnly is None, the degree of freedom is set to 0.\n")
        fitOnly = []
    else:
        if len(fitOnly) == 0:
            print("Warning: FitOnly is empty, the degree of freedom is set to 0.\n")

    if type(data) is not list:
        data = [data]

    param_plot = {
        "data": data,
        "param": param,
        "fitOnly": fitOnly,
        "hue": hue,
        "use_flag": use_flag,
    }
    df_cp, chi2_cp, chi2_cp_full, mod_cp = _plot_cp_residuals(**param_plot)
    if save_dir is not None:
        plt.savefig(save_dir + "residuals_CP_%sfit.png" % name, dpi=300)
    df_v2, chi2_vis2, chi2_vis2_full, mod_v2 = _plot_v2_residuals(**param_plot)
    if save_dir is not None:
        plt.savefig(save_dir + "residuals_V2_%sfit.png" % name, dpi=300)

    d_freedom = len(fitOnly)

    nv2 = len(df_v2["vis2"])
    ncp = len(df_cp["cp"])
    nobs = nv2 + ncp
    obs = np.zeros(nobs)
    e_obs = np.zeros(nobs)
    all_mod = np.zeros(nobs)

    for i in range(len(df_v2["vis2"])):
        obs[i] = df_v2["vis2"][i]
        e_obs[i] = df_v2["e_vis2"][i]
        all_mod[i] = df_v2["mod"][i]
    for i in range(len(df_cp["cp"])):
        obs[i + nv2] = df_cp["cp"][i]
        e_obs[i + nv2] = df_cp["e_cp"][i]
        all_mod[i + nv2] = df_cp["mod"][i]

    chi2_global = np.sum((obs - all_mod) ** 2 / (e_obs) ** 2) / (nobs - (d_freedom - 1))
    title = "Statistic of the model %s" % param["model"]
    if verbose:
        print(title)
        print("-" * len(title))
        print(f"χ² = {chi2_global:2.2f} (V² = {chi2_vis2:2.1f}, CP = {chi2_cp:2.1f})")
    return chi2_global, chi2_vis2, chi2_cp, mod_v2, mod_cp, chi2_vis2_full, chi2_cp_full


def plot_image_model(
    wl,
    param,
    base_max=130,
    npts=128,
    fov=400,
    blm=130,
    fwhm_apod=1e2,
    hamming=True,
    cont=False,
    p=0.5,
    data=None,
    apod=False,
    corono=False,
    expert_plot=False,
    verbose=False,
):
    """
    Compute and plot the image for the model based on `param` dictionnary.

    Parameters:
    -----------
    `wl` {float}:
        Wavelength of the observation [m],\n
    `param` {dict}:
        Dictionnary of the model parameters,\n
    `base_max` {float}:
        Maximum baseline (of single dish) used to convolved the model
        with a gaussian PSF [m],\n
    `blm` {float}:
        Maximum baseline to limit the spatial frequencies plot,\n
    `npts` {int}:
        Number of pixels in the image,\n
    `fov` {float}:
        Field of view of the image [mas],\n
    `hamming` {boolean}:
        If True, hamming windowing is used to reduce artifact,\n
    `cont` {boolean}:
        If True, add contours of the intensity values,\n
    `corono` {boolean}:
        If True, remove the central pixel,\n
    `expert_plot` {boolean}:
        If True, plot additional figure (for 'pwhl' model),\n
    `display` {boolean}:
        If True, plots are showed,\n

    Outputs:
    --------
    `image_orient` {array}:
        Image of the model (oriented north/up, east/left),\n
    `ima_conv_orient` {array}:
        Convolved model with a gaussian PSF,\n
    `xScales` {array}:
        Spatial coordinates of the images,\n
    `uv_scale` {array}:
        Fourier coordinated of the amplitude/phase image,\n
    `norm_amp` {array}:
        Normalized amplitudes of the visibility,\n
    `pixel_size` {float}:
        Pixel size of the image.
    """
    sns.reset_orig()
    # Compute base max to get the good fov
    fov = mas2rad(fov)
    bmax = (wl / fov) * npts

    pixel_size = rad2mas(fov) / npts  # Pixel size of the image [mas]

    # Creat UV coord
    UVTable = UVGrid(bmax, npts) / 2.0  # Factor 2 due to the fft
    Utable = UVTable[:, 0]
    Vtable = UVTable[:, 1]

    uv_scale = np.reshape(Utable, (npts, npts))

    modelname = param["model"]
    model_target = select_model(modelname)

    isValid, log = check_params_model(param)
    if not isValid:
        cprint("Model %s not valid:" % param["model"], "cyan")
        cprint(log, "cyan")
        return None

    wl_array = np.array([wl])
    if param["model"] == "pwhl":
        vis = model_target(
            Utable, Vtable, wl_array, param, expert_plot=expert_plot, verbose=verbose
        )
    else:
        vis = model_target(Utable, Vtable, wl_array, param)

    param_psf = {"fwhm": rad2mas(wl / (2 * base_max)), "x0": 0, "y0": 0}

    conv_psf = visGaussianDisk(Utable, Vtable, wl_array, param_psf)

    # Apodisation
    x, y = np.meshgrid(range(npts), range(npts))
    freq_max = rad2arcsec(bmax / wl) / 2.0
    pix_vis = 2 * freq_max / npts
    freq_map = np.sqrt((x - (npts / 2.0)) ** 2 + (y - (npts / 2.0)) ** 2) * pix_vis

    x = np.squeeze(np.linspace(0, 1.5 * np.sqrt(freq_max ** 2 + freq_max ** 2), npts))
    y = np.squeeze(np.exp(-(x ** 2) / (2 * (fwhm_apod / 2.355) ** 2)))

    # Can use hamming window to apodise the visibility
    if hamming:
        y = np.hamming(2 * npts)[npts:]

    f = interp1d(x, y)
    img_apod = 1

    if apod:
        img_apod = f(freq_map.flat).reshape(freq_map.shape)

    # Reshape because all visibililty are calculated in 1D array (faster computing)
    im_vis = vis.reshape(npts, -1) * img_apod
    fftVis = np.fft.ifft2(im_vis)

    amp = abs(vis)
    phi = np.arctan2(vis.imag, vis.real)

    x_u = Utable / wl
    freq_s_x = rad2arcsec(x_u[0:npts])
    extent_vis = (freq_s_x.min(), freq_s_x.max(), freq_s_x.min(), freq_s_x.max())

    # Create an image
    im_amp = amp.reshape(npts, -1)
    im_phi = np.rad2deg(phi.reshape(npts, -1))

    image = np.fft.fftshift(abs(fftVis))
    maxX = rad2mas(wl * npts / bmax) / 2.0
    xScales = np.linspace(-maxX, maxX, npts)

    extent_ima = (
        np.array((xScales.max(), xScales.min(), xScales.min(), xScales.max()))
        - pixel_size / 2.0
    )

    vis = vis * conv_psf
    im_vis = vis.reshape(npts, -1)
    fftVis = np.fft.ifft2(im_vis)
    ima_conv = abs(np.fft.fftshift(fftVis))

    tmp = np.fliplr(image)
    image_orient = tmp / np.max(tmp)
    ima_conv_orient = np.fliplr(ima_conv)
    ima_conv_orient /= np.max(ima_conv_orient)
    rb = rad2arcsec(2 * blm / wl)

    if corono:
        image_orient[npts // 2, npts // 2 - 1] = 0
        image_orient /= np.max(image_orient)

    # Convert data to the appropriate format (to be fixed)
    obs = None
    if data is not None:
        if type(data) is not list:
            data = [data]
        obs = np.concatenate([format_obs(x) for x in data])

    plt.figure(figsize=(13, 3.5), dpi=120)
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.99, top=1, wspace=0.18, hspace=0.25
    )
    plt.subplot(1, 4, 1)
    mymap = symmetrical_colormap(("gist_earth", None))
    plt.imshow(im_amp ** 2, origin="lower", extent=extent_vis, cmap="gist_earth")
    if obs is not None:
        save_obs = obs.copy()
        cond = save_obs[:, 1] == "V2"
        obs = save_obs[cond]
        for i in range(len(obs)):
            u = obs[i][0][0]
            v = obs[i][0][1]
            wl = obs[i][0][2]
            u_freq = rad2arcsec(u / wl)  # / (1/mas2rad(1000))
            v_freq = rad2arcsec(v / wl)  # / (1/mas2rad(1000))
            plt.scatter(u_freq, v_freq, s=4, marker="o", alpha=0.3, color="r")
            plt.scatter(-u_freq, -v_freq, s=4, marker="o", alpha=0.3, color="r")
    plt.axis([-rb, rb, -rb, rb])
    plt.xlabel("Sp. Freq [cycles/arcsec]")
    plt.ylabel("Sp. Freq [cycles/arcsec]")
    plt.title("Amplitude visibility", fontsize=12, color="grey", weight="bold")
    plt.subplot(1, 4, 2)
    plt.imshow(im_phi, origin="lower", extent=extent_vis, cmap=mymap)
    if obs is not None:
        save_obs = obs.copy()
        cond = save_obs[:, 1] == "V2"
        obs = save_obs[cond]
        for i in range(len(obs)):
            u = obs[i][0][0]
            v = obs[i][0][1]
            wl = obs[i][0][2]
            u_freq = rad2arcsec(u / wl)  # / (1/mas2rad(1000))
            v_freq = rad2arcsec(v / wl)  # / (1/mas2rad(1000))
            plt.scatter(u_freq, v_freq, s=4, marker="o", alpha=0.3, color="r")
            plt.scatter(-u_freq, -v_freq, s=4, marker="o", alpha=0.3, color="r")
    plt.title("Phase visibility", fontsize=12, color="grey", weight="bold")
    plt.xlabel("Sp. Freq [cycles/arcsec]")
    plt.ylabel("Sp. Freq [cycles/arcsec]")
    plt.axis([-rb, rb, -rb, rb])

    plt.subplot(1, 4, 3)

    plt.imshow(
        image_orient,
        cmap="turbo",
        norm=PowerNorm(p),
        interpolation=None,
        extent=np.array(extent_ima),
        origin="lower",
    )

    if cont:
        plt.contour(
            image_orient,
            levels=[0.5],
            colors=["r"],
            extent=np.array(extent_ima),
            origin="lower",
        )

    plt.xlabel(r"Relative R.A. [mas]")
    plt.ylabel(r"Relative DEC [mas]")
    plt.title("Model image", fontsize=12, color="grey", weight="bold")

    plt.subplot(1, 4, 4)
    plt.imshow(
        ima_conv_orient,
        cmap="afmhot",
        norm=PowerNorm(p),
        interpolation=None,
        extent=np.array(extent_ima),
        origin="lower",
    )
    plt.xlabel(r"Relative R.A. [mas]")
    plt.ylabel(r"Relative DEC [mas]")
    plt.title(
        "Model convolved B=%im" % base_max, fontsize=12, color="grey", weight="bold"
    )
    plt.subplots_adjust(
        top=0.93, bottom=0.153, left=0.055, right=0.995, hspace=0.24, wspace=0.3
    )
    norm_amp = im_amp / np.max(im_amp)
    return image_orient, ima_conv_orient, xScales, uv_scale, norm_amp, pixel_size


def plot_complex_model(
    grid,
    data=None,
    i_sp=0,
    bmax=100,
    unit_im="mas",
    unit_vis="lambda",
    p=0.5,
    rotation=0,
):
    """Plot model and corresponding visibility and phase plan. Additionallly, you
        can add data to show the u-v coverage compare to model.

    Parameters
    ----------
    `grid` : {class}
        Class generated using model2grid function,\n
    `i_sp` : {int}, optional
        Index number of the wavelength to display (in case of datacube), by default 0\n
    `bmax` : {int}, optional
        Maximum baseline to restrein the visibility field of view, by default 20\n
    `unit_im` : {str}, optional
        Unit of the spatial coordinates (model), by default 'arcsec'\n
    `unit_vis` : {str}, optional
        Unit of the complex coordinates (fft), by default 'lambda'\n
    `data` : {class}, optional
        Class containing oifits data (see oimalib.load()), by default None\n
    `p` : {float}, optional
        Normalization factor of the image, by default 0.5\n
    """
    fmin = grid.freq.min()
    fmax = grid.freq.max()
    fov = grid.fov
    if unit_im == "mas":
        f2 = 1
    else:
        unit_im = "arcsec"
        f2 = 1000.0

    if unit_vis == "lambda":
        f = 1e6
    elif unit_vis == "arcsec":
        f = rad2mas(1) / 1000.0

    extent_im = np.array([fov / 2.0, -fov / 2.0, -fov / 2.0, fov / 2.0]) / f2
    extent_vis = np.array([fmin, fmax, fmin, fmax]) / f

    fft2D = grid.fft
    cube = grid.cube

    im_phase = np.rad2deg(abs(np.angle(fft2D)[i_sp])[:, ::-1])
    im_amp = np.abs(fft2D)[i_sp][:, ::-1]
    im_model = cube[i_sp]

    wl_model = grid.wl[i_sp]

    umax = 2 * bmax / wl_model / f
    ax_vis = [-umax, umax, -umax, umax]
    modelname = grid.name

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    axs[0].set_title(fr'Model "{modelname}" ($\lambda$ = {wl_model * 1e6:2.2f} $\mu$m)')
    axs[0].imshow(
        im_model, norm=PowerNorm(p), origin="lower", extent=extent_im, cmap="afmhot"
    )
    axs[0].set_xlabel(r"$\Delta$RA [%s]" % (unit_im))
    axs[0].set_ylabel(r"$\Delta$DEC [%s]" % (unit_im))

    axs[1].set_title(r"Squared visibilities (V$^2$)")
    axs[1].imshow(
        im_amp ** 2,
        norm=PowerNorm(1),
        origin="lower",
        extent=extent_vis,
        cmap="gist_earth",
    )
    axs[1].axis(ax_vis)
    axs[1].plot(0, 0, "r+")

    if data is not None:
        _plot_uvdata_coord(data, ax=axs[1], rotation=rotation)

    if unit_vis == "lambda":
        axs[1].set_xlabel(r"U [M$\lambda$]")
        axs[1].set_ylabel(r"V [M$\lambda$]")
    else:
        axs[1].set_xlabel(r"U [arcsec$^{-1}$]")
        axs[1].set_ylabel(r"V [arcsec$^{-1}$]")

    # plt.subplot(1, 3, 3)
    axs[2].set_title("Phase [deg]")
    axs[2].imshow(
        im_phase,
        norm=PowerNorm(1),
        origin="lower",
        extent=extent_vis,
        cmap="gist_earth",
    )
    axs[2].plot(0, 0, "r+")

    if unit_vis == "lambda":
        axs[2].set_xlabel(r"U [M$\lambda$]")
        axs[2].set_ylabel(r"V [M$\lambda$]")
    else:
        axs[2].set_xlabel(r"U [arcsec$^{-1}$]")
        axs[2].set_ylabel(r"V [arcsec$^{-1}$]")
    axs[2].axis(ax_vis)
    plt.tight_layout()
    plt.show(block=False)
    return fig, axs


def plot_spectra(
    data,
    aver=False,
    offset=0,
    bounds=None,
    lbdBrg=2.1661,
    f_range=None,
    tellu=False,
    title=None,
    rest=0,
    speed=False,
    d_speed=1000,
    norm=True,
):
    if bounds is None:
        bounds = [2.14, 2.19]

    spectra = data.flux
    wave_cal = data.wl * 1e6
    tel = data.tel

    array_name = data.info["Array"]
    nbl = spectra.shape[0]

    n_spec = spectra.shape[0]
    l_spec, l_wave = [], []

    inbounds = (wave_cal <= bounds[1]) & (wave_cal >= bounds[0])
    for i in range(n_spec):
        spectrum = spectra[i].copy()
        spectrum = spectrum[inbounds]
        wave_bound = wave_cal[inbounds]
        inCont_bound = (np.abs(wave_bound - lbdBrg) < 0.1) * (
            np.abs(wave_bound - lbdBrg) > 0.002
        )
        nan_interp(spectrum)
        if norm:
            normalize_continuum(spectrum, wave_bound, inCont=inCont_bound)
        else:
            spectrum = spectra[i][inbounds]
        l_spec.append(spectrum)
        l_wave.append(wave_cal[inbounds] - offset)

    spec = np.array(l_spec).T
    if speed:
        wave = ((np.array(l_wave)[0] - rest) / rest) * c_light / 1e3
    else:
        wave = np.array(l_wave)[0]

    spec_aver = np.mean(spec, axis=1)

    max_spec = spec_aver.max()
    plt.figure(figsize=[6, 4])
    ax = plt.subplot(111)

    if aver:
        label = f"Averaged ({tel[0]}+{tel[1]}+{tel[2]}+{tel[3]})"
        plt.plot(wave, spec_aver, lw=1.5, label=label)
        plt.legend(fontsize=7)
    else:
        if array_name == "VLTI":
            ax.set_prop_cycle("color", ["#ffc258", "#c65d7b", "#3a6091", "#79ab8e"])
        for i in range(nbl):
            plt.plot(wave, spec[:, i], lw=1.5, label=tel[i])
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, fontsize=7)

    if not speed:
        plt.xlim(bounds[0], bounds[1])
    else:
        plt.xlim(-d_speed, d_speed)

    plt.grid(alpha=0.2)
    if tellu:
        plot_tellu()
    # if f_range is not None:
    plt.ylim(max_spec / 3.0, 1.2 * max_spec)
    plt.xlabel(r"Wavelength [$\mu$m]")
    plt.ylabel("Normalized flux [counts]")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return wave, spec


def plot_dvis(
    data,
    bounds=None,
    line=None,
    dvis_range=0.08,
    dphi_range=9,
    norm_phi=True,
    norm_vis=True,
    lbdBrg=2.1661,
):
    """
    Plot differential observables (visibility amplitude and phase).

    Parameters:
    -----------

    `data` {class}:
        Interferometric data from load()\n
    `bounds` {list}:
        Wavelengths range (by default around Br Gamma line 2.166 µm, [2.14, 2.19]),\n
    `line` {float}:
        Vertical line reference to be plotted (by default, Br Gamma line 2.166
        µm),\n
    `dvis_range`, `dphi_range` {float}:
        Range of dvis and dphi around the normalized values (normalized by the
        continuum)\n
    `norm_vis`, `norm_phi` {boolean}:
        If True, renormalized the differential quantities by the continuum (self
        reference).\n
    """
    if bounds is None:
        bounds = [2.14, 2.19]

    dic_color = _update_color_bl([data])

    bounds2 = [bounds[0] - 0.001, bounds[1] + 0.001]

    if len(data.flux.shape) == 1:
        spectrum = data.flux
    else:
        spectrum = data.flux.mean(axis=0)

    # to µm
    wl = data.wl * 1e6
    # Focus on a specific part of the spectrum
    cond_wl = (wl >= bounds[0]) & (wl <= bounds[1])

    nspec = len(spectrum)
    nwl = len(wl)

    flux = spectrum.copy()
    wl_sel = wl[cond_wl]
    n_sel = len(wl_sel)
    inCont = (np.abs(wl_sel - lbdBrg) < 0.1) * (np.abs(wl_sel - lbdBrg) > 0.004)

    if nwl == nspec:
        flux_sel = flux[cond_wl]
        normalize_continuum(flux_sel, wl_sel, inCont=inCont)
    else:
        flux_sel = [np.nan] * n_sel
        lbdBrg = line

    dphi = data.dphi
    dvis = data.dvis
    blname = data.blname
    bl = data.bl

    linestyle = {"lw": 1}
    fig = plt.figure(figsize=(4, 8.5))
    try:
        peak_line = flux_sel.max()
    except AttributeError:
        peak_line = 0

    # ------ PLOT AVERAGED SPECTRUM ------
    ax = plt.subplot(13, 1, 1)
    plt.plot(wl_sel, flux_sel, **linestyle)
    plt.text(
        0.14,
        0.8,
        "lcr = %2.2f" % peak_line,
        color="#1e82b8",
        fontsize=10,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    plt.ylabel("Spec.")

    if line is not None:
        plot_vline(line)
        plt.text(
            0.57,
            0.8,
            r"Br$\gamma$",
            color="#eab15d",
            fontsize=8,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.tick_params(axis="both", which="major", labelsize=8)
    hide_xlabel()
    plt.xlim(bounds2)
    # ------ PLOT VISIBILITY AMPLITUDE ------
    for i in range(dvis.shape[0]):
        ax = plt.subplot(13, 1, 2 + i, sharex=ax)

        data_dvis = dvis[i][cond_wl]
        dvis_m = data_dvis[~np.isnan(data_dvis)].mean()
        # inCont = (np.abs(wl[cond_wl] - lbdBrg) < 0.1) * (
        #     np.abs(wl[cond_wl] - lbdBrg) > 0.002
        # )

        nan_interp(data_dvis)
        cont_value = data_dvis[inCont].mean()
        if not np.isnan(dvis_m):
            X = wl[cond_wl]
            Y = data_dvis.copy()
            nan_interp(Y)
            if norm_vis:
                normalize_continuum(Y, X, inCont)
                Y *= cont_value
            plt.step(X, Y, color=dic_color[blname[i]], **linestyle)
            plt.text(
                0.16,
                0.8,
                "%s (%i m)" % (blname[i], bl[i]),
                fontsize=8,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            plt.ylabel("amp.")
            if line is not None:
                plot_vline(line)

            plt.ylim(dvis_m - dvis_range, dvis_m + dvis_range)
            ax.tick_params(axis="both", which="major", labelsize=8)
            hide_xlabel()
            plt.xlim(bounds2)
        else:
            plt.xlim(bounds2)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel("amp.")
            plt.text(
                0.03,
                0.8,
                "%s (%i m)" % (blname[i], bl[i]),
                fontsize=8,
                ha="left",
                va="center",
                transform=ax.transAxes,
            )

            plt.text(
                0.5,
                0.5,
                "Not available",
                color="red",
                fontsize=8,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    # ------ PLOT VISIBILITY PHASE ------
    for i in range(dphi.shape[0]):
        ax = plt.subplot(13, 1, 8 + i, sharex=ax)

        if np.diff(dphi[i][cond_wl]).mean() != 0:
            X = wl[cond_wl]
            Y = dphi[i][cond_wl]
            nan_interp(Y)
            # inCont = (np.abs(X - lbdBrg) < 0.1) * (np.abs(X - lbdBrg) > 0.002)
            if norm_phi:
                normalize_continuum(Y, X, inCont, phase=True)
            plt.step(X, Y, color=dic_color[blname[i]], **linestyle)
            dphi_m = Y.mean()

            plt.text(
                0.03,
                0.8,
                "%s (%i m)" % (blname[i], bl[i]),
                fontsize=8,
                ha="left",
                va="center",
                transform=ax.transAxes,
            )
            plt.ylabel(r"$\phi$ (deg)")
            if 8 + i != 13:
                hide_xlabel()
            else:
                plt.grid(lw=0.5, alpha=0.5)
                plt.xlabel(r"$\lambda$ ($\mu$m)")
            ax.tick_params(axis="both", which="major", labelsize=8)
            if line is not None:
                plot_vline(line)
            try:
                plt.ylim(dphi_m - dphi_range, dphi_m + dphi_range)
            except ValueError:
                plt.ylim(-dphi_range, dphi_range)
            plt.xlim(bounds2)
        else:
            if 8 + i != 13:
                plt.xticks([])
                plt.yticks([])
            else:
                plt.xlabel(r"$\lambda$ ($\mu$m)")
                ax.tick_params(axis="both", which="major", labelsize=8)

            plt.ylabel(r"$\phi$ (deg)")
            plt.text(
                0.92,
                0.8,
                blname[i],
                fontsize=8,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            plt.text(
                0.5,
                0.5,
                "Not available",
                color="red",
                fontsize=8,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            plt.xlim(bounds2)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, bottom=0.05, top=0.99)
    return fig


# Plot MCMC related results


def plot_mcmc_walker(sampler, param, fitOnly, burnin=100, savedir=None):
    """Plot walkers for the different iteration. `burnin` is the number of
    points to be ignored (first iterations). `fitOnly` is the list of fitted
    parameters. `param` is the dictionnary containing the initial parameters of
    the model. If `savedir` is given, the figure in saved."""
    fit_mcmc = get_mcmc_results(sampler, param, fitOnly=fitOnly, burnin=burnin)

    dict_label = {
        "fc": "f$_c$ [%]",
        "fh": "f$_h$ [%]",
        "la": "l$_a$",
        "incl": "i [deg]",
        "pa": "PA [deg]",
        "cj": "c$_j$",
        "sj": "s$_j$",
    }

    labels_mcmc = [dict_label.get(x, x) for x in fitOnly]

    ndim = len(fitOnly)

    sns.set_theme(color_codes=True)
    sns.set_context("talk", font_scale=0.9)
    fig, axes = plt.subplots(ndim, figsize=(7, 8.5), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "b", alpha=0.5)
        p = fit_mcmc["best"][fitOnly[i]]
        em = fit_mcmc["uncer"][fitOnly[i] + "_m"]
        ep = fit_mcmc["uncer"][fitOnly[i] + "_p"]
        ax.axhline(p, color="#ee9068")
        ax.axhspan(p - em, p + ep, color="#ee9068", alpha=0.3)
        ax.set_xlim(0, len(samples) - 10)
        ax.set_ylabel(labels_mcmc[i])
        ax.yaxis.set_label_coords(-0.15, 0.5)
    axes[-1].set_xlabel("N iteration")
    plt.subplots_adjust(top=0.98, bottom=0.08, left=0.17, right=0.982)
    if savedir is not None:
        plt.savefig(savedir + "walkers_%s_MCMC.png" % (param["model"]), dpi=300)
    return fig


def plot_mcmc_results(
    sampler,
    labels=None,
    burnin=200,
    compute_r=False,
    dpc=None,
    lk=None,
    prec=2,
    compute_w=False,
):
    """Plot modern corner plot using seaborn."""
    sns.set_theme(color_codes=True)
    flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)

    dict_mcmc = {}
    for i in range(len(labels)):
        f = 1
        if (labels[i] == "f$_c$") or (labels[i] == "f$_h$"):
            f = 100
            dict_mcmc[labels[i]] = flat_samples[:-1, i] * f
        elif labels[i] == "l$_a$":
            if lk is not None:
                ar = 10 ** flat_samples[:-1, i]
                dict_mcmc["a"] = ar
        else:
            dict_mcmc[labels[i]] = flat_samples[:-1, i]

    try:
        if lk is None:
            lk = flat_samples[:-1, np.where(np.array(labels) == "l$_k$")[0][0]]
        la = flat_samples[:-1, np.where(np.array(labels) == "l$_a$")[0][0]]
        ar = 10 ** la / (np.sqrt(1 + 10 ** (2 * lk)))
        ak = ar * (10 ** lk)
        a = (ar ** 2 + ak ** 2) ** 0.5
        dict_mcmc["a"] = a
        w = ak / a
        if compute_w:
            dict_mcmc["w"] = w
    except IndexError:
        pass

    try:
        del dict_mcmc["l$_k$"]
    except KeyError:
        pass

    if compute_r:
        if dpc is None:
            raise TypeError("Distance (dpc) is required to compute the radius in AU.")
        ar = dict_mcmc["a"]
        dict_mcmc["$r$"] = ar * dpc  # * 215.0 / 2.0
        try:
            del dict_mcmc["l$_a$"]
        except KeyError:
            pass
        del dict_mcmc["a"]

    df = pd.DataFrame(dict_mcmc)

    color = "#ee9068"
    g = sns.PairGrid(df, corner=True, height=1.7)
    g.map_lower(sns.histplot, bins=40, pthresh=0.0)
    g.map_diag(sns.histplot, bins=20, element="step", linewidth=1, kde=True, alpha=0.5)
    g.map_diag(_summary_corner_sns, color=color, prec=prec)
    g.map_lower(_results_corner_sns, color=color)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    return g


# Plot differential quantities results (photocenter shift, offset and pure line)


def plot_pcs(pcs, speed=True, r=None, dpc=None, xlim=50):
    """Plot the photocenter shift (oriented East-North)."""

    wave = pcs["wl"]
    if speed:
        rest = pcs["wl_line"]
        wave = ((pcs["wl"] - rest) / rest) * c_light / 1e3

    factor = 1
    if dpc is not None:
        factor = dpc / 1e3

    if r is not None:
        x = r * np.cos(np.linspace(0, 2 * np.pi, 100))
        y = r * np.sin(np.linspace(0, 2 * np.pi, 100))

    sns.set_theme(color_codes=True)
    sns.set_context("talk", font_scale=0.9)

    fig = plt.figure(figsize=(6, 4.8))
    ax = plt.gca()
    sc = ax.scatter(
        unumpy.nominal_values(pcs["east"]) * factor,
        unumpy.nominal_values(pcs["north"]) * factor,
        c=wave,
        cmap="coolwarm",
        edgecolor="k",
        zorder=3,
        linewidth=1,
    )
    ax.errorbar(
        unumpy.nominal_values(pcs["east"]) * factor,
        unumpy.nominal_values(pcs["north"]) * factor,
        xerr=unumpy.std_devs(pcs["east"]) * factor,
        yerr=unumpy.std_devs(pcs["north"]) * factor,
        color="k",
        ls="None",
        elinewidth=1,
        capsize=1,
    )

    if r is not None:
        ax.plot(x, y, "g--", lw=1)
    clabel = "Wavelength [km/s]"
    if not speed:
        clabel = "Wavelength [µm]"
    cbar_kws = {
        "label": clabel,
    }

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc, **cbar_kws, cax=cax)
    cbar.ax.tick_params(size=0)

    ax.set_xlim(xlim, -xlim)
    ax.set_ylim(-xlim, xlim)
    if dpc is not None:
        ax.set_xlabel("Photocenter shift [AU]")
        ax.set_ylabel("Photocenter shift [AU]")
    else:
        ax.set_xlabel("Photocenter shift [µas]")
        ax.set_ylabel("Photocenter shift [µas]")
    ax.axvline(0, ls="-", color="gray", lw=1)
    ax.axhline(0, ls="-", color="gray", lw=1)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    return fig


def plot_pco(output_pco, pcs, iwl_pc=0, pc_max=0.2):
    """Plot the photocenter offset (`pco`) and the fitted photocenter shift
    (`pcs`) for the spectral channel `iwl_pc`."""
    pco = output_pco["pco"]
    bl_pa = output_pco["bl_pa"]
    bl_length = output_pco["bl_length"]
    wl = output_pco["wl"]
    nbl = output_pco["nbl"]
    l_blname = output_pco["blname"]

    x_pc_mod = np.linspace(0, 360, 100)
    y_pc_mod = model_pcshift(x_pc_mod, pcs["fit_param"][iwl_pc]["best"])

    param = pcs["fit_param"][iwl_pc]["best"]
    uncer = pcs["fit_param"][iwl_pc]["uncer"]

    p1 = {"p": param["p"] - uncer["p"], "offset": param["offset"] - uncer["offset"]}
    p2 = {"p": param["p"] + uncer["p"], "offset": param["offset"] + uncer["offset"]}
    y_pc_mod1 = model_pcshift(x_pc_mod, p1)
    y_pc_mod2 = model_pcshift(x_pc_mod, p2)

    sns.set_theme(color_codes=True)
    sns.set_context("talk", font_scale=0.9)
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(nbl):
        blname = l_blname[i]
        color_bl = dic_color[blname]
        bl = bl_length[i]
        label = f"{blname} ({bl:2.2f} m)"
        plt.errorbar(
            bl_pa[i],
            pco[i, iwl_pc].nominal_value,
            yerr=pco[i, iwl_pc].std_dev,
            label=label,
            color=color_bl,
            **err_pts_style_pco,
        )
        plt.errorbar(
            bl_pa[i] - 180,
            -pco[i, iwl_pc].nominal_value,
            yerr=pco[i, iwl_pc].std_dev,
            color=color_bl,
            **err_pts_style_pco,
        )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.text(
        0.05,
        0.95,
        "λ = %2.4f µm\npc = %2.1f ± %2.1f µas\nθ = %2.1f ± %2.1f deg"
        % (
            wl[iwl_pc],
            pcs["fit_param"][iwl_pc]["best"]["p"] * 1000,
            pcs["fit_param"][iwl_pc]["uncer"]["p"] * 1000,
            pcs["fit_param"][iwl_pc]["best"]["offset"],
            pcs["fit_param"][iwl_pc]["uncer"]["offset"],
        ),
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )
    plt.plot(x_pc_mod, y_pc_mod, lw=1, label="Projected shift")
    plt.fill_between(x_pc_mod, y_pc_mod1, y_pc_mod2, alpha=0.5)
    plt.legend(fontsize=8, loc=1)
    plt.axhline(0, lw=1, color="gray")
    plt.ylabel("Photocenter offset [mas]")
    plt.xlabel("Baseline PA [deg]")
    plt.ylim(-pc_max, pc_max)
    plt.xlim(0, 360)
    plt.tight_layout()
    return fig


def plot_cvis_pure(pure, flc, phi_max=5, vis_range=None):
    """Plot pure line visibility and phase with the spectrum."""
    if vis_range is None:
        vis_range = [0, 1.1]
    wl = flc["wl"]
    flux = flc["flux"]
    e_flux = flc["e_flux"]
    inLine = flc["inLine"]

    fit = flc["fit"]
    try:
        line_fitted = fit["best"]["lbdBrg"]
        w_fitted = 2.355 * fit["best"]["sigBrg"] / 2.0
    except KeyError:
        line_fitted = fit["best"]["p1"]
        w_fitted = 2.355 * fit["best"]["w1"] / 2.0

    wl_model = np.linspace(2.15, 2.18, 1000)
    red_abs = flc["red_abs"]
    if not red_abs:
        flux_model = model_flux(wl_model, fit["best"]) + 1
    else:
        flux_model = model_flux_red_abs(wl_model, fit["best"]) + 1
    sns.set_theme(color_codes=True)
    sns.set_context("talk", font_scale=0.9)
    plt.figure(figsize=(6, 8))
    ax = plt.subplot(311)
    plt.errorbar(flc["wl"], flc["flux"], yerr=flc["e_flux"], **err_pts_style)
    plt.scatter(
        wl[inLine],
        flux[inLine],
        c=wl[inLine],
        cmap="coolwarm",
        edgecolor="k",
        zorder=3,
        s=30,
        linewidth=1,
    )
    plt.errorbar(
        wl[inLine],
        flux[inLine],
        yerr=e_flux,
        color="k",
        ls="None",
        elinewidth=1,
        capsize=1,
    )
    plt.plot(wl_model, flux_model, lw=1)
    plt.fill_between(
        wl_model, flux_model - e_flux, flux_model + e_flux, color="orange", alpha=0.3
    )
    plt.ylabel("Norm. flux")

    plt.xlim(2.1661 - 0.01, 2.1661 + 0.01)

    ax2 = plt.subplot(312, sharex=ax)
    plt.axvline(line_fitted, color="#b0bec4", lw=2, ls="--")
    plt.axvspan(
        line_fitted - w_fitted, line_fitted + w_fitted, color="#b0bec4", alpha=0.3
    )
    plt.errorbar(wl, pure.dvis, yerr=pure.e_dvis, **err_pts_style)
    plt.scatter(
        wl[inLine],
        pure.dvis_pure,
        c=wl[inLine],
        cmap="coolwarm",
        edgecolor="k",
        zorder=3,
        s=30,
        linewidth=1,
    )
    plt.errorbar(
        wl[inLine],
        pure.dvis_pure,
        yerr=pure.e_dvis_pure,
        color="k",
        ls="None",
        elinewidth=1,
        capsize=1,
    )
    plt.plot(wl, pure.mod_dvis, lw=1)
    plt.fill_between(
        wl,
        pure.mod_dvis - pure.e_mod_dvis,
        pure.mod_dvis + pure.e_mod_dvis,
        color="orange",
        alpha=0.3,
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.text(
        0.05,
        0.9,
        f"{pure.blname} ({pure.bl_length:2.2f} m)",
        transform=ax2.transAxes,
        # fontsize=13,
        verticalalignment="top",
        bbox=props,
    )
    plt.ylabel("Vis. Amp.")
    plt.ylim(vis_range[0], vis_range[1])
    plt.subplot(313, sharex=ax)
    plt.axvline(line_fitted, color="#b0bec4", lw=2, ls="--")
    plt.axvspan(
        line_fitted - w_fitted, line_fitted + w_fitted, color="#b0bec4", alpha=0.3
    )
    plt.errorbar(wl, pure.dphi, yerr=pure.e_dphi, **err_pts_style)
    plt.scatter(
        wl[inLine],
        pure.dphi_pure,
        c=wl[inLine],
        cmap="coolwarm",
        edgecolor="k",
        zorder=3,
        s=30,
        linewidth=1,
    )
    plt.errorbar(
        wl[inLine],
        pure.dphi_pure,
        yerr=pure.e_dphi_pure,
        color="k",
        ls="None",
        elinewidth=1,
        capsize=1,
    )
    plt.plot(wl, pure.mod_dphi, lw=1)
    plt.fill_between(
        wl,
        pure.mod_dphi - pure.e_mod_dphi,
        pure.mod_dphi + pure.e_mod_dphi,
        color="orange",
        alpha=0.3,
    )
    plt.xlabel("Wavelength [µm]")
    plt.ylabel("Vis. phase [deg]")
    plt.ylim(-phi_max, phi_max)
    plt.tight_layout()
    return ax2


# dependant functions for main
# ============================


def _update_color_bl(tab):
    data = tab[0]
    array_name = data.info["Array"]
    nbl_master = len(set(data.blname))

    if array_name == "CHARA":
        unknown_color = plt.cm.turbo(np.linspace(0, 1, nbl_master))
    else:
        unknown_color = plt.cm.Set2(np.linspace(0, 1, 8))

    i_cycle = 0
    for j in range(len(tab)):
        data = tab[j]
        nbl = data.blname.shape[0]
        for i in range(nbl):
            base = data.blname[i]
            if base not in dic_color.keys():
                dic_color[base] = unknown_color[i_cycle]
                i_cycle += 1
    return dic_color


def _create_match_tel(data):
    dic_index = {}
    for i in range(len(data.index_ref)):
        ind = data.index_ref[i]
        tel = data.teles_ref[i]
        if ind not in dic_index.keys():
            dic_index[ind] = tel
    return dic_index


def check_closed_triplet(data, i=0):
    U = [data.u1[i], data.u2[i], data.u3[i], data.u1[i]]
    V = [data.v1[i], data.v2[i], data.v3[i], data.v1[i]]

    triplet = data.cpname[i]
    fig = plt.figure(figsize=[5, 4.5])
    plt.plot(U, V, label=triplet)
    plt.legend()
    plt.plot(0, 0, "r+")
    plt.grid(alpha=0.2)
    plt.xlabel("U [m]")
    plt.ylabel("V [m]")
    plt.tight_layout()
    return fig


def plot_tellu(label=None, plot_ind=False, val=5000, lw=0.5):
    file_tellu = pkg_resources.resource_stream(
        "oimalib", "internal_data/Telluric_lines.txt"
    )
    tellu = np.loadtxt(file_tellu, skiprows=1)
    file_tellu.close()
    plt.axvline(np.nan, lw=lw, c="gray", alpha=0.5, label=label)
    for i in range(len(tellu)):
        plt.axvline(tellu[i], lw=lw, c="crimson", ls="--", alpha=0.5)
        if plot_ind:
            plt.text(tellu[i], val, i, fontsize=7, c="crimson")


def _plot_uvdata_coord(tab, ax=None, rotation=0):
    """Plot u-v coordinated of a bunch of data (see `plot_uv()`)."""
    if (type(tab) != list) & (type(tab) != np.ndarray):
        tab = [tab]

    dic_color = _update_color_bl(tab)

    list_bl = []
    for data in tab:
        nbl = data.blname.shape[0]
        for bl in range(nbl):
            flag = np.invert(data.flag_vis2[bl])
            u = data.u[bl] / data.wl[flag] / 1e6
            v = data.v[bl] / data.wl[flag] / 1e6
            base, label = data.blname[bl], ""

            vis2 = data.vis2[bl]
            if len(vis2[~np.isnan(vis2)]) == 0:
                continue

            if base not in list_bl:
                label = base
                list_bl.append(base)

            p_color = dic_color[base]
            angle = np.deg2rad(rotation)
            um = np.squeeze(u * np.cos(angle) - v * np.sin(angle))
            vm = np.squeeze(u * np.sin(angle) + v * np.cos(angle))
            ax.plot(um, vm, color=p_color, label=label, marker="o", ms=4)
            ax.plot(-um, -vm, ms=4, color=p_color, marker="o")
    return None


def _plot_v2_residuals(data, param, fitOnly=None, hue=None, use_flag=True):
    if fitOnly is None:
        fitOnly = []

    l_mod = compute_geom_model_fast(data, param)
    mod_v2 = []
    for i in range(len(l_mod)):
        mod_v2.append(l_mod[i]["vis2"])

    input_keys = ["vis2", "e_vis2", "freq_vis2", "wl", "blname", "set", "flag_vis2"]

    dict_obs = {}
    for k in input_keys:
        dict_obs[k] = []

    nobs = 0
    for d in data:
        for k in input_keys:
            nbl = d.vis2.shape[0]
            nwl = d.vis2.shape[1]
            if k == "wl":
                for _ in range(nbl):
                    dict_obs[k].extend(np.round(d[k] * 1e6, 3))
            elif k == "blname":
                for j in range(nbl):
                    for _ in range(nwl):
                        dict_obs[k].append(d[k][j])
            elif k == "set":
                for _ in range(nbl):
                    for _ in range(nwl):
                        dict_obs[k].append(nobs)
            else:
                dict_obs[k].extend(d[k].flatten())
        nobs += 1

    dict_obs["mod"] = np.array(mod_v2).flatten()

    dict_obs["res"] = (dict_obs["vis2"] - dict_obs["mod"]) / dict_obs["e_vis2"]

    flag = np.array(dict_obs["flag_vis2"])
    flag_nan = np.isnan(np.array(dict_obs["vis2"]))

    if use_flag:
        for k in dict_obs.keys():
            dict_obs[k] = np.array(dict_obs[k])[~flag & ~flag_nan]

    df = pd.DataFrame(dict_obs)

    d_freedom = len(fitOnly)

    chi2_vis2_full = np.sum((df["vis2"] - df["mod"]) ** 2 / (df["e_vis2"]) ** 2)
    chi2_vis2 = chi2_vis2_full / (len(df["e_vis2"]) - (d_freedom - 1))

    label = "DATA"
    if hue == "wl":
        label = "Wavelenght [µm]"

    fig = plt.figure(constrained_layout=False, figsize=(7, 7))
    axd = fig.subplot_mosaic(
        [["vis2"], ["res_vis2"]],
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax = sns.scatterplot(
        x="freq_vis2",
        y="vis2",
        data=df,
        palette="crest",
        zorder=10,
        label=label,
        ax=axd["vis2"],
        style=None,
        hue=hue,
    )

    sns.scatterplot(
        x="freq_vis2",
        y="mod",
        data=df,
        color="#e19751",
        zorder=10,
        marker="^",
        label=r"MODEL ($\chi^2_r=%2.2f$)" % chi2_vis2,
        ax=axd["vis2"],
    )
    ax.errorbar(
        df.freq_vis2,
        df.vis2,
        yerr=df.e_vis2,
        fmt="None",
        zorder=1,
        color="gray",
        alpha=0.4,
        capsize=2,
    )
    sns.scatterplot(
        x="freq_vis2",
        y="res",
        data=df,
        zorder=10,
        ax=axd["res_vis2"],
    )
    axd["res_vis2"].sharex(axd["vis2"])
    plt.xlabel(r"Sp. Freq. [arcsec$^{-1}$]")
    axd["vis2"].set_ylabel("V$^{2}$")
    axd["vis2"].set_xlabel("")
    axd["vis2"].set_ylim([0, 1.1])
    axd["res_vis2"].set_ylabel(r"Residuals [$\sigma$]")
    axd["res_vis2"].axhspan(-1, 1, alpha=0.6, color="#418fde")
    axd["res_vis2"].axhspan(-2, 2, alpha=0.6, color="#8bb8e8")
    axd["res_vis2"].axhspan(-3, 3, alpha=0.6, color="#c8d8eb")
    axd["res_vis2"].set_ylim(-5, 5)

    axd["vis2"].tick_params(
        axis="x",  # changes apply to the x-axis
        which="major",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off)
    )
    plt.subplots_adjust(hspace=0.1, top=0.98, right=0.98, left=0.11)
    return df, chi2_vis2, chi2_vis2_full, mod_v2


def _plot_cp_residuals(data, param, fitOnly=None, hue=None, use_flag=True):
    if fitOnly is None:
        fitOnly = []

    l_mod = compute_geom_model_fast(data, param, use_flag=False)
    mod_cp = []
    for i in range(len(l_mod)):
        mod_cp.append(l_mod[i]["cp"])

    input_keys = ["cp", "e_cp", "freq_cp", "wl", "cpname", "set", "flag_cp"]

    dict_obs = {}
    for k in input_keys:
        dict_obs[k] = []

    nobs = 0
    for d in data:
        for k in input_keys:
            nbl = d.cp.shape[0]
            nwl = d.cp.shape[1]
            if k == "wl":
                for _ in range(nbl):
                    dict_obs[k].extend(np.round(d[k] * 1e6, 3))
            elif k == "cpname":
                for j in range(nbl):
                    for _ in range(nwl):
                        dict_obs[k].append(d[k][j])
            elif k == "set":
                for _ in range(nbl):
                    for _ in range(nwl):
                        dict_obs[k].append(nobs)
            else:
                dict_obs[k].extend(d[k].flatten())
        nobs += 1

    dict_obs["mod"] = np.array(mod_cp).flatten()
    dict_obs["res"] = (dict_obs["cp"] - dict_obs["mod"]) / dict_obs["e_cp"]

    if use_flag:
        flag = np.array(dict_obs["flag_cp"])
        flag_nan = np.isnan(np.array(dict_obs["cp"]))
        for k in dict_obs.keys():
            dict_obs[k] = np.array(dict_obs[k])[~flag & ~flag_nan]

    df = pd.DataFrame(dict_obs)

    d_freedom = len(fitOnly)

    chi2_cp_full = np.sum((df["cp"] - df["mod"]) ** 2 / (df["e_cp"]) ** 2)
    chi2_cp = chi2_cp_full / (len(df["e_cp"]) - (d_freedom - 1))

    res_max = 5
    if np.max(abs(df["res"])) >= 5:
        res_max = abs(df["res"]).max() * 1.2

    fig = plt.figure(constrained_layout=False, figsize=(7, 7))
    axd = fig.subplot_mosaic(
        [["cp"], ["res_cp"]],
        gridspec_kw={"height_ratios": [3, 1]},
    )
    label = "DATA"
    if hue == "wl":
        label = "Wavelenght [µm]"
    ax = sns.scatterplot(
        x="freq_cp",
        y="cp",
        data=df,
        palette="crest",
        zorder=10,
        label=label,
        ax=axd["cp"],
        style=None,
        hue=hue,
    )
    sns.scatterplot(
        x="freq_cp",
        y="mod",
        data=df,
        color="#e19751",
        zorder=10,
        marker="^",
        label=r"MODEL ($\chi^2_r=%2.2f$)" % chi2_cp,
        ax=axd["cp"],
    )
    ax.errorbar(
        df.freq_cp,
        df.cp,
        yerr=df.e_cp,
        fmt="None",
        zorder=1,
        color="gray",
        alpha=0.4,
        capsize=2,
    )
    sns.scatterplot(
        x="freq_cp",
        y="res",
        data=df,
        zorder=10,
        ax=axd["res_cp"],
    )
    axd["res_cp"].sharex(axd["cp"])
    plt.xlabel(r"Sp. Freq. [arcsec$^{-1}$]")
    axd["cp"].set_ylabel(r"Closure phase $\phi$ [deg]")
    axd["cp"].set_xlabel("")
    axd["cp"].set_ylim(-10, 10)
    axd["res_cp"].set_ylabel(r"Residuals [$\sigma$]")
    axd["res_cp"].axhspan(-1, 1, alpha=0.6, color="#418fde")
    axd["res_cp"].axhspan(-2, 2, alpha=0.6, color="#8bb8e8")
    axd["res_cp"].axhspan(-3, 3, alpha=0.6, color="#c8d8eb")
    axd["res_cp"].set_ylim(-res_max, res_max)

    axd["cp"].tick_params(
        axis="x",  # changes apply to the x-axis
        which="major",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off)
    )
    plt.subplots_adjust(hspace=0.1, top=0.98, right=0.98, left=0.11)
    return df, chi2_cp, chi2_cp_full, mod_cp


def symmetrical_colormap(cmap_settings, new_name=None):
    """This function take a colormap and create a new one, as the concatenation of itself by a symmetrical fold."""
    # get the colormap
    cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_" + cmap_settings[0]  # ex: 'sym_Blues'

    # this defined the roughness of the colormap, 128 fine
    n = 128

    # get the list of color from colormap
    colors_r = cmap(np.linspace(0, 1, n))  # take the standard colormap # 'right-part'
    colors_l = colors_r[
        ::-1
    ]  # take the first list of color and flip the order # "left-part"

    # combine them and build a new colormap
    colors = np.vstack((colors_l, colors_r))
    mymap = mcolors.LinearSegmentedColormap.from_list(new_name, colors)

    return mymap


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


def _summary_corner_sns(x, prec=2, color="#ee9068", **kwargs):
    t_unit = {
        "f$_c$": "%",
        "incl": "deg",
        "i": "deg",
        "a$_r*$": r"r$_{star}$",
        "a$_r$": "mas",
        "PA": "deg",
        "c$_j$": "",
        "s$_j$": "",
        "l$_a$": "",
        "$r$": "AU",
    }
    mcmc = np.percentile(x, [16, 50, 84])
    q = np.diff(mcmc)
    txt = r"{3} = {0:.%if}$_{{-{1:.%if}}}^{{+{2:.%if}}}$ {4}" % (prec, prec, prec)
    try:
        txt = txt.format(mcmc[1], q[0], q[1], x.name, t_unit[x.name])
    except KeyError:
        txt = txt.format(mcmc[1], q[0], q[1], x.name, "")
    ax = plt.gca()
    ax.set_axis_off()
    ax.axvline(mcmc[0], lw=1, color=color, alpha=0.8, ls="--")
    ax.axvline(mcmc[1], lw=1, color=color, alpha=0.8, ls="-")
    ax.axvline(mcmc[2], lw=1, color=color, alpha=0.8, ls="--")
    ax.set_title(txt, fontsize=9)


def _results_corner_sns(x, y, color="#ee9068", **kwargs):
    p1 = np.percentile(x, [16, 50, 84])
    p2 = np.percentile(y, [16, 50, 84])
    ax = plt.gca()
    ax.plot(p1[1], p2[1], "s", color=color, alpha=0.8)
    ax.axvline(p1[1], lw=1, color=color, alpha=0.8)
    ax.axhline(p2[1], lw=1, color=color, alpha=0.8)


def plot_diffobs_model(dobs, data, dobs_full=None, speed=False):
    wl0 = dobs.p_line
    if speed:
        wave3 = ((dobs.wl - wl0) / wl0) * c_light / 1e3
        if dobs_full is not None:
            wave4 = ((dobs_full.wl - wl0) / wl0) * c_light / 1e3
    else:
        wave3 = dobs.wl
        if dobs_full is not None:
            wave4 = dobs_full.wl

    sns.set_theme(color_codes=True)
    sns.set_context("talk", font_scale=0.9)
    fig = plt.figure(figsize=(6, 6))
    plt.subplot(3, 1, 1)
    plt.scatter(
        wave3,
        dobs.flux,
        s=40,
        c=wave3,
        cmap="coolwarm",
        marker="s",
        linewidth=0.5,
        edgecolor="k",
        zorder=3,
    )
    plt.plot(
        wave4,
        dobs_full.flux,
        lw=1,
    )
    plt.ylabel("Norm. Flux")
    frame1 = plt.gca()
    frame1.tick_params(axis="x", colors="w")

    plt.subplot(3, 1, 2)
    frame2 = plt.gca()
    plt.scatter(
        wave3,
        dobs.dvis,
        s=40,
        c=wave3,
        cmap="coolwarm",
        marker="s",
        linewidth=0.5,
        edgecolor="k",
        zorder=3,
    )
    plt.plot(wave4, dobs_full.dvis, lw=1)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.text(
        0.05,
        0.3,
        "%s (%i m)" % (data.blname[dobs.ibl], data.bl[dobs.ibl]),
        transform=frame2.transAxes,
        fontsize=13,
        verticalalignment="top",
        bbox=props,
    )
    plt.ylabel("Diff. Vis.")
    frame2.tick_params(axis="x", colors="w")
    plt.subplot(3, 1, 3)
    plt.scatter(
        wave3,
        dobs.dphi,
        s=40,
        c=wave3,
        cmap="coolwarm",
        marker="s",
        linewidth=0.5,
        edgecolor="k",
        zorder=3,
    )
    plt.plot(wave4, dobs_full.dphi, lw=1)
    plt.ylabel("Diff. Phase [deg]")
    if speed:
        plt.xlabel("Velocity [km/s]")
    else:
        plt.xlabel("Wavelengths [µm]")
    plt.subplots_adjust(left=0.17, right=0.98, top=0.99)
    return fig


def plot_image_model_pcs(
    iwl,
    pcs,
    modelfile,
    flip=True,
    rotation=0,
    sub_cont=None,
    fov=None,
    results=False,
    cm=False,
):
    """Plot PCS and model image in the same plot, data should come from aspro."""
    hdu = fits.open(modelfile)
    cube = hdu[0].data
    image = cube[0]
    hdr = hdu[0].header
    hdu.close()
    pix = abs(rad2mas(np.deg2rad(hdr["CDELT1"])) * 1000.0)
    n_wl = cube.shape[0]
    delta_wl = hdr["CDELT3"]
    wl0 = hdr["CRVAL3"]
    wl_model = np.linspace(wl0, wl0 + delta_wl * (n_wl - 1), n_wl)
    idx = find_nearest(wl_model, pcs["wl"][iwl])

    if flip:
        image = np.fliplr(cube[idx])
    else:
        image = cube[idx]

    # axes_rot = (1, 2)
    image = rotate(image, -rotation, reshape=False)
    im_cont = rotate(cube[0], -rotation, reshape=False)

    npix = len(image)

    xx, yy = np.arange(npix), np.arange(npix)
    xx_c = xx - npix // 2 - 0.5
    yy_c = yy - npix // 2
    distance = np.sqrt(xx_c ** 2 + yy_c[:, np.newaxis] ** 2) * pix

    l_cmass = []
    for x in pcs["wl"]:
        idx = find_nearest(wl_model, x)
        if flip:
            tmp = np.fliplr(cube[idx].copy())
        else:
            tmp = cube[idx].copy()
        if sub_cont:
            tmp -= im_cont
        tmp = rotate(tmp, -rotation, reshape=False)
        tmp[tmp < tmp.max() / 1e5] = 1e-30
        tmp[distance > 300] = 1e-30
        c_mass = np.array(center_of_mass(tmp))
        c_mass2 = ((c_mass) - npix / 2.0) * pix
        l_cmass.append(c_mass2)
    l_cmass = np.array(l_cmass)

    extent = (np.array([npix, 0, 0, npix]) - npix // 2) * pix

    if sub_cont:
        image -= im_cont
    image[image < 0] = 0

    sns.set_theme(color_codes=True)
    sns.set_context("talk", font_scale=0.9)
    plt.figure(figsize=(6, 4.8))
    ax = plt.subplot(111)
    if results:
        plt.text(
            0.05,
            0.95,
            "λ = %2.4f µm\npc = %2.1f ± %2.1f µas\nθ = %2.1f ± %2.1f deg"
            % (
                pcs["wl"][iwl],
                pcs["fit_param"][iwl]["best"]["p"],
                pcs["fit_param"][iwl]["uncer"]["p"],
                pcs["fit_param"][iwl]["best"]["offset"],
                pcs["fit_param"][iwl]["uncer"]["offset"],
            ),
            transform=ax.transAxes,
            verticalalignment="top",
            color="#f1d68e",
            alpha=0.7,
        )

    rest = pcs["wl_line"]
    wave = ((pcs["wl"][iwl] - rest) / rest) * c_light / 1e3
    plt.text(
        0.05,
        0.95,
        f"speed = {wave:2.1f} km/s",
        transform=ax.transAxes,
        verticalalignment="top",
        color="#f1d68e",
        alpha=0.7,
    )
    plt.axvline(0, color="gray", alpha=0.2, lw=1)
    plt.axhline(0, color="gray", alpha=0.2, lw=1)
    plt.imshow(
        image,
        origin="lower",
        extent=extent,
        cmap="gist_stern",
        norm=PowerNorm(1),
        interpolation="bilinear",
    )
    ax.grid(False)
    plt.scatter(
        unumpy.nominal_values(pcs["east"]),
        unumpy.nominal_values(pcs["north"]),
        c=pcs["wl"],
        cmap="coolwarm",
        edgecolor="k",
        zorder=3,
        linewidth=1,
    )

    i_east = unumpy.nominal_values(pcs["east"])[iwl]
    i_north = unumpy.nominal_values(pcs["north"])[iwl]
    plt.plot(i_east, i_north, "g+", zorder=3)
    if cm:
        plt.scatter(
            -l_cmass[:, 1],
            l_cmass[:, 0],
            c=pcs["wl"],
            alpha=0.8,
            s=200,
            marker="s",
            cmap="coolwarm",
        )
    plt.xlabel("Relative RA [µas]")
    plt.ylabel("Relative DEC [µas]")
    if fov is not None:
        plt.axis([fov / 2, -fov // 2, -fov / 2, fov / 2])
    plt.tight_layout()
    return image, extent, l_cmass


def plot_model_RT(
    iwl,
    pcs,
    modelfile,
    flip=True,
    rotation=0,
    sub_cont=None,
    fov=None,
):
    """Plot model image from the RT modelling cube."""
    hdu = fits.open(modelfile)
    cube = hdu[0].data
    image = cube[0]
    hdr = hdu[0].header
    hdu.close()
    pix = abs(rad2mas(np.deg2rad(hdr["CDELT1"])) * 1000.0)
    n_wl = cube.shape[0]
    delta_wl = hdr["CDELT3"]
    wl0 = hdr["CRVAL3"]
    wl_model = np.linspace(wl0, wl0 + delta_wl * (n_wl - 1), n_wl)
    idx = find_nearest(wl_model, pcs["wl"][iwl])

    if flip:
        image = np.fliplr(cube[idx])
    else:
        image = cube[idx]

    # axes_rot = (1, 2)
    image = rotate(image, -rotation, reshape=False)
    im_cont = rotate(cube[0], -rotation, reshape=False)

    npix = len(image)
    extent = (np.array([npix, 0, 0, npix]) - npix // 2) * pix

    if sub_cont:
        image -= im_cont
    image[image < 0] = 0

    sns.set_theme(color_codes=True)
    sns.set_context("talk", font_scale=0.9)
    plt.figure(figsize=(6, 4.8))
    ax = plt.subplot(111)
    rest = pcs["wl_line"]
    wave = ((pcs["wl"][iwl] - rest) / rest) * c_light / 1e3
    plt.text(
        0.05,
        0.95,
        f"speed = {wave:2.1f} km/s",
        transform=ax.transAxes,
        verticalalignment="top",
        color="#f1d68e",
        alpha=0.7,
    )
    plt.axvline(0, color="gray", alpha=0.2, lw=1)
    plt.axhline(0, color="gray", alpha=0.2, lw=1)
    plt.imshow(
        image,
        origin="lower",
        extent=extent,
        cmap="gist_stern",
        norm=PowerNorm(1),
        interpolation="bilinear",
    )
    ax.grid(False)
    plt.xlabel("Relative RA [µas]")
    plt.ylabel("Relative DEC [µas]")
    if fov is not None:
        plt.axis([fov / 2, -fov // 2, -fov / 2, fov / 2])
    plt.tight_layout()
    return image, extent
