import math
import os
from glob import glob

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.constants import c as c_light
from scipy.stats import gaussian_kde
from uncertainties import unumpy

import oimalib as oi


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


def select_data_time_range(list_dataset, time_lim=None):
    """Check the relative time between the first dataset and the others.
    Select only the dataset in a range of time specified on time_lim (e.g.: time_lim = [0, 6],
    so the maximum time difference between files is 6h including the first one."""
    if time_lim is None:
        time_lim = [0, 24]

    oi = list_dataset[0].info.filename
    with fits.open(oi) as fo:
        mjd0 = fo["OI_VIS2", None].data.field("MJD")[0]
    l_hour = []
    for d in list_dataset:
        tmp_oi = d.info.filename
        with fits.open(tmp_oi) as fo:
            mjd = fo["OI_VIS2", None].data.field("MJD")[0]
        l_hour.append(round((mjd - mjd0) * 24, 1))
    l_hour = np.array(l_hour)
    l_hour = l_hour[~np.isnan(l_hour)]
    t0 = time_lim[0]
    t1 = time_lim[1]
    file_to_be_combined = [
        i for i in range(len(list_dataset)) if (l_hour[i] >= t0) & (l_hour[i] <= t1)
    ]
    return file_to_be_combined


def build_dataset_list(datadir, calibrated=True, i_start=0, i_end=-1, part=None):
    if datadir[-1] != "/":
        datadir.append("/")
    end_file = "*ted.fits" if calibrated else "*scivis.fits"
    list_files = sorted(glob(datadir + end_file))
    list_data = [oi.load(x, cam="SC") for x in list_files][i_start:i_end]
    list_data_ft = [oi.load(x, cam="FT") for x in list_files][i_start:i_end]
    part = str(i_start) + "-" + str(i_end)
    list_data[0].info.part = part
    return list_data, list_data_ft


def compute_pcs_files(
    list_dataset, file_to_be_combined, param_lcr, param_pure=None, list_dataset_ft=None
):
    if param_pure is None:
        param_pure = {}
    dataset = convert_ind_data(list_dataset[0])
    oi_pure = oi.OiPure(dataset)
    oi_pure.fit_lcr(**param_lcr)
    wl_inline = oi_pure.wl[oi_pure.inLine]
    wvl0 = 2.1661178  # wavelength of BrGamma (in microns)
    vLine = (wl_inline - wvl0) / wvl0 * c_light / 1e3

    pcs_matrix = np.zeros([len(file_to_be_combined), 4, len(vLine)])
    for i, ifile in enumerate(file_to_be_combined):
        dataset = convert_ind_data(list_dataset[ifile], corr_tellu=True)
        if list_dataset_ft is not None:
            dataset_ft = convert_ind_data(list_dataset_ft[ifile])
        else:
            dataset_ft = None

        oi_pure = oi.OiPure(dataset)
        oi_pure.fit_lcr(**param_lcr)
        oi_pure.compute_quantities_ibl(ft=dataset_ft, **param_pure)
        oi_pure.compute_pcs_direct()

        # Get the pcs (oriented toward east and north)
        x = unumpy.nominal_values(oi_pure.pcs_east_dir)
        y = unumpy.nominal_values(oi_pure.pcs_north_dir)
        e_x = unumpy.std_devs(oi_pure.pcs_east_dir)
        e_y = unumpy.std_devs(oi_pure.pcs_north_dir)

        pcs_matrix[i, 0, :] = x
        pcs_matrix[i, 1, :] = y
        pcs_matrix[i, 2, :] = e_x
        pcs_matrix[i, 3, :] = e_y

    return pcs_matrix, vLine


def plot_pcs_files(pcs_matrix, vLine, ax=None, alpha=0.1, error=False, m_size=50):
    """Plot the pcs over the files computed with pcs.compute_pcs_files() function."""
    if ax is None:
        ax = plt.gca()
    for i in range(pcs_matrix.shape[0]):
        x_ref, y_ref = 0, 0
        x = pcs_matrix[i, 0, :]
        y = pcs_matrix[i, 1, :]
        e_x = pcs_matrix[i, 2, :]
        e_y = pcs_matrix[i, 3, :]
        ax.scatter(
            (x - x_ref),
            (y - y_ref),
            c=vLine,
            s=m_size,
            cmap="coolwarm",
            edgecolor="k",
            zorder=3,
            linewidth=1,
            alpha=alpha,
        )

        if error:
            ax.errorbar(
                x - x_ref,
                y - y_ref,
                xerr=e_x,
                yerr=e_y,
                color="k",
                ls="None",
                elinewidth=1,
                capsize=1,
            )


def compute_orient_mag(
    OiPure,
    pt1=2,
    pt2=-1,
    n_sim=10000,
    norm_err=1.0,
    save=False,
    verbose=True,
):
    """Compute the orientation from the velocity point 'pt1' to 'pt2' (i.e.: center to
    red if 'pt1'=2, 'pt2'=-1)."""
    A = OiPure.pcs_east_dir
    B = OiPure.pcs_north_dir

    x1, y1 = A[pt1].nominal_value, B[pt1].nominal_value
    e_x1, e_y1 = A[pt1].std_dev, B[pt1].std_dev

    x2, y2 = A[pt2].nominal_value, B[pt2].nominal_value
    e_x2, e_y2 = A[pt2].std_dev, B[pt2].std_dev

    angles, perturbed_points = [], []

    # Monte Carlo Metho
    for _ in range(n_sim):
        x1_sim = x1 + np.random.normal(0, e_x1 / norm_err)
        y1_sim = y1 + np.random.normal(0, e_y1 / norm_err)
        x2_sim = x2 + np.random.normal(0, e_x2 / norm_err)
        y2_sim = y2 + np.random.normal(0, e_y2 / norm_err)

        angle_rad = math.atan2(y2_sim - y1_sim, x2_sim - x1_sim)
        angle_deg = math.degrees(angle_rad)

        angles.append(angle_deg)
        perturbed_points.append(((x1_sim, y1_sim), (x2_sim, y2_sim)))

    angles = np.array(angles)
    angles -= 90.0
    angles *= -1

    if (x2 < x1) & (y2 > y1):
        angles = angles + 360

    mean_angle = np.mean(angles)
    med_angle = np.median(angles)

    Q1 = np.percentile(angles, 25)  # First quartile (25%)
    Q3 = np.percentile(angles, 75)  # Third quartile (75%)
    stat_IQR = Q3 - Q1  # Intervalle interquartile (plage de l'erreur)

    kde = gaussian_kde(angles, bw_method=0.1)
    x_vals = np.linspace(min(angles), max(angles), 1000)
    y_vals = kde(x_vals)

    # Most probable value
    mode_probable = x_vals[np.argmax(y_vals)]

    if verbose:
        print(f"\n=== Relative orientation between {pt1} and {pt2} points ==== ")
        print(f"Most probale angle {mode_probable:.0f} ± {stat_IQR:.0f} deg")
        print(f"Mean angle {mean_angle:.1f} deg, median {med_angle:.1f} deg")

    plt.figure(figsize=(11, 5))
    ax = plt.subplot(1, 2, 1)
    ax.scatter([x1, x2], [y1, y2], color="red", zorder=5, label="Points originaux")

    # Tracer les perturbations des points (Monte Carlo)
    for p1, p2 in perturbed_points[:100]:  # Afficher seulement un échantillon pour la clarté
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", alpha=0.1)

    # Tracer les vecteurs pour les 100 premières simulations
    for p1, p2 in perturbed_points[:100]:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", alpha=0.2)

    # Tracer l'angle entre les points perturbés

    angle_rad = math.atan2(y2 - y1, x2 - x1)
    angle_deg_no_err = -1 * (math.degrees(angle_rad) - 90.0)  # relative to the north

    if (x2 < x1) & (y2 > y1):
        angle_deg_no_err += 360

    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(facecolor="blue", edgecolor="blue", arrowstyle="->", lw=2),
    )
    plt.plot(
        np.nan,
        np.nan,
        "b-",
        label=f"Absolute angle = {angle_deg_no_err:.1f} deg",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.xlim([250, -250])
    plt.ylim([-250, 250])
    plt.axvline(0, color="k")
    plt.axhline(0, color="k")
    ax.set_aspect("equal")
    ax.legend()

    # Tracer la distribution des angles
    ax1 = plt.subplot(1, 2, 2)
    ax1 = plt.gca()
    ax1.hist(angles, bins=50, color="skyblue", edgecolor="black", alpha=0.7)

    ax2 = ax1.twinx()

    ax2.plot(x_vals, y_vals, label="Estimation de la densité", color="blue")

    ax1.axvline(
        mode_probable,
        color="g",
        linestyle="dashed",
        label=f"Most probable = {mode_probable:.1f} ± {stat_IQR:.1f}°",
    )

    ax1.axvline(
        mean_angle,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Average = {mean_angle:.2f}°",
    )

    ax1.axvline(
        mean_angle + stat_IQR,
        color="orange",
        linestyle="dashed",
        linewidth=2,
        label=f"+1 STD= {mean_angle + stat_IQR:.2f}°",
    )
    ax1.axvline(
        mean_angle - stat_IQR,
        color="orange",
        linestyle="dashed",
        linewidth=2,
        label=f"-1 STD = {mean_angle - stat_IQR:.2f}°",
    )
    ax1.legend(loc=2)
    plt.title("Monte-Carlo distribution of orientations.")
    plt.xlabel("Orientation (degrees)")
    plt.ylabel("Frequency")
    ax2.legend(loc=3)

    res = np.array([angle_deg_no_err, mode_probable, stat_IQR, mean_angle])

    mjd = OiPure.mjd
    part = OiPure.part
    if save:
        result_dir = f"Results_{OiPure.target}_pcs"
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        plt.savefig(f"{result_dir}/fit_orient_mjd{mjd}_pt1={pt1}_pt2={pt2}_p{part}.pdf")

        np.savetxt(
            f"{result_dir}/res_orient_mjd{mjd}_pt1={pt1}_pt2={pt2}_p{part}.txt",
            np.append(res, mjd),
        )

    return res
