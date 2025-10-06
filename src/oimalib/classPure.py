import contextlib

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Ellipse
from scipy.constants import c as c_light
from termcolor import cprint
from uncertainties import ufloat, unumpy

from oimalib.complex_models import model_acc_mag
from oimalib.data_processing import (
    normalize_dphi_continuum,
    normalize_dvis_continuum,
    perform_fit_dphi,
    perform_fit_dvis,
)
from oimalib.fitting import err_pts_style_f, fit_flc_spectra, leastsqFit, model_pcshift
from oimalib.plotting import dic_color
from oimalib.tools import cart2pol, compute_oriented_shift


class OiPure:
    """Class to compute and analyse pure line quantities."""

    def __init__(self, data):
        self.data = data

        nbl = len(data.u)
        nwl = len(data.wl)

        self.bl_length = np.zeros(nbl)

        self.dvis = np.zeros([nbl, nwl])
        self.e_dvis = np.zeros([nbl, nwl])
        self.dphi = np.zeros([nbl, nwl])
        self.e_dphi = np.zeros([nbl, nwl])
        self.mod_dvis = np.zeros([nbl, nwl])
        self.mod_dphi = np.zeros([nbl, nwl])

        self.V_cont = np.zeros(nbl)
        self.e_V_cont = np.zeros(nbl)

        self.e_V_tot = np.zeros(nbl)

        self.u_dvis_pure = []
        self.u_dphi_pure = []
        self.target = data.info.Target[0]
        self.mjd = data.info.mjd
        with contextlib.suppress(AttributeError):
            self.part = data.info.part

    def __ufloat_quantities(self, V_tot, e_V_tot, F_tot, V_cont, e_V_cont, dphi_tot, e_dphi_tot):
        from uncertainties import ufloat

        n = len(V_tot)
        self.u_V_tot = np.array([ufloat(V_tot[i], e_V_tot) for i in range(n)])
        self.u_F_tot = np.array([ufloat(F_tot[i], self.e_flux) for i in range(n)])
        self.u_V_cont = np.array([ufloat(V_cont, e_V_cont) for i in range(n)])
        self.u_dphi_tot = np.array([ufloat(dphi_tot[i], e_dphi_tot) for i in range(n)])

    def __compute_pure_vis(self, full=True):
        """
        Compute the pure line visibility. If `full` is True, the differential
        phase is used (else approximations are performed).

        """
        u_nominator = (
            abs(self.u_V_tot * self.u_F_tot) ** 2
            + abs(self.u_V_cont) ** 2
            - (2 * self.u_V_tot * self.u_F_tot * self.u_V_cont * unumpy.cos(self.u_dphi_tot))
        )

        u_F_line = self.u_F_tot - 1

        if full:
            u_dvis_pure = u_nominator**0.5 / u_F_line
        else:
            u_dvis_pure = (self.u_F_tot * self.u_V_tot - self.u_V_cont) / (u_F_line)
        return u_dvis_pure

    def __compute_pure_phi(self):
        """Compute the pure differential phase (need pure vis first)."""
        u_dphi_pure = (
            180
            * (
                unumpy.arcsin(
                    (self.u_F_tot * self.u_V_tot * unumpy.sin(self.u_dphi_tot))
                    / ((self.u_F_tot - 1) * self.u_dvis_pure)
                )
            )
            / np.pi
        )

        return u_dphi_pure[0]

    def fit_lcr(
        self,
        r_brg=1,
        wl0=2.1661,
        red_abs=False,
        err_cont=True,
        verbose=3,
        display=True,
        norm=True,
        use_model=True,
        tellu=False,
        force_wBrg=None,
        force_restframe=None,
    ):
        """
        Fit the spectral line of GRAVITY (`lbdBrg`) and return the fitted line to
        continuum ratio `lcr` (used for pure line visibility computation).

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
            If True, the continuum is used as error.
        """
        res = fit_flc_spectra(
            self.data,
            r_brg=r_brg,
            wl0=wl0,
            red_abs=red_abs,
            err_cont=err_cont,
            verbose=verbose,
            display=display,
            norm=norm,
            use_model=use_model,
            tellu=tellu,
            force_wBrg=force_wBrg,
            force_restframe=force_restframe,
        )
        self.lcr = res["F_lc"]
        self.flux = res["flux"]
        self.e_flux = res["e_flux"]
        self.wl = res["wl"]
        self.restframe = res["restframe"]
        self.widthline = res["widthline"]
        self.inCont = res["inCont"]
        self.inLine = res["inLine"]
        self.wl_model = res["wl_model"]
        self.flux_model = res["flux_model"]
        # print("fit lcr n channel %i" % len(self.wl[self.inLine]))

    def compute_quantities_ibl(
        self,
        use_mod=True,
        use_vis2=False,
        ft=None,
        cont_vis2=True,
        ft_as_cont=False,
        use_cont_err=True,
        add_noise=None,
        verbose=False,
        display=False,
    ):
        """
        Compute the pure line quantities (vis and phase). Gaussian fit is
        performed on the data.

        Parameters
        ----------
        `use_mod` {bool}: If True, the gaussian fit are used instead of data,\n
        `cont_vis2` {bool}: If True, the continuum is set used V2,\n
        `use_cont_err` {bool}: If True, the errors are set as continuum std of
        V2,\n
        `add_noise` {list}: Add Gaussian noise (add_noise[0] is dispersion for
        dvis and add_noise[1] is the one for the phase).
        -------
        """
        from oimalib.tools import nan_interp

        self.use_mod = use_mod
        # Check inputs
        data = self.data
        inCont = self.inCont
        inLine = self.inLine

        nbl = len(data.u)
        nline = len(data.wl[inLine])
        self.V_tot = np.zeros([nbl, nline])
        self.e_dphi_tot = np.zeros(nbl)
        self.bl_pa = np.zeros_like(self.bl_length)

        if verbose:
            cprint("Extract Pure line quantities", "cyan")
            cprint("----------------------------", "cyan")
        for ibl in range(len(data.u)):
            ucoord = data.u[ibl]
            vcoord = data.v[ibl]
            bl_length, bl_pa = cart2pol(ucoord, vcoord)
            self.bl_length[ibl] = bl_length
            self.bl_pa[ibl] = bl_pa

            # Compute the value of the continuum (from vis2 or dvis)
            if cont_vis2:
                cvis_in_cont = data.vis2[ibl][inCont].copy() ** 0.5
            else:
                cvis_in_cont = data.dvis[ibl][inCont].copy()

            nan_interp(cvis_in_cont)

            if ft_as_cont:
                vis2_ft = ft.vis2[ibl]
                wl_ft = ft.wl * 1e6
                close_line = abs(wl_ft - self.restframe) < 0.12
                cont_ft = np.mean(vis2_ft[close_line] ** 0.5)
            else:
                cont_ft = cvis_in_cont.mean()

            # Normalise differential quantities
            dvis, e_dvis = normalize_dvis_continuum(
                ibl,
                data,
                inCont=inCont,
                force_cont=cont_ft,
                lbdBrg=self.restframe,
                use_vis2=use_vis2,
            )
            dphi, e_dphi = normalize_dphi_continuum(
                ibl, data, lbdBrg=self.restframe, inCont=inCont, degree=1
            )

            if add_noise is not None:
                np.random.seed(41)
                dvis += np.random.normal(0, add_noise[0], len(dvis))
                dphi += np.random.normal(0, add_noise[1], len(dphi))

            # print(use_cont_err)
            if use_cont_err:
                e_dphi = np.ones(len(e_dphi)) * np.std(dphi[inCont])
                e_dvis = np.ones(len(e_dvis)) * np.std(dvis[inCont])

            self.dvis[ibl] = dvis
            self.e_dvis[ibl] = e_dvis
            self.dphi[ibl] = dphi
            self.e_dphi[ibl] = e_dphi

            # Initial parameters for dvis fit
            param_dvis = {
                "A": dvis[inLine].max() - dvis[inCont].mean(),
                "B": dvis[inLine].max() - dvis[inCont].mean(),
                "C": dvis[inCont].mean(),
                "sigmaA": self.widthline,
                "sigmaB": self.widthline,
                "pos": self.restframe,
                "dp": 0,
            }

            # Initial parameters for dphi fit
            p1 = dphi[self.wl < self.restframe]
            p2 = dphi[self.wl > self.restframe]
            peak_phase1 = dphi.max()
            peak_phase2 = dphi.min()
            if p1.max() < p2.max():
                peak_phase1 = dphi.min()
                peak_phase2 = dphi.max()

            param_dphi = {
                "A": peak_phase1,
                "B": peak_phase2,
                "sigmaA": self.widthline / 5,
                "sigmaB": self.widthline / 5,
                "pos": self.restframe,
                "dp": self.widthline / 2.0,
            }

            # Fit the diff amp and phase by a gaussian model (single or double peaks)
            mod_dvis, _fit_dvis = perform_fit_dvis(
                self.wl,
                dvis,
                e_dvis,
                param_dvis,
                inCont=None,
                display=display,
            )
            mod_dphi, _fit_dphi = perform_fit_dphi(
                self.wl,
                dphi,
                e_dphi,
                param_dphi,
                inCont=inLine,
                display=display,
            )
            self.mod_dvis[ibl] = mod_dvis
            self.mod_dphi[ibl] = mod_dphi

            # Compute pure line vis and phi
            if use_mod:
                V_tot = mod_dvis[inLine]
                # dphi_tot = np.deg2rad(mod_dphi[inLine])
                F_tot = self.lcr[inLine]
            else:
                V_tot = dvis[inLine]
                # dphi_tot = np.deg2rad(dphi[inLine])
                try:
                    F_tot = self.data.flux[inLine]
                except IndexError:
                    F_tot = self.flux[inLine]

            dphi_tot = np.deg2rad(dphi[inLine])

            e_V_tot = np.mean(e_dvis[inLine])
            e_dphi_tot = np.mean(np.deg2rad(e_dphi[inLine]))
            # print("bl_length", bl_length, "m")
            # print("dphi[inLine]", np.deg2rad(dphi[inLine]))
            # print("e_dphi[inLine]", np.deg2rad(e_dphi[inLine]))
            # print(
            #     "rel err", 100 * np.deg2rad(e_dphi[inLine]) / np.deg2rad(dphi[inLine])
            # )
            # print("e_dphi[inLine] mean", e_dphi[inLine].mean(), "\n")

            self.V_tot[ibl] = V_tot
            self.e_V_tot[ibl] = e_V_tot
            self.e_dphi_tot[ibl] = e_dphi_tot

            # Continuum normalisation
            V_cont = dvis[inCont].mean()
            e_V_cont = dvis[inCont].std()

            if ft_as_cont:
                vis2_ft = ft.vis2[ibl]
                wl_ft = ft.wl * 1e6
                close_line = abs(wl_ft - self.restframe) < 0.12

            if verbose:
                print(
                    f"{data.blname[ibl]} Continuum amp = ",
                    ufloat(V_cont, e_V_cont),
                )

            self.V_cont[ibl] = V_cont
            self.e_V_cont[ibl] = e_V_cont
            # Format the input with error with ufloat
            self.__ufloat_quantities(V_tot, e_V_tot, F_tot, V_cont, e_V_cont, dphi_tot, e_dphi_tot)

            # Compute pure visibility and phase
            self.u_dvis_pure.append(self.__compute_pure_vis())
            self.u_dphi_pure.append(self.__compute_pure_phi())

    def __compute_pco(self):
        """ """
        from oimalib.tools import rad2mas

        wl_inline = self.wl[self.inLine]
        n_wl_inline = len(wl_inline)

        nbl = len(self.data.u)
        self.pco = np.zeros([nbl, n_wl_inline])
        self.e_pco = np.zeros([nbl, n_wl_inline])

        print()
        for ibl in range(nbl):
            tmp = np.zeros(n_wl_inline)
            e_tmp = np.zeros(n_wl_inline)
            dphi_pure = unumpy.nominal_values(self.u_dphi_pure[ibl])
            e_dphi_pure = unumpy.std_devs(self.u_dphi_pure[ibl])
            rel_err = abs(e_dphi_pure / dphi_pure)
            print(rf"{self.data.blname[ibl]} sigmaφ = {e_dphi_pure.mean():2.2f} deg")
            for iwl in range(n_wl_inline):
                pi = rad2mas(
                    (-np.deg2rad(dphi_pure[iwl]) / (2 * np.pi))
                    * ((wl_inline[iwl] * 1e-6) / (self.bl_length[ibl]))
                )
                e_pi = rel_err[iwl] * pi
                # e_pi = rad2mas(
                #     (-np.deg2rad(e_dphi_pure[iwl]) / (2 * np.pi))
                #     * ((wl_inline[iwl] * 1e-6) / (self.bl_length[ibl]))
                # )
                tmp[iwl] = pi
                e_tmp[iwl] = e_pi
            self.pco[ibl] = tmp
            self.e_pco[ibl] = e_tmp

    def plot_pco_fit(self):
        """Plot the photocenter offset (sinusoid model)."""
        err_pts_style_pco = {
            "linestyle": "None",
            "capsize": 1,
            "ecolor": "#364f6b",
            "mec": "#364f6b",
            "elinewidth": 0.5,
            "alpha": 1,
        }
        from oimalib.plotting import _update_color_bl

        pco = self.pco
        e_pco = self.e_pco
        bl_pa = self.bl_pa
        # bl_length = output_pco["bl_length"]
        wl = self.wl[self.inLine]
        nbl = len(self.data.u)
        l_blname = self.data.blname
        fit_pco = self.fit_pco

        dic_color = _update_color_bl([self.data])

        sns.set_context("talk", font_scale=0.9)
        plt.figure(figsize=(16, 8))
        nff = 2
        if len(wl) > 10:
            nff = 3
        nframe = 2
        for iwl_pc in np.argsort(wl):
            x_pc_mod = np.linspace(0, 360, 100)
            y_pc_mod = model_pcshift(x_pc_mod, fit_pco[iwl_pc]["best"])
            chi2 = fit_pco[iwl_pc]["chi2"]

            param = fit_pco[iwl_pc]["best"]
            uncer = fit_pco[iwl_pc]["uncer"]

            p1 = {
                "p": param["p"] - uncer["p"],
                "offset": param["offset"] - uncer["offset"],
            }
            p2 = {
                "p": param["p"] + uncer["p"],
                "offset": param["offset"] + uncer["offset"],
            }
            y_pc_mod1 = model_pcshift(x_pc_mod, p1)
            y_pc_mod2 = model_pcshift(x_pc_mod, p2)

            ax = plt.subplot(nff, 5, nframe)
            for i in range(nbl):
                blname = l_blname[i]
                color_bl = dic_color[blname]
                label = None
                plt.errorbar(
                    bl_pa[i],
                    pco[i, iwl_pc],
                    yerr=abs(e_pco[i, iwl_pc]),
                    label=label,
                    color=color_bl,
                    marker=".",
                    ms=11,
                    **err_pts_style_pco,
                )
                plt.errorbar(
                    bl_pa[i] - 180,
                    -pco[i, iwl_pc],
                    yerr=abs(e_pco[i, iwl_pc]),
                    color=color_bl,
                    marker="d",
                    ms=6,
                    **err_pts_style_pco,
                )

            props = dict(boxstyle="round", facecolor="#d8e9f8", alpha=1)
            plt.text(
                0.5,
                1.05,
                "λ = {:2.4f} µm\npc = {:2.1f} ± {:2.1f} µas\nθ = {:2.1f} ± {:2.1f} deg".format(
                    wl[iwl_pc],
                    fit_pco[iwl_pc]["best"]["p"] * 1000,
                    fit_pco[iwl_pc]["uncer"]["p"] * 1000,
                    fit_pco[iwl_pc]["best"]["offset"],
                    fit_pco[iwl_pc]["uncer"]["offset"],
                ),
                transform=ax.transAxes,
                verticalalignment="top",
                ha="center",
                bbox=props,
            )

            facecolor = "#d8f8dd"
            if chi2 >= 5:
                facecolor = "#e8baba"
            props2 = dict(boxstyle="round", facecolor=facecolor, alpha=1)
            plt.text(
                0.95,
                0.05,
                rf"$\chi^2$={chi2:2.1f}",
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                bbox=props2,
            )
            plt.plot(x_pc_mod, y_pc_mod, lw=1, label=f"Projected shift (chi2={chi2:2.2f})")
            plt.fill_between(x_pc_mod, y_pc_mod1, y_pc_mod2, alpha=0.5)
            # plt.legend(loc=2,)
            plt.axhline(0, lw=1, color="gray")
            plt.ylabel("Photocenter offset [mas]")
            plt.xlabel("Baseline PA [deg]")
            pc_max = y_pc_mod.max() + 0.5
            plt.ylim(-pc_max, pc_max)
            plt.xlim(0, 360)
            nframe += 1

        from oimalib.plotting import _plot_uvdata_coord

        ax = plt.subplot(2, 5, 1)
        _plot_uvdata_coord(self.data, ax=ax)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            fancybox=True,
            shadow=True,
            fontsize=10,
        )
        plt.xlim(70, -70)
        plt.ylim(-70, 70)
        plt.xlabel(r"U [M$\lambda$]")
        plt.ylabel(r"V [M$\lambda$]")
        plt.tight_layout()

    def compute_pcs(self, p=0.05, norm_error=False):
        """Compute the photocenter shift from the pure line phases."""
        # First, compute the photocenter offset
        self.__compute_pco()
        pco = self.pco
        e_pco = self.e_pco
        nbl = pco.shape[0]
        nwl = pco.shape[1]
        l_fit, l_u_x, l_u_y = [], [], []

        # Format the data to be fitted
        for j in range(nwl):
            x_pc, y_pc, e_pc = [], [], []
            for i in range(nbl):
                x_pc.append(self.bl_pa[i])
                x_pc.append(self.bl_pa[i] - 180.0)
                y_pc.append(pco[i, j])
                y_pc.append(-pco[i, j])
                e_pc.append(e_pco[i, j])
                e_pc.append(e_pco[i, j])
            x_pc = np.array(x_pc)
            y_pc = np.array(y_pc)
            e_pc = np.array(e_pc)

            flag = np.array(list(self.flag[:, 0]) * 2)

            chi2_tmp = 1e50
            y_error = e_pc[~flag]
            for o in np.arange(0, 360, 45):
                param = {"p": p, "offset": o}
                fit_tmp = leastsqFit(
                    model_pcshift,
                    x_pc[~flag],
                    param,
                    y_pc[~flag],
                    err=y_error,
                    verbose=False,
                )
                chi2 = fit_tmp["chi2"]
                if chi2 <= chi2_tmp:
                    fit_pc = fit_tmp
                    chi2_tmp = chi2

            if norm_error:
                y_error = e_pc[~flag] * np.sqrt(chi2_tmp)
                chi2_tmp = 1e50
                for o in np.arange(0, 360, 45):
                    param = {"p": p, "offset": o}
                    fit_tmp = leastsqFit(
                        model_pcshift,
                        x_pc[~flag],
                        param,
                        y_pc[~flag],
                        err=y_error,
                        verbose=False,
                    )
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
        self.fit_pco = l_fit
        self.pcs_east = pcs_east
        self.pcs_north = pcs_north

    def compute_size(
        self,
        p0,
        fitOnly=None,
        use_flag=True,
        r0=None,
        scale_err=1,
        verbose=3,
        dkpc=0.160,
        rstar=0.0093,
        display=True,
    ):
        """
        Compute the size of the line emitting region.

        Parameters
        ----------

        `p0` {dict}: initial guess (gaussian or UD),\n
        `use_flag` {bool}: If true, flag are used (reject baseline based on the
        `detection_check_dvis()` quality check function),\n
        `r0` {float}: If set, used as reference radius (simulation only).
        -------

        """
        u_dvis_pure = self.u_dvis_pure
        dvis_pure = unumpy.nominal_values(u_dvis_pure)
        e_dvis_pure = unumpy.std_devs(u_dvis_pure) * scale_err

        used_bl = np.arange(len(u_dvis_pure))

        if use_flag:
            flag_bl = self.flag_bl
            good_bl = used_bl[~flag_bl]
            bad_bl = used_bl[flag_bl]
            l_bl_bad = [self.data.bl[i] for i in bad_bl]
            l_amp_bad = np.array([np.mean(dvis_pure[i]) for i in bad_bl])
            l_err_bad = np.array([np.mean(e_dvis_pure[i]) for i in bad_bl])
            l_blname_bad = np.array([self.data.blname[i] for i in bad_bl])

        l_bl = [self.data.bl[i] for i in good_bl]
        l_amp = np.array([np.mean(dvis_pure[i]) for i in good_bl])
        l_err = np.array([np.mean(e_dvis_pure[i]) for i in good_bl])
        X = [
            [self.data.u[i] for i in good_bl],
            [self.data.v[i] for i in good_bl],
            np.array([np.mean(self.wl[self.inLine])]) * 1e-6,
        ]

        if len(l_amp) == 0:
            return None

        name_param = next(iter(p0.keys()))

        cprint(f"\nFit the line size as {name_param}", "cyan")
        cprint("-------------------------", "cyan")
        for i in range(len(l_amp)):
            blname = self.data.blname[i]
            print(f"{blname} Pure amp aver. = ", ufloat(l_amp[i], l_err[i]))
        print()
        fit = leastsqFit(
            model_acc_mag,
            X,
            p0,
            l_amp,
            l_err,
            verbose=3,
            normalizedUncer=False,
            fitOnly=fitOnly,
        )

        u = np.linspace(0, 250, 100)
        v = np.zeros(100)

        mod = model_acc_mag([u, v, X[2]], fit["best"])

        radius_mag = ufloat(fit["best"]["fwhm"] / 2.0, fit["uncer"]["fwhm"] / 2.0)
        self.r_mag = radius_mag

        txt = ""
        if r0 is not None:
            rel_diff = round(1e2 * abs(r0 - radius_mag.nominal_value) / r0, 1)
            txt = f"({r0}, Δr = {rel_diff}%)"

        cprint("------------------------------", "magenta")
        cprint(f"r_mag = {radius_mag} mas%s" % txt, "magenta")
        cprint(f"      = {radius_mag * ufloat(dkpc, 0.0004):.1uf} au%s" % txt, "magenta")
        cprint(
            f"      = {radius_mag * ufloat(dkpc, 0.0004) / ufloat(rstar, 0.15 * rstar)} Rs%s" % txt,
            "magenta",
        )
        cprint("------------------------------", "magenta")
        if display:
            err_pts_style = {
                "linestyle": "None",
                "capsize": 4,
                "ecolor": "#364f6b",
                "mec": "#364f6b",
                "marker": ".",
                "elinewidth": 3,
                "ms": 15,
            }
            # sns.reset_orig()
            # sns.set_context("talk", font_scale=0.9)
            plt.figure(figsize=(7.2, 4))
            plt.errorbar(
                l_bl,
                l_amp,
                yerr=l_err,
                color="tab:blue",
                label="Aver. pure line visibility",
                **err_pts_style,
            )
            if use_flag:
                for i in range(len(bad_bl)):
                    plt.text(
                        l_bl_bad[i] + 2,
                        l_amp_bad[i],
                        s=l_blname_bad[i],
                        fontsize=9,
                        color="tab:red",
                        alpha=0.5,
                    )
                plt.errorbar(
                    l_bl_bad,
                    l_amp_bad,
                    yerr=l_err_bad,
                    color="tab:red",
                    alpha=0.5,
                    **err_pts_style,
                )
            plt.plot(
                u,
                mod,
                color="tab:blue",
                alpha=0.5,
                label=rf"Br$\gamma$ model (r = {radius_mag} mas)",
            )
            plt.yticks([0.7, 0.8, 0.9, 1.0])
            plt.legend(loc="best")
            plt.xlim([0, 200])
            plt.xlabel("Baseline lenght [m]")
            plt.ylabel("Visibility")
            plt.grid(alpha=0.2)
            plt.tight_layout()
        return radius_mag

    def plot_cvis_pure(self, ibl, vis_range=None, phi_max=None):
        """Plot pure line visibility and phase with the spectrum."""

        def common_plot(set_flux=False):
            l1 = l2 = ""
            if set_flux:
                l1 = rf"$w$={self.widthline:2.4f} µm"
                l2 = rf"$\lambda_{{0}}$={self.restframe:2.4f} µm"

            plt.axvspan(
                self.restframe - self.widthline / 2.0,
                self.restframe + self.widthline / 2.0,
                zorder=1,
                color="#6c3bce",
                alpha=0.1,
                label=l1,
            )
            plt.axvline(self.restframe, c="#6c3bce", alpha=0.5, lw=1, label=l2)

        wl_inline = self.wl[self.inLine]
        dvis_pure = unumpy.nominal_values(self.u_dvis_pure[ibl])
        dphi_pure = unumpy.nominal_values(self.u_dphi_pure[ibl])
        e_dvis_pure = unumpy.std_devs(self.u_dvis_pure[ibl])
        e_dphi_pure = unumpy.std_devs(self.u_dphi_pure[ibl])

        if vis_range is None:
            min_dvis = np.min([np.min(dvis_pure), np.min(self.dvis[ibl])]) - 3 * np.std(
                self.dvis[ibl]
            )
            max_dvis = np.max([np.max(dvis_pure), np.max(self.dvis[ibl])]) + 3 * np.std(
                self.dvis[ibl]
            )
            vis_range = [min_dvis, max_dvis]

        if phi_max is None:
            tmp = np.max([np.max(abs(dphi_pure)), np.max(abs(self.dphi[ibl]))]) + 5 * np.std(
                self.dphi[ibl]
            )
            phi_max = np.max([tmp, 1])

        plt.figure(figsize=(9, 8))
        # sns.set_context("talk", font_scale=0.9)
        ax = plt.subplot(311)
        plt.errorbar(self.wl, self.flux, yerr=self.e_flux, **err_pts_style_f)
        plt.plot(self.wl_model, self.flux_model, alpha=0.5, color="tab:blue")
        plt.scatter(
            self.wl[self.inLine],
            self.flux[self.inLine],
            c=self.wl[self.inLine],
            cmap="coolwarm",
            s=30,
            edgecolors="k",
            linewidth=0.5,
            marker="s",
            zorder=3,
        )
        common_plot(set_flux=True)
        plt.axhline(1, lw=2, color="gray", alpha=0.2)
        plt.axhline(1 - self.e_flux, lw=2, color="crimson", alpha=0.5, ls="--")
        plt.axhline(1 + self.e_flux, lw=2, color="crimson", alpha=0.5, ls="--")
        plt.xlim(self.restframe - 5 * self.widthline, self.restframe + 5 * self.widthline)

        plt.legend(loc=1, fontsize=10)

        plt.ylabel("Norm. flux")

        ax2 = plt.subplot(312, sharex=ax)
        common_plot()

        plt.errorbar(self.wl, self.dvis[ibl], yerr=self.e_dvis[ibl], **err_pts_style_f)
        plt.scatter(
            wl_inline,
            dvis_pure,
            c=wl_inline,
            cmap="coolwarm",
            s=30,
            edgecolors="k",
            linewidth=0.5,
            marker="s",
            zorder=3,
            label="Pure line vis.",
        )

        ax2.legend(loc=1, fontsize=12)
        plt.errorbar(
            wl_inline,
            dvis_pure,
            yerr=e_dvis_pure,
            color="k",
            ls="None",
            elinewidth=1,
            capsize=1,
        )

        plt.axhline(
            self.V_cont[ibl] - self.e_V_cont[ibl],
            lw=2,
            color="crimson",
            alpha=0.5,
            ls="--",
        )
        plt.axhline(
            self.V_cont[ibl] + self.e_V_cont[ibl],
            lw=2,
            color="crimson",
            alpha=0.5,
            ls="--",
        )

        plt.plot(self.wl, self.mod_dvis[ibl], alpha=0.5, color="tab:blue")
        plt.fill_between(
            self.wl,
            self.mod_dvis[ibl] - self.e_V_tot[ibl],
            self.mod_dvis[ibl] + self.e_V_tot[ibl],
            color="tab:blue",
            alpha=0.1,
        )

        props = dict(boxstyle="round", facecolor="#d8e9f8", alpha=1)
        plt.text(
            0.05,
            1.02,
            f"{self.data.blname[ibl]} ({self.bl_length[ibl]:2.2f} m)",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=props,
        )
        plt.ylabel("Vis. Amp.")
        plt.ylim(vis_range[0], vis_range[1])

        ax3 = plt.subplot(313, sharex=ax)
        common_plot()

        plt.errorbar(self.wl, self.dphi[ibl], yerr=self.e_dphi[ibl], **err_pts_style_f)
        plt.scatter(
            wl_inline,
            dphi_pure,
            c=wl_inline,
            cmap="coolwarm",
            s=30,
            edgecolors="k",
            linewidth=0.5,
            marker="s",
            zorder=3,
            label="Pure line phase",
        )

        plt.errorbar(
            wl_inline,
            dphi_pure,
            yerr=e_dphi_pure,
            color="k",
            ls="None",
            elinewidth=1,
            capsize=1,
        )

        box1 = TextArea(f" PA = {self.bl_pa[ibl]:2.1f} deg", textprops=dict(color="k"))
        box2 = DrawingArea(22, 22, 0, 0)

        el1 = Ellipse(
            (11, 11),
            width=18,
            height=2,
            angle=90 + -self.bl_pa[ibl],
            fc=dic_color[self.data.blname[ibl]],
        )
        box2.add_artist(el1)
        box = HPacker(children=[box1, box2], align="center", pad=0, sep=5)

        anchored_box = AnchoredOffsetbox(
            loc="lower left",
            child=box,
            pad=0.0,
            frameon=True,
            bbox_to_anchor=(0.0, 1.02),
            bbox_transform=ax3.transAxes,
            borderpad=0.0,
        )

        ax3.add_artist(anchored_box)
        ax3.axhline(0, color="k", ls="--", alpha=0.2, lw=1)
        try:
            plt.plot(self.wl, self.mod_dphi[ibl], alpha=0.5, color="tab:blue")
            plt.fill_between(
                self.wl,
                self.mod_dphi[ibl] - np.rad2deg(self.e_dphi_tot[ibl]),
                self.mod_dphi[ibl] + np.rad2deg(self.e_dphi_tot[ibl]),
                color="tab:blue",
                alpha=0.1,
            )
        except Exception:
            print(f"Model not correctly fitted for the phase ({self.data.blname[ibl]}).")

        plt.axhline(
            -np.rad2deg(self.e_dphi_tot[ibl]),
            lw=2,
            color="crimson",
            alpha=0.5,
            ls="--",
        )
        plt.axhline(
            np.rad2deg(self.e_dphi_tot[ibl]),
            lw=2,
            color="crimson",
            alpha=0.5,
            ls="--",
        )

        plt.xlabel("Wavelength [µm]")
        plt.ylabel("Vis. phase [deg]")
        plt.ylim(-phi_max, phi_max)
        plt.tight_layout(pad=1.01)
        return ax2

    def plot_pcs(
        self,
        direct=True,
        dpc=None,
        xlim=180,
        vel_map=False,
        phase=None,
        rs_mas=None,
        unit="µmas",
        ax=None,
        s_marker=50,
    ):
        # [cond_sel & cond_sel_abs] * factor
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if not direct:
            pcs_east = unumpy.nominal_values(self.pcs_east)
            pcs_north = unumpy.nominal_values(self.pcs_north)
            e_pcs_east = unumpy.std_devs(self.pcs_east)
            e_pcs_north = unumpy.std_devs(self.pcs_north)
        else:
            pcs_east = unumpy.nominal_values(self.pcs_east_dir)
            pcs_north = unumpy.nominal_values(self.pcs_north_dir)
            e_pcs_east = unumpy.std_devs(self.pcs_east_dir)
            e_pcs_north = unumpy.std_devs(self.pcs_north_dir)

        if dpc is None:
            dpc = 1
            unit = "µmas"
        else:
            dpc /= 1e3
            unit = "au"

        if rs_mas is None:
            rs_mas = 1e-3
        else:
            unit = "Rstar"
            dpc = 1

        pcs_east = dpc * pcs_east / (rs_mas * 1e3)
        pcs_north = dpc * pcs_north / (rs_mas * 1e3)
        e_pcs_east = dpc * e_pcs_east / (rs_mas * 1e3)
        e_pcs_north = dpc * e_pcs_north / (rs_mas * 1e3)

        if vel_map:
            rest = self.restframe
            wl_inline = self.wl[self.inLine]
            wave = (wl_inline - rest) / rest * c_light / 1e3
        else:
            wave = self.wl[self.inLine]

        if ax is None:
            ax = plt.gca()
            plt.figure(figsize=(7.2, 6))

        sc = ax.scatter(
            pcs_east,
            pcs_north,
            c=wave,
            cmap="coolwarm",
            edgecolor="k",
            zorder=3,
            linewidth=1,
            s=s_marker,
        )
        ax.plot(pcs_east, pcs_north, ls="--", color="k", lw=1)
        ax.errorbar(
            pcs_east,
            pcs_north,
            xerr=e_pcs_east,
            yerr=e_pcs_north,
            color="k",
            ls="None",
            elinewidth=1,
            capsize=1,
        )
        clabel = "Velocity [km/s]"
        if not vel_map:
            clabel = "Wavelength [µm]"
        cbar_kws = {
            "label": clabel,
        }

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(sc, **cbar_kws, cax=cax)
        cbar.ax.tick_params(size=0)
        # if dpc is not None:
        ax.set_xlabel(f"Photocenter shift [{unit}]")
        ax.set_ylabel(f"Photocenter shift [{unit}]")
        # else:
        #     ax.set_xlabel("Photocenter shift [µas]")
        #     ax.set_ylabel("Photocenter shift [µas]")

        if phase is not None:
            rs_mas = 0.09975229076556673 * 1e3
            x_star = rs_mas * np.cos(np.linspace(0, 2 * np.pi, 100))
            y_star = rs_mas * np.sin(np.linspace(0, 2 * np.pi, 100))
            ax.plot(x_star, y_star)
            ax.text(
                0,
                0.94 * rs_mas,
                s="Star",
                color="tab:blue",
                va="top",
                ha="center",
            )
            ax.text(
                0.1,
                0.9,
                s=rf"$\phi$={phase:2.2f}",
                color="k",
                va="center",
                ha="left",
                transform=ax.transAxes,
            )

        ax.set_xlim(xlim, -xlim)
        ax.set_ylim(-xlim, xlim)
        # ax.set_aspect("equal", adjustable="box")
        plt.tight_layout(pad=1.01)
        return ax, pcs_east, pcs_north

    def detection_check_dvis(self, limit=1, exclude=None, display=True):
        """Check the detection limit for the differential visibility.
        Default `limit` is set as 1 sigma."""
        nbl = len(self.data.u)
        if exclude is None:
            exclude = []

        def common_plot(ax, max_det, frac):
            facecolor = "#d8f8dd"
            if max_det < limit:
                facecolor = "#e8baba"
            facecolor2 = "w"
            if frac < 50:
                facecolor2 = "#ebdaa3"
            props = dict(boxstyle="round", facecolor=facecolor, alpha=1)
            props2 = dict(boxstyle="round", facecolor=facecolor2, alpha=1)
            plt.text(
                0.08,
                0.92,
                rf"{max_det}$\sigma$",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=props,
            )
            plt.text(
                0.93,
                0.92,
                rf"{frac}%",
                transform=ax.transAxes,
                va="top",
                ha="right",
                bbox=props2,
            )
            plt.axhline(limit, color="#7fc389", ls="--")
            plt.xlabel("Wavelength [µm]")
            plt.ylabel(r"Detection [$\sigma$]")

        Y = self.dvis
        if self.use_mod:
            Y = self.mod_dvis

        max_detect = np.max(
            [abs((Y[ibl] - self.V_cont[ibl]) / self.e_V_cont[ibl]) for ibl in range(nbl)]
        )

        wl_in = self.wl[self.inLine]

        self.flag = np.array([[False] * len(wl_in)] * nbl)
        self.flag_bl = np.array([False] * nbl)
        self.cond_up = np.array([False] * nbl)
        if display:
            fig = plt.figure(figsize=(9, 6))
            fig.suptitle("Diff. visibility detection")
            i = 1
            for ibl in np.argsort(self.bl_length):
                diff = abs((Y[ibl] - self.V_cont[ibl]) / self.e_V_cont[ibl])
                diff_raw = abs((self.dvis[ibl] - self.V_cont[ibl]) / self.e_V_cont[ibl])
                diff_in = diff[self.inLine]

                cond_up = diff_in >= limit
                n_up = len(diff_in[cond_up]) / len(diff_in)

                frac_good = round(1e2 * n_up, 0)
                sigma_max = round(np.max(diff_in), 1)
                # sigma_mean = round(np.mean(diff_in), 1)

                ax = plt.subplot(2, 3, i)
                plt.title(f"{self.data.blname[ibl]} (%i m)" % self.bl_length[ibl])
                plt.plot(self.wl, diff)
                plt.plot(self.wl, diff_raw, ".", alpha=0.5, color="gray")
                plt.scatter(
                    wl_in[diff_in >= limit],
                    diff_in[diff_in >= limit],
                    s=30,
                    edgecolors="k",
                    color="green",
                    zorder=3,
                )
                plt.scatter(
                    wl_in[diff_in < limit],
                    diff_in[diff_in < limit],
                    s=30,
                    edgecolors="k",
                    color="red",
                    zorder=3,
                )
                plt.ylim(-0.5, max_detect + 1)
                plt.xlim(
                    self.restframe - 3 * self.widthline,
                    self.restframe + 3 * self.widthline,
                )
                common_plot(ax, sigma_max, frac_good)
                i += 1
                self.flag[ibl] = diff_in < limit
                self.flag_bl[ibl] = np.max(diff_in) < limit
                blname = self.data.blname[ibl]
                b1 = blname.split("-")[0]
                b2 = blname.split("-")[1]
                if (b1 in exclude) or (b2 in exclude):
                    self.flag_bl[ibl] = True
                print(ibl, self.flag_bl[ibl], blname)
                # self.cond_up[ibl] = cond_up
            plt.tight_layout()

    def detection_check_dphi(self, limit=1, display=True):
        """Check the detection limit for the differential phases.
        Default `limit` is set as 1 deg difference from the sigma error."""
        nbl = len(self.data.u)

        def common_plot(ax, max_det):
            facecolor = "#d8f8dd"
            if max_det < limit:
                facecolor = "#e8baba"
            props = dict(boxstyle="round", facecolor=facecolor, alpha=1)
            plt.text(
                0.08,
                0.92,
                rf"{max_det}$\sigma$",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=props,
            )
            plt.axhline(limit, color="#7fc389", ls="--")
            plt.xlabel("Wavelength [µm]")
            plt.ylabel(r"Detection [$\sigma$]")

        Y = self.dphi
        if self.use_mod:
            Y = self.mod_dphi

        max_detect = np.max(
            [abs((Y[ibl]) / np.rad2deg(self.e_dphi_tot[ibl])) for ibl in range(nbl)]
        )

        wl_in = self.wl[self.inLine]

        self.flag = np.array([[False] * len(wl_in)] * nbl)
        if display:
            fig = plt.figure(figsize=(9, 6))
            fig.suptitle("Diff. phase detection")
            i = 1
            for ibl in np.argsort(self.bl_length):
                err_phase = np.rad2deg(self.e_dphi_tot[ibl])
                diff = abs((Y[ibl]) / err_phase)
                diff_raw = abs(self.dphi[ibl] / err_phase)
                diff_in = diff[self.inLine]
                ax = plt.subplot(2, 3, i)
                plt.title(f"{self.data.blname[ibl]} (%i m)" % self.bl_length[ibl])
                plt.plot(self.wl, diff)
                plt.plot(self.wl, diff_raw, ".", alpha=0.5, color="gray")
                plt.scatter(
                    wl_in[diff_in >= limit],
                    diff_in[diff_in >= limit],
                    s=30,
                    edgecolors="k",
                    color="green",
                    zorder=3,
                )
                plt.scatter(
                    wl_in[diff_in < limit],
                    diff_in[diff_in < limit],
                    s=30,
                    edgecolors="k",
                    color="red",
                    zorder=3,
                )
                plt.ylim(-0.5, max_detect + 1)
                plt.xlim(
                    self.restframe - 3 * self.widthline,
                    self.restframe + 3 * self.widthline,
                )
                common_plot(ax, round(np.max(diff_in), 2))
                i += 1
                self.flag[ibl] = round(np.max(diff_in), 1) < limit

            plt.tight_layout()

    def compa_sc_ft(self, data_ft):
        sns.set_context("talk", font_scale=0.9)
        plt.figure(figsize=(12, 9))
        wl_ft = data_ft.wl * 1e6
        wl = self.data.wl * 1e6

        vis2 = self.data.vis2
        vis2_ft = data_ft.vis2
        inCont = self.inCont

        j = 1
        for i in np.argsort(self.data.bl):
            plt.subplot(3, 2, j)
            plt.title(f"{self.data.blname[i]} ({self.data.bl[i]} m)")
            dvis = vis2[i] ** 0.5
            dvis_ft = vis2_ft[i] ** 0.5
            vcont = np.mean(dvis[inCont])
            e_vcont = np.std(dvis[inCont])

            close_brg = abs(wl_ft - np.mean(wl)) < 0.12
            vcont_ft = np.mean(dvis_ft[close_brg])
            e_vcont_ft = np.std(dvis_ft[close_brg])

            plt.plot(wl, dvis, color="gray", zorder=5)
            plt.axhline(vcont, label=f"SC={vcont:2.2f}±{e_vcont:2.2f}", alpha=0.5)
            plt.axhline(
                vcont_ft,
                label=f"FT={vcont_ft:2.2f}±{e_vcont_ft:2.2f}",
                color="#dd8246",
                alpha=0.6,
            )
            plt.axhspan(vcont - e_vcont, vcont + e_vcont, alpha=0.1)
            plt.axhspan(vcont_ft - e_vcont_ft, vcont_ft + e_vcont_ft, alpha=0.1, color="#dd8246")
            plt.plot(wl_ft, dvis_ft, "s-", color="#e58f8f", ms=6)
            plt.legend(fontsize=10, loc=3)
            plt.xlim(2.05, 2.3)
            plt.ylim(vcont - 0.1, vcont + 0.1)
            plt.ylabel("Visibility")
            if j > 4:
                plt.xlabel("Wavelength [µm]")
            j += 1
        plt.tight_layout()

    def compute_pcs_direct(self, wvl0=2.1661178):
        wl_inline = self.wl[self.inLine]
        dphi_pure = unumpy.nominal_values(self.u_dphi_pure)
        e_dphi_pure = unumpy.std_devs(self.u_dphi_pure)

        Nchannel = len(wl_inline)  # Nb of spectral channels probed in BrGamma
        Nobs = 1  # Nb of files used (=1 when merged)
        NBL = dphi_pure.shape[0]  # Nb of baselines (=6 for GRAVITY)
        c = c_light / 1e3  # light speed (in km/s)

        vLine = (wl_inline - wvl0) / wvl0 * c_light / 1e3
        vLine = vLine.reshape(1, len(vLine))
        # array of velocities retained for the BrGamma line (in km/s) ; shape = ()

        purePhase = dphi_pure.reshape(NBL, Nobs, Nchannel)
        err_purePhase = e_dphi_pure.reshape(NBL, Nobs, Nchannel)
        # purePhase = array of pure line phases (in degrees) ; shape = (NBL, Nobs, Nchannel)
        # err_purePhase = array of absolute  errors in pure line phases (in degrees)

        B = self.bl_length  # array of projected baselines length (in meters)
        Bangle = self.bl_pa  # array of baselines' PA (in degrees)

        ### Creating empty photocenter shifts + error arrays
        x = np.zeros((Nchannel, Nobs)).T
        err_x = np.zeros((Nchannel, Nobs)).T
        y = np.zeros((Nchannel, Nobs)).T
        err_y = np.zeros((Nchannel, Nobs)).T

        #### Creating the u-v coverage matrix
        B_mat = np.array(
            [B * np.cos(Bangle * np.pi / 180), B * np.sin(Bangle * np.pi / 180)]
        ).reshape(2, 6, 1)

        ## Should be possible to vectorize the loops to gain time
        # Looping over spectral channels
        for i in range(Nchannel):
            # Looping over single files
            for j in range(Nobs):
                # Extracting the pure-line phase + error corresponding to given obs + channel
                PHI = purePhase[:, j, i] * np.pi / 180
                dPHI = err_purePhase[:, j, i] * np.pi / 180
                # Computing the pseudo-invert matrix of u-v coverage at this obs + channel
                Bi = np.linalg.pinv(B_mat[:, :, j])
                wvl = wvl0 * (1 + vLine[j, i] / c) * 1e-6
                # Computing the photocenter shifts  (in mas)
                P = -wvl / 2 / np.pi * np.dot(Bi.T, PHI) * 180 / np.pi * 3600000
                dP = -wvl / 2 / np.pi * np.dot(Bi.T, dPHI) * 180 / np.pi * 3600000

                # Adding values to x,y arrays
                x[j, i], y[j, i] = P[0], P[1]
                err_x[j, i], err_y[j, i] = abs(dP[0]), abs(dP[1])

        pcs_east = []
        pcs_north = []
        for i in range(x.shape[1]):
            pcs_east.append(ufloat(y[0, i], err_y[0, i]))
            pcs_north.append(ufloat(x[0, i], err_x[0, i]))
        self.pcs_east_dir = np.array(pcs_east) * 1e3
        self.pcs_north_dir = np.array(pcs_north) * 1e3
