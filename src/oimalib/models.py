import os
import pickle
import time

import numpy as np
import seaborn as sns
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.interpolate import interp1d

from oimalib.alex import (
    contvis,
    cube_interpolator,
    pcshift,
    pldp,
    plvis,
    shiftcomp,
    shiftfit,
    totflux,
    totphase,
    totvis,
)
from oimalib.fitting import fit_multi_size


class Model_Light:
    def __init__(self):
        return None


class Model:
    """This class is designed to read an MCFOST model and create a model class with
    all the necessary information to extract interferometric observables.

    Parameters:
    -----------
    `filename` {str}:
        Name of the model file,\n
    `mparam` {dict}:
        Parameters dictionnary obtained from the readme file. See
        oimalib.mcfost.get_model_file() for details,\n
    `distance` {float}:
        Distance in parsec (defautl: 1),\n
    `res` {float}:
        Spectral resolution in km/s (default: 60 (GRAVITY)).
    """

    def __init__(self, filename, mparam, distance=1, res=75, mcfost=True):
        self.filename = filename
        self.mcfost = mcfost

        hdu = fits.open(filename)

        self.param = mparam

        n_az = hdu[0].header.get("NAXIS5", 1)
        n_incl = hdu[0].header.get("NAXIS4", 1)
        self.nz = n_az
        self.nincl = n_incl

        nx = hdu[0].header["NAXIS1"]
        self.nx = nx
        nwl = hdu[0].header["NAXIS3"]
        self.nwl = nwl

        pixscale = hdu[0].header["CDELT2"] * 3600.0 * distance  # AU per pixel
        pixdeg = hdu[0].header["CDELT2"]  # Degree per pixel

        self.pixscale = pixscale
        self.pixdeg = pixdeg
        self.distance = distance

        # Compute the field of view in AU
        halfsize = np.asarray([nx // 2, nx // 2]) * pixscale
        extent = np.asarray([-halfsize[0], halfsize[0], -halfsize[1], halfsize[1]])

        self.extent = extent

        cube = hdu[0].data[0] if mcfost else hdu[0].data
        self.cube = cube  # Flux data from the model

        try:
            waves = hdu[1].data * 1e-3  # wavelengths are in µm
        except Exception:
            waves = hdu[1].data["wave"] * 1e-3  # wavelengths are in µm
        self.waves = waves

        lam0 = hdu[0].header["LAMBDA0"] * 1e-3
        self.lam0 = lam0
        # Position of Bry line in the model (Note: Benjamin uses the wavelength in air)

        vel = ((waves.T - lam0) / lam0 * 3e5).T  # Wavelengths in km/s
        res_vel = np.diff(vel).mean()
        dwl = np.diff(waves).mean()
        self.dwl = dwl
        print(f"[INFO] Velocity resolution of the model = {int(res_vel)} km/s")

        self.kernel = res / res_vel / 2.355

        self.vel = vel

    def compute_lcr(self, avgrange=5, display=True):
        """Extract and normalize the line to the continuum ratio. `avgrange` is the
        index range from the first (and last) item of the flux table to compute
        the left (and right) continuum values."""
        sns.set_context("talk", font_scale=1)

        # Total flux of the image per wavelength and inclination
        F2 = np.sum(self.cube, axis=(2, 3))

        self.totflux = F2
        # Average continuum left
        contflux1 = np.average(F2[:, 0:avgrange], axis=1)

        # Average continuum right
        contflux2 = np.average(F2[:, len(self.waves) - avgrange : len(self.waves)], axis=1)
        contflux = np.add(contflux1, contflux2) / 2

        fluxrat = np.array(
            [[F2[n, k] / contflux[n] for k in range(len(F2[0, :]))] for n in range(len(F2[:, 0]))]
        )
        # Total line to continuum ratio per wavelength and inclination.
        self.fluxrat = fluxrat

        if display:
            plt.figure()
            plt.plot(self.waves * 1e-3, np.squeeze(fluxrat))
            plt.xlabel("Wavelength [µm]")
            plt.ylabel("Norm. flux")
            plt.tight_layout()

    def build_coordinates(self, baseline_length, baseline_angle, pa=None, display=False):
        """Build array coordinates and rotate (if any)."""
        if pa is None:
            pa = [0]

        if isinstance(pa, float | int):
            pa = [pa]

        if len(pa) != 1:
            display = False

        self.npa = len(pa)
        self.pa = pa

        nx = self.nx
        ny = self.nx
        xvals = -np.linspace(nx / 2, -nx / 2, nx)
        yvals = np.linspace(-ny / 2, ny / 2, ny)
        # The switch in sign here is necessary since the models are usually
        # defined on the usual north-east coordinate system. This is reflected
        # in two different pixelscales in the file header (one with a minus in front),
        # but we have neglected that before so we need to take it into account here.

        rotated_baselines = np.array(
            [[bl_angle - posa for bl_angle in baseline_angle] for posa in pa]
        )

        bx = np.array(
            [
                [
                    shiftcomp(rotated_baselines[pa, bl], baseline_length[bl])[0]
                    for bl in range(len(baseline_length))
                ]
                for pa in range(len(rotated_baselines[:, 0]))
            ]
        )
        by = np.array(
            [
                [
                    shiftcomp(rotated_baselines[pa, bl], baseline_length[bl])[1]
                    for bl in range(len(baseline_length))
                ]
                for pa in range(len(rotated_baselines[:, 0]))
            ]
        )
        self.bx = bx
        self.by = by
        self.bl_pa = rotated_baselines

        blx = np.array(shiftcomp(baseline_angle, baseline_length)[0])
        bly = np.array(shiftcomp(baseline_angle, baseline_length)[1])

        # x and y coordinates of the baseline vectors in meters. Bx and By take into
        # account the global PA shift. Blx and bly are the original values
        # without the shift. If the PA is set to 0, both should be equal.
        # You only need bx and by to compute the observables.
        x, y = np.meshgrid(xvals, yvals)
        self.x = np.deg2rad(x * self.pixdeg)
        self.y = np.deg2rad(y * self.pixdeg)
        self.waves3 = self.waves * 1e-6

        bx = np.squeeze(bx)
        by = np.squeeze(by)
        blx = np.squeeze(blx)
        bly = np.squeeze(bly)

        self.nbl = len(bx)
        self.bl = baseline_length

        if display:
            plt.figure(figsize=(7, 6))
            ax = plt.gca()
            plt.title("u-v coverage")
            plt.plot(bx, by, "o", color="#349edf", label=f"PA = {pa[0]:2.1f} deg")
            plt.plot(blx, bly, "o", color="#349edf", alpha=0.2)
            plt.plot(-bx, -by, "o", color="#349edf")
            plt.plot(-blx, -bly, "o", color="#349edf", alpha=0.2)
            plt.legend(loc="best")
            plt.grid(alpha=0.1)
            plt.xlabel("u [m]")
            plt.ylabel("v [m]")
            ax.axis([-220, 220, -220, 220])
            ax.set_aspect("equal")
            plt.tight_layout()

    def get_compvis(self, conv=False):
        """Compute the complex visibility array from the different baseline
        coordinates defined by build_coordinates(). If `conv` is True, the
        observable are convolved (along the spectral domain) by the kernel
        defined by the resolution of GRAVITY (60 km/s)."""

        compvis = np.array(
            [
                [
                    [
                        [
                            np.divide(
                                np.sum(
                                    np.multiply(
                                        self.cube[n, k, :],
                                        np.exp(
                                            -2
                                            * 1j
                                            * np.pi
                                            / self.waves3[k]
                                            * (self.bx[pa, i] * self.x + self.by[pa, i] * self.y)
                                        ),
                                    )
                                ),
                                self.totflux[n, k],
                            )
                            for k in range(len(self.waves))
                        ]
                        for n in range(self.nincl)
                    ]
                    for i in range(self.nbl)
                ]
                for pa in range(self.npa)
            ]
        )
        print(
            "[INFO] Output complex visibility:",
            np.shape(compvis),
            "= (n_pa, n_bl, n_incl, n_wl)",
        )
        self.cvis = compvis

        self.visamp = np.absolute(compvis)  # visibility amplitude
        self.visphi = np.rad2deg(np.angle(compvis))
        self.contphi = self.visphi[:, :, :, 0]

        if conv:
            gauss_kernel = Gaussian1DKernel(self.kernel)
            fluxrat_conv = np.array(
                [
                    convolve(self.fluxrat[n, :], gauss_kernel, boundary="extend")
                    for n in range(self.nincl)
                ]
            )

            visphi_conv = np.array(
                [
                    [
                        [
                            convolve(
                                self.visphi[pa, bl, inc, :],
                                gauss_kernel,
                                boundary="extend",
                            )
                            for inc in range(self.nincl)
                        ]
                        for bl in range(self.nbl)
                    ]
                    for pa in range(self.npa)
                ]
            )

            visamp_conv = np.array(
                [
                    [
                        [
                            convolve(
                                self.visamp[pa, bl, inc, :],
                                gauss_kernel,
                                boundary="extend",
                            )
                            for inc in range(self.nincl)
                        ]
                        for bl in range(self.nbl)
                    ]
                    for pa in range(self.npa)
                ]
            )

            self.fluxrat = fluxrat_conv
            self.visphi = visphi_conv
            self.visamp = visamp_conv

    def interpolate_data(self, wl, kind="cubic"):
        """Interpolate the raw observables on a new wavelength grid (from real
        data or not)."""

        # Interpolate the convolved observables on the new wave grid.
        fct_fluxrat = np.array(
            [interp1d(self.waves, self.fluxrat[n, :], kind=kind) for n in range(self.nincl)]
        )

        fct_visamp = np.array(
            [
                [
                    [
                        interp1d(self.waves, self.visamp[pa, bl, inc, :], kind=kind)
                        for inc in range(self.nincl)
                    ]
                    for bl in range(self.nbl)
                ]
                for pa in range(self.npa)
            ]
        )

        fct_visphi = np.array(
            [
                [
                    [
                        interp1d(self.waves, self.visphi[pa, bl, inc, :], kind=kind)
                        for inc in range(self.nincl)
                    ]
                    for bl in range(self.nbl)
                ]
                for pa in range(self.npa)
            ]
        )

        lcr_data = np.array([fct_fluxrat[n](wl) for n in range(self.nincl)])

        visamp_data = np.array(
            [
                [
                    [fct_visamp[pa, bl, inc](wl) for inc in range(self.nincl)]
                    for bl in range(self.nbl)
                ]
                for pa in range(self.npa)
            ]
        )

        visphi_data = np.array(
            [
                [
                    [
                        fct_visphi[pa, bl, inc](wl) - fct_visphi[pa, bl, inc](wl)[0]
                        for inc in range(self.nincl)
                    ]
                    for bl in range(self.nbl)
                ]
                for pa in range(self.npa)
            ]
        )

        self.wl = wl
        self.lcr_data = lcr_data
        self.visamp_data = visamp_data
        self.visphi_data = visphi_data

        return lcr_data, visamp_data, visphi_data

    def add_disk(self, disk_size, f_c, f_h, incl):
        """Add the contribution of the disk in the visibility amplitude, phase
        and fluxes.

        Parameters:
        -----------
        `disk_size` {list}:
            Disk radius [radian],\n
        `f_c` {list}:
            Relative contribution of the disk,\n
        `f_h` {list}:
            Relative contribution of the halo,\n
        `incl` {list}:
            Inclination of the disk [degree].
        """
        if isinstance(incl, int | float):
            incl = [incl]

        self.n_size = len(disk_size)
        self.f_c = f_c
        self.f_h = f_h

        # Compute the continuum visibility of a disk (if any)
        cv = np.array(
            [
                [
                    [
                        [
                            [
                                contvis(
                                    disk_size[siz],
                                    self.bl_pa[pa, i],
                                    incl[n],
                                    self.pa[pa],
                                    x * 10 ** (-6),
                                    self.bl[i],
                                )
                                for x in self.wl
                            ]
                            for n in range(self.nincl)
                        ]
                        for i in range(self.nbl)
                    ]
                    for pa in range(self.npa)
                ]
                for siz in range(self.n_size)
            ]
        )

        self.cv = cv
        print(
            "[INFO] Output continuum visibility:",
            np.shape(cv),
            "= (n_size, n_pa, n_bl, n_incl, n_wl)",
        )

        if (len(f_c) == 1) & (len(f_h) == 1) & (f_c[0] == 0) & (f_h[0] == 0):
            ir_ex = np.array([[0.0]])
        else:
            ir_ex = np.array([[1 / (1 / (HS + DS) - 1) for HS in f_h] for DS in f_c])

        # Compute the total visibility (model+continuum) for wl_data
        tv = np.array(
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        totvis(
                                            ir_ex[dflux, hflux],
                                            cv[siz, pa, i, n, 0],
                                            self.lcr_data[n, k],
                                            self.visamp_data[pa, i, n, k],
                                            f_c[dflux],
                                        )
                                        for k in range(len(self.wl))
                                    ]
                                    for n in range(self.nincl)
                                ]
                                for i in range(self.nbl)
                            ]
                            for pa in range(self.npa)
                        ]
                        for siz in range(self.n_size)
                    ]
                    for hflux in range(len(f_h))
                ]
                for dflux in range(len(f_c))
            ]
        )
        self.tv = tv
        print(
            "[INFO] Output total visibility:",
            np.shape(tv),
            "= (n_disk_flux, n_halo, n_size, n_pa, n_bl, n_incl, n_wl)",
        )

        # Compute the total flux (add contribution of the disk and halo)
        tf = np.array(
            [
                [
                    [
                        [
                            totflux(self.lcr_data[n, k], ir_ex[dflux, hflux])
                            for k in range(len(self.wl))
                        ]
                        for n in range(self.nincl)
                    ]
                    for hflux in range(len(f_h))
                ]
                for dflux in range(len(f_c))
            ]
        )
        self.tf = tf

        # Compute the total phase (add continuum of the disk)
        tph = np.array(
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        totphase(
                                            self.lcr_data[n, k],
                                            self.visamp_data[pa, i, n, k],
                                            ir_ex[dflux, hflux],
                                            f_c[dflux],
                                            cv[siz, pa, i, n, 0],
                                            self.visphi_data[pa, i, n, k],
                                        )
                                        for k in range(len(self.wl))
                                    ]
                                    for n in range(self.nincl)
                                ]
                                for i in range(self.nbl)
                            ]
                            for pa in range(self.npa)
                        ]
                        for siz in range(self.n_size)
                    ]
                    for hflux in range(len(f_h))
                ]
                for dflux in range(len(f_c))
            ]
        )
        self.tph = tph
        return cv, tv, tph, tf

    def get_pureline(self, cond=None):
        """Compute pure line quantities (visamp and phase)."""

        if cond is None:
            cond = self.wl > 0

        plvisamp = np.array(
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        plvis(
                                            self.tv[dflux, hflux, siz, pa, i, n, k],
                                            self.tv[dflux, hflux, siz, pa, i, n, 0],
                                            self.tf[dflux, hflux, n, k],
                                        )
                                        for k in range(len(self.wl))
                                    ]
                                    for n in range(self.nincl)
                                ]
                                for i in range(self.nbl)
                            ]
                            for pa in range(self.npa)
                        ]
                        for siz in range(self.n_size)
                    ]
                    for hflux in range(len(self.f_h))
                ]
                for dflux in range(len(self.f_c))
            ]
        )

        plvisphi = np.array(
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        pldp(
                                            self.tph[dflux, hflux, siz, pa, i, n, k],
                                            self.tv[dflux, hflux, siz, pa, i, n, k],
                                            plvisamp[dflux, hflux, siz, pa, i, n, k],
                                            self.tf[dflux, hflux, n, k],
                                        )
                                        for k in range(len(self.wl))
                                    ]
                                    for n in range(self.nincl)
                                ]
                                for i in range(self.nbl)
                            ]
                            for pa in range(self.npa)
                        ]
                        for siz in range(self.n_size)
                    ]
                    for hflux in range(len(self.f_h))
                ]
                for dflux in range(len(self.f_c))
            ]
        )
        self.plvisamp = plvisamp[:, :, :, :, :, :, cond]
        self.plvisphi = plvisphi[:, :, :, :, :, :, cond]
        self.pllcr = self.lcr_data[:, cond]
        print(
            "[INFO] Output pure line visibility:",
            np.shape(self.plvisamp),
            "= (n_disk_flux, n_halo, n_size, n_pa, n_bl, n_incl, n_plwl)",
        )
        self.plwl = self.wl[cond]
        self.plvel = (self.plwl - self.lam0) / self.lam0 * 3e5
        self.cond = cond

    def get_pcs(self):
        """Compute the phocenter barycenter offset (fit a sinusoid model on the
        pure line offset phases)."""

        pureoffset = np.array(
            [
                [
                    [
                        [
                            [
                                [
                                    [
                                        pcshift(
                                            self.plwl[k],
                                            self.plvisphi[dflux, hflux, siz, pa, i, n, k],
                                            self.bl[i],
                                        )
                                        for k in range(len(self.plwl))
                                    ]
                                    for n in range(self.nincl)
                                ]
                                for i in range(self.nbl)
                            ]
                            for pa in range(self.npa)
                        ]
                        for siz in range(self.n_size)
                    ]
                    for hflux in range(len(self.f_h))
                ]
                for dflux in range(len(self.f_c))
            ]
        ) * (3600 * self.distance)

        # Save the pure line offset in AU
        self.pureoffset = pureoffset

        # Pure line offsets per baseline (in au I think)
        def _catch(x):
            try:
                return optimize.curve_fit(shiftfit, np.squeeze(self.bl_pa), x, p0=[0, 0])[0]
            except ValueError:
                return np.array([0, 0])

        coeffs2 = np.array(
            [
                [
                    [
                        [
                            [
                                [
                                    _catch(pureoffset[dflux, hflux, siz, pa, :, n, k])
                                    for k in range(len(self.plwl))
                                ]
                                for n in range(self.nincl)
                            ]
                            for pa in range(self.npa)
                        ]
                        for siz in range(self.n_size)
                    ]
                    for hflux in range(len(self.f_h))
                ]
                for dflux in range(len(self.f_c))
            ]
        )
        self.sinus_coeff = coeffs2

        barycenter = np.array(
            [
                [
                    [
                        [
                            [
                                [
                                    shiftcomp(
                                        coeffs2[dflux, hflux, siz, pa, n, k, 1],
                                        coeffs2[dflux, hflux, siz, pa, n, k, 0],
                                    )
                                    for k in range(len(self.plwl))
                                ]
                                for n in range(self.nincl)
                            ]
                            for pa in range(self.npa)
                        ]
                        for siz in range(self.n_size)
                    ]
                    for hflux in range(len(self.f_h))
                ]
                for dflux in range(len(self.f_c))
            ]
        )
        self.pcs = barycenter
        print(
            "[INFO] photocenter offset:",
            np.shape(self.pcs),
            "= (n_disk_flux, n_halo, n_size, n_pa, n_incl, n_plwl, RA/DEC)",
        )

    def get_interp_cube(self, wl=None):
        starttime = time.time()
        if wl is None:
            print("[INFO] Interpolate the cube over GRAVITY pure wavelengths...")
        else:
            print("[INFO] Interpolate the cube over the user grid wl...")
        icube = cube_interpolator(self, wl=wl)

        print(f"Done ({int(time.time() - starttime)} s).")
        self.icube = icube

    def get_size(self, param="fwhm", display=False):
        pl_size, l_gauss = fit_multi_size(self, param=param, display=display)
        self.pl_size = pl_size
        self.l_gauss = l_gauss

    def save(self, update=False, savedir=None):
        if savedir is None:
            savedir = "modelDB/"
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        incl = self.param.incl
        key = self.param.key
        phase = self.param.phase

        savedir += f"model{key}/"
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        savedir += f"incl{incl}/"
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        mlight = Model_Light()
        mlight.plwl = self.plwl
        mlight.plvisamp = self.plvisamp
        mlight.bx = self.bx
        mlight.by = self.by
        mlight.bl = self.bl
        mlight.bl_pa = self.bl_pa
        mlight.pcs = self.pcs
        mlight.cond = self.cond
        mlight.icube = self.icube
        mlight.nbl = self.nbl
        mlight.plvel = self.plvel
        mlight.wl = self.wl
        mlight.lcr_data = self.lcr_data
        mlight.pllcr = self.pllcr
        mlight.tv = self.tv
        mlight.tph = self.tph
        mlight.plvisphi = self.plvisphi
        mlight.extent = self.extent
        mlight.distance = self.distance
        mlight.lam0 = self.lam0
        mlight.param = self.param
        mlight.l_gauss = self.l_gauss
        mlight.pl_size = self.pl_size

        dpy_file = savedir + f"model_{self.param.key}_i={self.param.incl}_phase={phase:2.2f}.dpy"
        if not os.path.exists(dpy_file) or update:
            with open(dpy_file, "wb") as file:
                pickle.dump(mlight, file, protocol=2)
