import numpy as np
import seaborn as sns
from astropy.io import fits
from matplotlib import pyplot as plt

from magneto.tools import contvis, totvis, totphase, totflux, plvis, pldp


class Model:
    """Class to read MCFOST model. It creates the model class with all the information
    needed to extract observables. The distance can be defined as `distance` in
    pc. `res` is the spectral resolution of your favorite instrument (60 km/s for
    GRAVITY), which is used to compute the kernel value (for convolution)."""

    def __init__(self, filename, distance=1, res=60):
        """"""
        self.filename = filename

        hdu = fits.open(filename)

        n_az = hdu[0].header["NAXIS5"]
        n_incl = hdu[0].header["NAXIS4"]
        self.nz = n_az
        self.nincl = n_incl

        nx = hdu[0].header["NAXIS1"]  # nx
        self.nx = nx
        # ny = lines[0].header["NAXIS2"]
        nwl = hdu[0].header["NAXIS3"]  # nlam_max
        self.nwl = nwl

        pixscale = hdu[0].header["CDELT2"] * 3600.0 * distance  # AU per pixel
        pixdeg = hdu[0].header["CDELT2"]  # Degree per pixel

        self.pixscale = pixscale
        self.pixdeg = pixdeg

        # Compute the field of view in AU
        halfsize = np.asarray([nx // 2, nx // 2]) * pixscale
        extent = np.asarray([-halfsize[0], halfsize[0], -halfsize[1], halfsize[1]])

        self.extent = extent

        cube = hdu[0].data[0]
        self.cube = cube  # Flux data from the model

        waves = hdu[1].data * 1e-3  # wavelengths are in µm
        self.waves = waves

        lam0 = hdu[0].header["LAMBDA0"] * 1e-3
        self.lam0 = lam0
        # Position of Bry line in the model (Note: Benjamin uses the wavelength in air)

        vel = ((waves.T - lam0) / lam0 * 3e5).T  # Wavelengths in km/s
        res_vel = np.diff(vel).mean()
        dwl = np.diff(waves).mean()
        self.dwl = dwl
        print("[INFO] Velocity resolution of the model = %i km/s" % (res_vel))

        self.kernel = res / res_vel / 2.355

        self.vel = vel

    def compute_lcr(self, avgrange=5, display=True):
        """Extract and normalize the line to continuum ratio."""
        sns.set_context("talk", font_scale=1)
        F2 = np.sum(
            self.cube, axis=(2, 3)
        )  # Total flux of the image per wavelength and inclination
        self.totflux = F2
        # Average continuum left
        contflux1 = np.average(F2[:, 0:avgrange], axis=1)

        # Average continuum right
        contflux2 = np.average(
            F2[:, len(self.waves) - avgrange : len(self.waves)], axis=1
        )
        contflux = np.add(contflux1, contflux2) / 2

        fluxrat = np.array(
            [
                [F2[n, k] / contflux[n] for k in range(len(F2[0, :]))]
                for n in range(len(F2[:, 0]))
            ]
        )  # Total line to continuum ratio per wavelength and inclination.
        self.fluxrat = fluxrat

        if display:
            plt.figure()
            plt.plot(self.waves, np.squeeze(fluxrat))
            plt.xlabel("Wavelength [µm]")
            plt.ylabel("Norm. flux")
            plt.tight_layout()

    def build_coordinates(self, baseline_length, baseline_angle, pa=[0], display=False):
        """Build array coordinates and rotate (if any)."""
        from magneto.tools import shiftcomp

        if (type(pa) is float) or (type(pa) is int):
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
            plt.plot(bx, by, "o", color="#349edf", label="PA = %2.1f deg" % (pa[0]))
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
        from astropy.convolution import Gaussian1DKernel, convolve

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
                                            * (
                                                self.bx[pa, i] * self.x
                                                + self.by[pa, i] * self.y
                                            )
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
        from scipy.interpolate import interp1d

        # Interpolate the convolved observables on the new wave grid.
        fct_fluxrat = np.array(
            [
                interp1d(self.waves, self.fluxrat[n, :], kind="cubic")
                for n in range(self.nincl)
            ]
        )

        fct_visamp = np.array(
            [
                [
                    [
                        interp1d(self.waves, self.visamp[pa, bl, inc, :], kind="cubic")
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
                        interp1d(self.waves, self.visphi[pa, bl, inc, :], kind="cubic")
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
        and fluxes."""
        if type(incl) is float or int:
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
