"""
Created on Wed Nov  4 13:16:58 2019

@author: asoulain
"""

from math import atan2, degrees

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as cs
from scipy.interpolate import interp1d

M_sun = cs.M_sun.cgs.value
au = cs.au.cgs.value
G = cs.G.cgs.value
year = 3.1556926e7


class _StarHistory:
    """a simple container to hold the solution"""

    def __init__(self, num_steps):
        self.t = np.zeros(num_steps, np.float64)
        self.x = np.zeros(num_steps, np.float64)
        self.y = np.zeros(num_steps, np.float64)
        self.vx = np.zeros(num_steps, np.float64)
        self.vy = np.zeros(num_steps, np.float64)


class Binary:
    def __init__(self, M1, M2, a, e, theta, annotate=False):
        """
        define a binary system:
        M1 is the mass of object (star/planet) 1
        M2 is the mass of object (star/planet) 2
        a is the sum of the semi-major axes (a1 + a2)
        e is the eccentricity
        theta is an angle to rotate the semi-major axis wrt +x
        """

        self.M1 = M1
        self.M2 = M2
        self.a = a
        self.e = e
        self.theta = theta

        # determine the individual semi-major axes
        # a1 + a2 = a,  M1 a1 = M2 a2
        self.a1 = self.a / (1.0 + self.M1 / self.M2)
        self.a2 = self.a - self.a1

        # we put the center of mass at the origin
        # we put star 1 on the -x axis and star 2 on the +x axis
        self.x1_init = -self.a1 * (1.0 - self.e) * np.cos(self.theta)
        self.y1_init = -self.a1 * (1.0 - self.e) * np.sin(self.theta)

        self.x2_init = self.a2 * (1.0 - self.e) * np.cos(self.theta)
        self.y2_init = self.a2 * (1.0 - self.e) * np.sin(self.theta)

        # Kepler's laws should tell us the orbital period
        # P^2 = 4 pi^2 (a_star1 + a_star2)^3 / (G (M_star1 + M_star2))
        self.P = np.sqrt(4 * np.pi**2 * (self.a1 + self.a2) ** 3 / (G * (self.M1 + self.M2)))

        # compute the initial velocities velocities

        # first compute the velocity of the reduced mass at perihelion
        # (C&O Eq. 2.33)
        v_mu = np.sqrt(
            (G * (self.M1 + self.M2) / (self.a1 + self.a2)) * (1.0 + self.e) / (1.0 - self.e)
        )

        # then v_star2 = (mu/m_star2)*v_mu
        self.vx2_init = -(self.M1 / (self.M1 + self.M2)) * v_mu * np.sin(self.theta)
        self.vy2_init = (self.M1 / (self.M1 + self.M2)) * v_mu * np.cos(self.theta)

        # then v_star1 = (mu/m_star1)*v_mu
        self.vx1_init = (self.M2 / (self.M1 + self.M2)) * v_mu * np.sin(self.theta)
        self.vy1_init = -(self.M2 / (self.M1 + self.M2)) * v_mu * np.cos(self.theta)

        self.annotate = annotate

        self.orbit1 = None
        self.orbit2 = None

    def integrate(self, dt, tmax):
        """integrate our system to tmax using a stepsize dt"""

        # allocate storage for R-K intermediate results
        # y[0:3] will hold the star1 info, y[4:7] will hold the star2 info
        k1 = np.zeros(8, np.float64)
        k2 = np.zeros(8, np.float64)
        k3 = np.zeros(8, np.float64)
        k4 = np.zeros(8, np.float64)

        y = np.zeros(8, np.float64)

        t = 0.0

        # initial conditions

        # star 1
        y[0] = self.x1_init  # initial x position
        y[1] = self.y1_init  # initial y position

        y[2] = self.vx1_init  # initial x-velocity
        y[3] = self.vy1_init  # initial y-velocity

        # star 2
        y[4] = self.x2_init  # initial x position
        y[5] = self.y2_init  # initial y position

        y[6] = self.vx2_init  # initial x-velocity
        y[7] = self.vy2_init  # initial y-velocity

        # how many steps will we need?
        nsteps = int(tmax / dt)

        # solution storage
        s1 = _StarHistory(nsteps + 1)
        s2 = _StarHistory(nsteps + 1)

        s1.x[0] = self.x1_init
        s1.y[0] = self.y1_init
        s1.vx[0] = self.vx1_init
        s1.vy[0] = self.vy1_init

        s2.x[0] = self.x2_init
        s2.y[0] = self.y2_init
        s2.vx[0] = self.vx2_init
        s2.vy[0] = self.vy2_init

        s1.t[0] = s2.t[0] = t

        for n in range(1, nsteps + 1):
            k1[:] = dt * self.rhs(t, y, self.M1, self.M2)
            k2[:] = dt * self.rhs(t + 0.5 * dt, y[:] + 0.5 * k1[:], self.M1, self.M2)
            k3[:] = dt * self.rhs(t + 0.5 * dt, y[:] + 0.5 * k2[:], self.M1, self.M2)
            k4[:] = dt * self.rhs(t + dt, y[:] + k3[:], self.M1, self.M2)

            y[:] += (1.0 / 6.0) * (k1[:] + 2.0 * k2[:] + 2.0 * k3[:] + k4[:])

            t = t + dt

            s1.x[n] = y[0]
            s1.y[n] = y[1]
            s1.vx[n] = y[2]
            s1.vy[n] = y[3]

            s2.x[n] = y[4]
            s2.y[n] = y[5]
            s2.vx[n] = y[6]
            s2.vy[n] = y[7]

            s1.t[n] = s2.t[n] = t

        self.orbit1 = s1
        self.orbit2 = s2

    def kinetic_energies(self):
        KE1 = 0.5 * self.M1 * (self.orbit1.vx**2 + self.orbit1.vy**2)
        KE2 = 0.5 * self.M2 * (self.orbit2.vx**2 + self.orbit2.vy**2)
        return KE1, KE2

    def potential_energy(self):
        PE = (
            -G
            * self.M1
            * self.M2
            / np.sqrt((self.orbit1.x - self.orbit2.x) ** 2 + (self.orbit1.y - self.orbit2.y) ** 2)
        )
        return PE

    def rhs(self, t, y, M_star1, M_star2):
        """the RHS of our system"""

        f = np.zeros(8, np.float64)

        # y[0] = x_star1, y[1] = y_star1, y[2] = vx_star1, y[3] = vy_star1
        # y[4] = x_star2, y[5] = y_star2, y[6] = vx_star2, y[7] = vy_star2

        # unpack
        x_star1 = y[0]
        y_star1 = y[1]

        vx_star1 = y[2]
        vy_star1 = y[3]

        x_star2 = y[4]
        y_star2 = y[5]

        vx_star2 = y[6]
        vy_star2 = y[7]

        # distance between stars
        r = np.sqrt((x_star2 - x_star1) ** 2 + (y_star2 - y_star1) ** 2)

        f[0] = vx_star1  # d(x_star1) / dt
        f[1] = vy_star1  # d(y_star1) / dt

        f[2] = -G * M_star2 * (x_star1 - x_star2) / r**3  # d(vx_star1) / dt
        f[3] = -G * M_star2 * (y_star1 - y_star2) / r**3  # d(vy_star1) / dt

        f[4] = vx_star2  # d(x_star2) / dt
        f[5] = vy_star2  # d(y_star2) / dt

        f[6] = -G * M_star1 * (x_star2 - x_star1) / r**3  # d(vx_star2) / dt
        f[7] = -G * M_star1 * (y_star2 - y_star1) / r**3  # d(vy_star2) / dt

        return f


def kepler_solve(t, P, ecc):
    """Compute deformated polar coordinates for a eccentric binary."""
    maxj = 50  # Max number of iteration
    tol = 1e-8  # Convergence tolerance
    M = 2 * np.pi / P * t
    E = np.zeros(len(t))
    tj = 0
    for i in range(len(t)):
        E0 = M[i]
        # Newton's formula to solve for eccentric anomoly
        for _ in range(1, maxj + 1):
            E1 = E0 - (E0 - ecc * np.sin(E0) - M[i]) / (1 - ecc * np.cos(E0))
            if abs(E1 - E0) < tol:
                E0 = E1
            j = _
        E[i] = E1
        tj = tj + j

    # --- Compute 2-dimensional spiral angles & radii --- #
    theta = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(E / 2))
    return theta, E


def AngleBtw2Points(pointA, pointB):
    """Compute angle between 2 points in space."""
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    theta = round(degrees(atan2(changeInX, changeInY)), 2)

    if theta < 0:
        theta = 360 + theta

    return theta


def getBinaryPos(mjd, param, mjd0=57590, revol=1, v=5, au=False, anim=False, display=False):
    """Compute the spatial positon of a binary star."""
    P = param["P"]
    e = param["e"]
    M1 = param["M1"]
    M2 = param["M2"]
    dpc = param["dpc"]
    i = param["incl"]
    angleSky = param["angleSky"]
    phi = param["angle_0"]

    diff = mjd0 - mjd

    # set the masses
    M_star1 = M1 * M_sun  # star 1's mass
    M_star2 = M2 * M_sun  # star 2's mass

    P2 = P * 24 * 3600.0
    a = ((G * (M_star1 + M_star2) * P2**2) / (4 * np.pi**2)) ** (1 / 3.0)

    a_au = a / cs.au.cgs.value
    fact = diff / P

    pphase = -fact % 1  # + 0.5

    if pphase > 1:
        pphase = abs(1 - pphase)

    # set the eccentricity
    ecc = e
    theta = np.pi
    annotate = False

    b = Binary(M_star1, M_star2, a, ecc, theta, annotate=annotate)

    # set the timestep in terms of the orbital period
    dt = b.P / 1000
    tmax = revol * b.P  # maximum integration time

    b.integrate(dt, tmax)
    s1 = b.orbit1
    s2 = b.orbit2

    if au:
        dpc = 1
        unit = "AU"
    else:
        unit = "mas"
        pass

    incl, angleSky, phi = np.deg2rad(i), np.deg2rad(angleSky), np.deg2rad(-phi)

    au_cgs = cs.au.cgs.value
    X1_b, Y1_b = -s1.x / au_cgs / dpc, s1.y / au_cgs / dpc
    X2_b, Y2_b = -s2.x / au_cgs / dpc, s2.y / au_cgs / dpc

    l_theta = []
    for i in range(len(X1_b)):
        theta = AngleBtw2Points([X1_b[i], Y1_b[i]], [X2_b[i], Y2_b[i]])
        l_theta.append(theta)

    l_theta = np.array(l_theta)

    X1_rot1 = X1_b * np.cos(phi) + Y1_b * np.sin(phi)
    Y1_rot1 = -X1_b * np.sin(phi) + Y1_b * np.cos(phi)

    X2_rot1 = X2_b * np.cos(phi) + Y2_b * np.sin(phi)
    Y2_rot1 = -X2_b * np.sin(phi) + Y2_b * np.cos(phi)

    X1_rot2, Y1_rot2 = X1_rot1 * np.cos(incl), Y1_rot1
    X2_rot2, Y2_rot2 = X2_rot1 * np.cos(incl), Y2_rot1

    X1 = X1_rot2 * np.cos(angleSky) + Y1_rot2 * np.sin(angleSky)
    Y1 = -X1_rot2 * np.sin(angleSky) + Y1_rot2 * np.cos(angleSky)

    X2 = X2_rot2 * np.cos(angleSky) + Y2_rot2 * np.sin(angleSky)
    Y2 = -X2_rot2 * np.sin(angleSky) + Y2_rot2 * np.cos(angleSky)

    phase = s1.t / b.P
    r = ((s1.x[:] ** 2 + s1.y[:] ** 2) ** 0.5 + (s2.x[:] ** 2 + s2.y[:] ** 2) ** 0.5) / au_cgs / dpc

    fx1 = interp1d(phase, X1)
    fy1 = interp1d(phase, Y1)
    fx2 = interp1d(phase, X2)
    fy2 = interp1d(phase, Y2)
    fr = interp1d(phase, r)
    ftheta = interp1d(phase, l_theta)

    xmod1, ymod1 = fx1(pphase), fy1(pphase)
    xmod2, ymod2 = fx2(pphase), fy2(pphase)

    r_act = fr(pphase)
    theta_act = ftheta(pphase)

    try:
        rs = param["s_prod"] / dpc
        no_rs = False
    except KeyError:
        no_rs = True
        rs = np.nan

    # Now we fix the WR position to the center.
    x_star1, y_star1 = 0, 0
    x_star2, y_star2 = xmod2 - xmod1, ymod2 - ymod1

    X_star1, Y_star1 = X1 - X1, Y1 - Y1
    X_star2, Y_star2 = X2 - X1, Y2 - Y1

    if r.min() > rs:
        nodust = True
        days_prod = 0
    else:
        nodust = False
        if no_rs:
            cond_prod1 = (r <= 1e100) & (phase < 0.5)
            cond_prod2 = (r <= 1e100) & (phase > 0.5)
            days_prod = P
        else:
            cond_prod1 = (r <= rs) & (phase <= 0.5)
            cond_prod2 = (r <= rs) & (phase > 0.5)
            days_prod = 2 * (phase[cond_prod1].max()) * P

    tab = {
        "star1": {"x": x_star1 / param["dpc"], "y": y_star1 / param["dpc"]},
        "star2": {"x": x_star2 / param["dpc"], "y": y_star2 / param["dpc"]},
        "orbit1": {"x": X_star1, "y": Y_star1},
        "orbit2": {"x": X_star2, "y": Y_star2},
        "phase": phase,
        "r": r,
        "cond": r <= rs,
        "r_act": r_act,
        "theta_act": theta_act,
        "pphase": pphase,
        "s1": s1,
        "s2": s2,
        "l_theta": l_theta,
        "rs": rs,
        "a": a_au,
        "d_prod": days_prod,
        "f_prod": 100 * (days_prod / P),
    }

    x1, y1 = tab["star1"]["x"], tab["star1"]["y"]
    x2, y2 = tab["star2"]["x"], tab["star2"]["y"]
    t = AngleBtw2Points([x1, y1], [x2, y2])

    tab["t"] = t

    d_post_pa = pphase * P

    v = 2 * np.max([X_star2.max(), Y_star2.max(), abs(X_star2.min()), abs(Y_star2.min())])
    if display:
        xmin, xmax, ymin, ymax = -v, v, -v, v
        plt.figure(figsize=(10, 5))
        if anim:
            plt.clf()
        plt.subplot(1, 2, 1)
        plt.text(0.8 * v, 0.8 * v, rf"$\theta$ = {t:2.1f} $°$ ({d_post_pa:2.1f} d)")
        plt.plot(X_star2, Y_star2, "#008b8b", alpha=0.2, linewidth=1)
        plt.plot(x_star1, y_star1, "*", color="crimson", label="WR star")
        plt.plot(x_star2, y_star2, "*", color="#008b8b", label="O star")
        plt.vlines(0, -v, v, linewidth=1, color="gray", alpha=0.5)
        plt.hlines(0, -v, v, linewidth=1, color="gray", alpha=0.5)
        plt.legend()
        plt.xlabel(f"X [{unit}]")
        plt.ylabel(f"Y [{unit}]")
        plt.axis([xmax, xmin, ymin, ymax])
        plt.subplot(1, 2, 2)
        plt.plot(
            phase,
            r,
            linewidth=1,
            linestyle="-",
            zorder=2,
            label=r"$\phi_{{prod}}$ = {:2.1f} % ({:2.1f} d)".format(tab["f_prod"], tab["d_prod"]),
        )
        plt.plot(
            pphase,
            r_act,
            "o",
            color="#008b8b",
            zorder=3,
            label=f"r = {r_act:2.2f} {unit}",
        )
        if not nodust:
            plt.plot(phase[cond_prod1], r[cond_prod1], "-", color="#a0522d", lw=4, alpha=0.5)
            plt.plot(phase[cond_prod2], r[cond_prod2], "-", color="#a0522d", lw=4, alpha=0.5)

        if ~np.isnan(rs):
            plt.hlines(
                rs,
                0,
                1,
                linestyle="-.",
                color="#006400",
                label=rf"Threshold d$_{{nuc}}$ = {rs:2.2f}",
            )
        plt.legend(loc="best")
        plt.grid(alpha=0.2)
        plt.xlim(0, 1)
        plt.ylim(0, 2 * r.mean())
        plt.xlabel("Orbital phase")
        plt.ylabel(f"r [{unit}]")
        plt.tight_layout()
        if anim:
            plt.pause(0.3)
            plt.draw()
    return tab
