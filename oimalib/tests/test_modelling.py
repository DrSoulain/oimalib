import munch
import numpy as np
import pytest

import oimalib


@pytest.mark.parametrize("choice", [True, False])
def test_grid(example_model, choice):
    grid = oimalib.model2grid(example_model, fliplr=choice)

    wl0 = grid.wl
    tmp = 50.0 / wl0
    x = grid.sign * tmp
    y = tmp
    pts = (wl0, x, y)

    cv_real = grid.real(pts)
    cv_imag = grid.imag(pts)
    cond_all_inf_1 = False in (cv_imag <= 1)
    assert isinstance(grid, munch.Munch)
    assert isinstance(cv_real, np.ndarray)
    assert isinstance(cv_imag, np.ndarray)
    assert ~cond_all_inf_1


def test_grid_user(example_model_nochromatic):
    grid = oimalib.model2grid(
        example_model_nochromatic,
        wl_user=np.array([3.0]),
        pix_user=1.0,
        light=False,
    )

    assert isinstance(grid, munch.Munch)


def test_compute_model_grid(example_model, example_oifits_rmat):
    d = oimalib.load(example_oifits_rmat)
    grid = oimalib.model2grid(example_model)
    mod_v2_grid, mod_cp_grid = oimalib.compute_grid_model(d, grid, verbose=True)
    ncp = len(d.cp)
    nbl = len(d.vis2)
    nwl = len(d.wl)
    assert np.shape(mod_v2_grid)[1] == nbl
    assert np.shape(mod_cp_grid)[1] == ncp
    assert np.shape(mod_cp_grid)[2] == nwl


# def test_compute_model_grid_user(example_model_nochromatic, example_oifits_rmat):
#     d = oimalib.load(example_oifits_rmat)
#     grid = oimalib.model2grid(
#         example_model_nochromatic, wl_user=np.array([3e-6]), pix_user=1.0
#     )
#     mod_v2_grid, mod_cp_grid = oimalib.compute_grid_model(d, grid, verbose=True)
#     ncp = len(d.cp)
#     nbl = len(d.vis2)
#     nwl = len(d.wl)
#     assert np.shape(mod_v2_grid)[1] == nbl
#     assert np.shape(mod_cp_grid)[1] == ncp
#     assert np.shape(mod_cp_grid)[2] == nwl


def test_model_disk(example_oifits_rmat):
    param = {"model": "disk", "x0": 0, "y0": 0, "diam": 60}
    d = oimalib.load(example_oifits_rmat)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


def test_model_binary(example_oifits_grav):
    param = {"model": "binary", "x0": 0, "y0": 0, "sep": 3, "pa": 45, "dm": 3}
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


def test_model_rbinary(example_oifits_grav):
    param = {
        "model": "binary_res",
        "x0": 0,
        "y0": 0,
        "sep": 3,
        "pa": 45,
        "dm": 3,
        "diam": 1,
    }
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


def test_model_edisk(example_oifits_grav):
    param = {
        "model": "edisk",
        "x0": 0,
        "y0": 0,
        "majorAxis": 3,
        "pa": 0,
        "incl": 45,
    }
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


def test_model_ddisk(example_oifits_grav):
    param = {
        "model": "debrisDisk",
        "x0": 0,
        "y0": 0,
        "majorAxis": 3,
        "pa": 90,
        "incl": 45,
        "cr": 1,
        "w": 0.5,
    }
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


def test_model_cdisk(example_oifits_grav):
    param = {
        "model": "clumpyDebrisDisk",
        "x0": 0,
        "y0": 0,
        "majorAxis": 3,
        "pa": 30,
        "incl": 45,
        "w": 0.5,
        "d_clump": 0.5,
        "pa_clump": 45,
        "fc": 10,
        "fs": 10,
    }
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


def test_model_gdisk(example_oifits_grav):
    param = {
        "model": "gdisk",
        "x0": 0,
        "y0": 0,
        "fwhm": 3,
    }
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


def test_model_egdisk(example_oifits_grav):
    param = {"model": "egdisk", "x0": 0, "y0": 0, "majorAxis": 3, "pa": 90, "incl": 45}
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


def test_model_ering(example_oifits_grav):
    param = {
        "model": "ering",
        "x0": 0,
        "y0": 0,
        "majorAxis": 3,
        "pa": 90,
        "incl": 45,
        "kr": -1,
    }
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


def test_model_lor(example_oifits_grav):
    param = {
        "model": "lor",
        "x0": 0,
        "y0": 0,
        "fwhm": 3,
    }
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


def test_model_yso(example_oifits_grav):
    param = {
        "model": "yso",
        "incl": 45,
        "hfr": 1.5,
        "fh": 0.0,
        "fc": 0.9,
        "pa": 0,
        "x0": 0,
        "y0": 0,
        "flor": 0,
    }
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


@pytest.mark.parametrize("types", ["smooth", "uniform", "no"])
def test_model_lazareff(example_oifits_grav, types):
    param = {
        "model": "lazareff",
        "incl": 45,
        "la": 0.2385,
        "lk": -1,
        "fc": 1,
        "fs": 0.0,
        "pa": 0,
        "x0": 0,
        "y0": 0,
        "flor": 0,
        "cj": 0,
        "sj": 1,
        "kc": 0,
        "ks": 0,
        "type": types,
    }
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


@pytest.mark.parametrize("types", ["smooth", "uniform", "no"])
def test_model_hlazareff(example_oifits_grav, types):
    param = {
        "model": "lazareff_halo",
        "incl": 45,
        "la": 0.2385,
        "lk": -1,
        "fc": 0.8,
        "fh": 0.1,
        "pa": 0,
        "x0": 0,
        "y0": 0,
        "flor": 0,
        "cj": 0,
        "sj": 1,
        "kc": 0,
        "ks": 0,
        "type": types,
    }
    d = oimalib.load(example_oifits_grav, simu=True)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


def test_model_pwhl(example_oifits_rmat):
    param = {}
    param["model"] = "pwhl"
    # Spiral parameters
    # -----------------
    # r: thin ring, 'e_ud': uniform disk, 'e_gd' : gaussian disk
    param["compo"] = "r"
    param["rounds"] = 1.5  # elapsed days before mjd in fraction of period (N coils)
    param["step"] = 65  # [mas]
    param["opening_angle"] = 55.0  # [deg] full opening angle alpha
    param["T_sub"] = 2000
    param["q"] = 0.45
    param["r_nuc"] = 10  # [mas]
    param["gap_factor"] = 1.5  # 1.5 in T is 10 is flux (roughtly)
    param["nelts"] = 200
    param["thickness"] = 1

    # Orientation in the sky
    param["angle_0"] = 0
    param["angleSky"] = 0
    param["incl"] = 0

    # SED parameters
    # --------------
    param["contrib_star"] = 10  # [%] contribution of the star @ 1 µm
    param["f_scale_pwhl"] = 1 / 4.0  # Scaling factor of the SED

    # Phase parameters
    # ----------------
    param["mjd0"] = 50000  # periastron date
    param["mjd"] = 50000 + 120

    # Binary parameters
    # -----------------
    param["P"] = 241.5  # [days]
    param["L_WR/O"] = 2.0
    param["T_WR"] = 45000.0
    param["T_OB"] = 30000.0
    param["dpc"] = 2.52
    param["e"] = 0.0
    param["M1"] = 10.0
    param["M2"] = 20.0

    # Enveloppe parameters
    # --------------------
    param["contrib_halo"] = 0

    # Absolute position
    # -----------------
    param["x0"], param["y0"] = 0, 0
    d = oimalib.load(example_oifits_rmat, simu=False)
    nbl = len(d.vis2)
    ncp = len(d.cp)
    mod_v2, mod_cp = oimalib.compute_geom_model(d, param)
    assert np.shape(mod_v2)[1] == nbl
    assert np.shape(mod_cp)[1] == ncp


@pytest.mark.parametrize("ncore", [1])
def test_model_fast(example_oifits_rmat, ncore):
    d = oimalib.load(example_oifits_rmat)
    fake_list_d = [d, d, d]
    param = {"model": "gdisk", "fwhm": 3}
    mod = oimalib.compute_geom_model_fast(fake_list_d, param, ncore=ncore)
    ninput = len(fake_list_d)
    nmod = len(mod)
    shape_mod = mod[0]["vis2"].shape
    shape_dat = d.vis2.shape
    assert nmod == ninput
    assert type(mod) == list
    assert shape_mod == shape_dat
