import numpy as np
import pytest

import oimalib
from oimalib.data_processing import _check_good_tel


def test_data_select_wl(example_oifits_rgrav):
    d = oimalib.load(example_oifits_rgrav)
    param_sel = {"cond_wl": True, "wave_lim": [2.14, 2.18]}
    # Expected with only wavelength selection
    cond_expected = (d.wl >= param_sel["wave_lim"][0] * 1e-6) & (
        d.wl <= param_sel["wave_lim"][1] * 1e-6
    )
    flag_vis2 = d.flag_vis2
    flag_cp = d.flag_cp
    n_expected1 = len(d.wl[cond_expected & ~flag_vis2[0]]) * len(d.vis2)
    n_expected2 = len(d.wl[cond_expected & ~flag_cp[0]]) * len(d.cp)
    n_expected = n_expected1 + n_expected2

    d_sel = oimalib.select_data(d, **param_sel)
    npts_sel = oimalib.get_stat_data([d_sel])
    assert isinstance(d, dict)
    assert n_expected == npts_sel


def test_data_select_err(example_oifits_rgrav):
    d = oimalib.load(example_oifits_rgrav)
    param_sel = {"cond_uncer": True, "rel_max": 1}
    # Expected with only uncertainty selection
    vis2 = d.vis2
    e_vis2 = d.e_vis2
    e_cp = d.e_cp
    cond_expected_v2 = (e_vis2 / vis2) <= (param_sel["rel_max"] / 100.0)
    cond_expected_cp = e_cp > 0
    flag_vis2 = d.flag_vis2
    flag_cp = d.flag_cp
    vis2_sel = vis2[cond_expected_v2 & ~flag_vis2]
    n_expected1 = len(np.hstack(vis2_sel.flatten()))
    n_expected2 = len(d.cp[cond_expected_cp & ~flag_cp].flatten())
    n_expected = n_expected1 + n_expected2

    d_sel = oimalib.select_data(d, **param_sel)
    npts_sel = oimalib.get_stat_data([d_sel])
    assert isinstance(d, dict)
    assert n_expected == npts_sel


def test_data_select_flag(example_oifits_rgrav):
    d = oimalib.load(example_oifits_rgrav)
    param_sel = {"use_flag": False}
    # Expected with only uncertainty selection
    vis2 = d.vis2
    cp = d.cp
    n_expected1 = len(np.hstack(vis2.flatten()))
    n_expected2 = len(np.hstack(cp.flatten()))
    n_expected = n_expected1 + n_expected2
    d_sel = oimalib.select_data(d, **param_sel)
    npts_sel = oimalib.get_stat_data([d_sel])
    assert isinstance(d, dict)
    assert n_expected == npts_sel


def test_check_bl(example_merge_ref):
    d = oimalib.load(example_merge_ref)
    check = _check_good_tel([d], verbose=False)
    tel_out = check[0][0]
    flag_cp = np.invert([(tel_out in x) for x in d.cpname])
    flag_vis2 = np.invert([(tel_out in x) for x in d.blname])
    assert len(d.cp[flag_cp]) == 1
    assert len(d.vis2[flag_vis2]) == 3


def test_data_temp_bin(example_t1, example_t2, example_t3, example_merge_ref):
    list_file = [example_t1, example_t2, example_t3]
    list_d = [oimalib.load(x) for x in list_file]
    d_ref = oimalib.load(example_merge_ref)
    d_bin = oimalib.temporal_bin_data(list_d)
    bl1_ref = d_ref.bl[0]
    bl2_ref = d_ref.bl[2]

    check = _check_good_tel(list_d, verbose=False)
    tel_out = check[0][0]
    flag_data_cp = np.invert([(tel_out in x) for x in d_bin.cpname])
    flag_ref_cp = np.invert([(tel_out in x) for x in d_ref.cpname])
    # Check baseline lenght (normal average)
    assert d_bin.bl[0] == bl1_ref
    assert d_bin.bl[1] == bl2_ref
    # Check vis2 (weighted average)
    assert np.all(d_bin.vis2[0] == pytest.approx(d_ref.vis2[0], 0.1))
    assert np.all(d_bin.vis2[1] == pytest.approx(d_ref.vis2[2], 0.1))
    assert np.all(d_bin.vis2[2] == pytest.approx(d_ref.vis2[4], 0.1))
    # Check cp (weighted average)
    assert np.all(d_bin.cp[flag_data_cp] == pytest.approx(d_ref.cp[flag_ref_cp], 1))


def test_data_spec_bin(example_oifits_rgrav):
    d = oimalib.load(example_oifits_rgrav)
    nbox = 50  # spectral binning box size
    d_bin = oimalib.spectral_bin_data(d, nbox=nbox)

    # Flag
    wl_min = 2.0
    wl_max = 2.2
    d_bin_flag = oimalib.spectral_bin_data(d, nbox=nbox, wave_lim=[wl_min, wl_max])
    wl_flagged = d_bin_flag.wl[~d_bin_flag.flag_vis2[0]] * 1e6

    nwl = len(d.wl)
    nwl_new = len(d_bin.wl)
    assert nwl_new == nwl // nbox
    assert wl_flagged[0] >= wl_min
    assert wl_flagged[-1] <= wl_max
