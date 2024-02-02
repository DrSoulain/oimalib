import pytest
from astropy.io import fits

from magneto.mcfost import _azim_to_phase, find_model_key, get_model_param, get_spectrum


def test_azim_phase():
    phase1 = _azim_to_phase(90)
    phase2 = _azim_to_phase(270)

    azim = _azim_to_phase(0.3, reverse=True, clock=False)
    azim_clock = _azim_to_phase(0.3, reverse=True, clock=True)

    assert phase1 == 0
    assert phase2 == 0.5
    assert azim == 200
    assert azim_clock == -20


def test_read_log(global_datadir):
    grid_path = str(global_datadir) + "/example_grid/"
    keys = find_model_key(grid_path)[0]
    assert len(keys) == 2


def test_get_model_param(global_datadir):
    grid_path = str(global_datadir) + "/example_grid/"
    file1, mparam1 = get_model_param(1, phase=0, md=grid_path)
    file2 = get_model_param(1, phase=0.25, md=grid_path)[0]
    mparam3 = get_model_param(2, phase=0, md=grid_path)[1]

    Tmax_mod1 = 8000.0
    Tmax_mod2 = 9000.0
    assert mparam1.Tmax == Tmax_mod1
    assert mparam3.Tmax == Tmax_mod2
    assert file1 != file2


def test_get_spectrum(global_datadir):
    grid_path = str(global_datadir) + "/example_grid/"
    file1 = get_model_param(1, phase=0, md=grid_path)[0]
    hdu = fits.open(file1)

    wave, vel, spectrum = get_spectrum(hdu)
    nwl = 51
    vel_max = vel[-1].value
    assert len(wave) == nwl
    assert len(wave) == len(vel)
    assert len(wave) == len(spectrum)
    assert vel_max == pytest.approx(500, 0.1)
