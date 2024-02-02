import pathlib

import pytest
from matplotlib import pyplot as plt


@pytest.fixture()
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def global_datadir():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture()
def example_model_mcfost(global_datadir):
    return global_datadir / "example_model_mcfost.fits"


@pytest.fixture()
def grid_path(global_datadir):
    return global_datadir / "example_grid/"


@pytest.fixture()
def example_oifits_mat(global_datadir):
    return global_datadir / "example_MATISSE.oifits"


@pytest.fixture()
def example_oifits_grav(global_datadir):
    return global_datadir / "example_GRAVITY.oifits"


@pytest.fixture()
def example_oifits_rmat(global_datadir):
    return global_datadir / "example_MATISSE_real.oifits"


@pytest.fixture()
def example_oifits_rgrav(global_datadir):
    return global_datadir / "example_GRAVITY_real.fits"


@pytest.fixture()
def example_model(global_datadir):
    return global_datadir / "example_model_chromatic.fits"


@pytest.fixture()
def example_model_nochromatic(global_datadir):
    return global_datadir / "example_model_nochromatic.fits"


@pytest.fixture()
def example_t1(global_datadir):
    return global_datadir / "example_t1.fits"


@pytest.fixture()
def example_t2(global_datadir):
    return global_datadir / "example_t2.fits"


@pytest.fixture()
def example_t3(global_datadir):
    return global_datadir / "example_t3.fits"


@pytest.fixture()
def example_merge_ref(global_datadir):
    return global_datadir / "example_merge_reference.fits"
