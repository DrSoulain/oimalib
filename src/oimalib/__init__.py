"""
OIMALIB: Optical Interferometry Modeling and Analysis Library.

This package provides tools for modeling, analyzing, and visualizing
optical interferometric data. It includes modules for handling OIFITS
files, computing observables, and generating synthetic datasets.

Submodules:
    - oifits: read and write OIFITS files
    - plotting: visualize interferometric observables
    - modelling: generate and fit physical models
    - _cli: command-line interface entry point
"""

from .alex import cube_interpolator
from .classPure import OiPure
from .data_processing import select_data, spectral_bin_data, temporal_bin_data
from .fit.dpfit import showFit as showfit
from .fitting import (
    fit_multi_size,
    fit_size,
    format_obs,
    get_mcmc_results,
    get_stat_data,
    mcmcfit,
    smartfit,
)

# from .mcfost import model, read
from .modelling import (
    compute_geom_model,
    compute_geom_model_fast,
    compute_grid_model,
    model2grid,
)
from .models import Model
from .oifits import get_condition, load
from .plotting import (
    plot_complex_model,
    plot_data_paper,
    plot_dvis,
    plot_image_model,
    plot_mcmc_results,
    plot_oidata,
    plot_plvis,
    plot_residuals,
    plot_spectra,
    plot_uv,
)

__all__ = [
    "plot_complex_model",
    "plot_data_paper",
    "plot_dvis",
    "plot_image_model",
    "plot_mcmc_results",
    "plot_oidata",
    "plot_plvis",
    "plot_residuals",
    "plot_spectra",
    "plot_uv",
]

__version__ = "0.3"
