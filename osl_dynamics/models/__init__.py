"""Implemented models.

"""

import yaml

from osl_dynamics.models import (
    dynemo,
    mdynemo,
    hmm,
    hmm_poi,
    hmm_wishart,
    hive,
    dive,
)
from osl_dynamics.utils import misc

models = {
    "DyNeMo": dynemo.Model,
    "M-DyNeMo": mdynemo.Model,
    "HMM": hmm.Model,
    "HMM-Poisson": hmm_poi.Model,
    "HMM-Wishart": hmm_wishart.Model,
    "HIVE": hive.Model,
    "DIVE": dive.Model,
}


def load(dirname, single_gpu=True):
    """Load model from dirname.

    Parameters
    ----------
    dirname : str
        Path to directory where the config.yml and weights are stored.
    single_gpu : bool, optional
        Should we compile the model on a single GPU?

    Returns
    -------
    model : DyNeMo model
        Model object.
    """
    with open(f"{dirname}/config.yml", "r") as file:
        config_dict = yaml.load(file, misc.NumpyLoader)

    if "model_name" not in config_dict:
        raise ValueError(
            "Either use a specific `Model.load` method or "
            + "provide a `model_name` field in config"
        )

    try:
        model_type = models[config_dict["model_name"]]
    except KeyError:
        raise NotImplementedError(
            f"{config_dict['model_name']} was not found. "
            + f"Options are {', '.join(models.keys())}"
        )

    return model_type.load(dirname, single_gpu=single_gpu)
