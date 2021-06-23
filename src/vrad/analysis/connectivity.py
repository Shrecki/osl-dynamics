import numpy as np
from nilearn.plotting import plot_connectome
from visbrain.objects import BrainObj, ConnectObj, SceneObj
from vrad.files import mask
from vrad.utils.parcellation import Parcellation


def exclude_by_sigma(edges, sigma=1):
    edges = edges.copy()
    np.fill_diagonal(edges, np.nan)

    mean = edges[~np.isnan(edges)].mean()
    std = edges[~np.isnan(edges)].std()

    np.fill_diagonal(edges, mean)
    selection = (edges >= (mean + sigma * std)) & (edges <= (mean - sigma * std))
    return selection


def std_filter(arr, sigma=0.95):
    copy = arr.copy()

    mean = copy.mean()
    std = copy.std()

    low_pass = mean - sigma * std
    high_pass = mean + sigma * std

    copy[(copy > low_pass) & (copy < high_pass)] = 0
    return copy


def make_symmetric(arr, upper=True):
    copy = arr.copy()
    index = np.tril_indices(copy.shape[-1], -1)
    copy[..., index[0], index[1]] = np.inf
    return np.minimum(copy, np.swapaxes(copy, -2, -1))


def plot_connectivity(edges, parcellation, sigma=0, **kwargs):
    filtered = std_filter(edges, sigma)
    if not isinstance(parcellation, Parcellation):
        parcellation = Parcellation(parcellation)
    plot_connectome(filtered, parcellation.roi_centers(), **kwargs)


def visbrain_plot_connectivity(edges, parcellation, *, inflation=0, selection=None):
    if isinstance(parcellation, str):
        parcellation = Parcellation(parcellation)
    nodes = parcellation.roi_centers()

    scene = SceneObj()

    c = ConnectObj(
        "Connect", nodes, edges, select=selection, cmap="inferno", antialias=True
    )

    points, triangles = mask.get_surf(inflation)
    b = BrainObj("Custom", vertices=points, faces=triangles)

    scene.add_to_subplot(c)
    scene.add_to_subplot(b)

    scene.preview()
