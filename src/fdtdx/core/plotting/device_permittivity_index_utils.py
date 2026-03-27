import jax
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fdtdx.core.misc import get_background_material_name
from fdtdx.materials import Material, compute_ordered_names
from fdtdx.typing import ParameterType


def index_matrix_to_str(indices: jax.Array) -> str:
    """Converts a 2D matrix of indices to a formatted string representation.

    Args:
        indices (jax.Array): A 2D JAX array containing numerical indices.

    Returns:
        str: A string representation of the matrix where each row is space-separated
        and rows are separated by newlines.
    """
    indices_str = ""
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            indices_str += str(indices[i, j].squeeze()) + " "
        indices_str += "\n"
    return indices_str


def device_matrix_index_figure(
    device_matrix_indices: jax.Array,
    material: dict[str, Material],
    parameter_type: ParameterType,
) -> Figure:
    """Creates a visualization figure of device matrix indices with permittivity configurations.

    Args:
        device_matrix_indices (jax.Array): A 3D JAX array containing the device matrix indices.
            Shape should be (height, width, channels) where channels is typically 1.
        material (dict[str, Material]): A tuple of (name, value) pairs defining the permittivity
            configurations, where name is a string identifier (e.g., "Air") and value
            is the corresponding permittivity value.
        parameter_type (ParameterType): Type of the parameters to be plotted

    Returns:
        Figure: A matplotlib Figure object containing the visualization with:
        - A heatmap of the device matrix indices
        - Color-coded regions based on permittivity configurations
        - Optional text labels showing index values (for smaller matrices)
        - A legend mapping colors to permittivity configurations
        - Proper axis labels and grid settings
    """
    assert device_matrix_indices.ndim == 3
    fig, ax = plt.subplots(figsize=(12, 12))
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    image_palette = sns.color_palette("YlOrBr", as_cmap=True)
    ordered_name_list = compute_ordered_names(material)
    background_name = get_background_material_name(material)
    foreground_names = [n for n in ordered_name_list if n != background_name]

    # Compute per-cell fill fraction in [0, 1]: 0 = all background, 1 = all foreground.
    # For continuous output the indices already interpolate between 0 and 1.
    # For discrete/binary output, mean(axis=-1) gives the fraction of foreground layers.
    values = np.array(device_matrix_indices, dtype=np.float32).mean(axis=-1)

    cax = ax.imshow(
        values.T,
        cmap=image_palette,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        origin="lower",
    )
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    fig.colorbar(cax, ax=ax, label=f"fill fraction  (0={background_name}, 1={'/'.join(foreground_names)})")
    ax.set_aspect("equal")
    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_alpha(0.0)
    return fig
