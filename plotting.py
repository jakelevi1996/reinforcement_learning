import os
import matplotlib.pyplot as plt
import matplotlib.lines
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "Results")

class Line:
    def __init__(
        self,
        x,
        y,
        colour="b",
        line_style="-",
        marker_style=None,
        alpha=None,
        zorder=None,
        label=None,
        fill=None,
    ):
        if x is None:
            x = range(len(y))
        self.x = x
        self.y = y
        self.label = label
        self.fill = fill
        self._colour = colour
        self._line_style = line_style
        self._marker_style = marker_style
        self._alpha = alpha
        self._zorder = zorder

    def get_kwargs(self):
        kwargs = dict()
        if self._colour is not None:
            kwargs["c"] = self._colour
        if self._line_style is not None:
            kwargs["ls"] = self._line_style
        if self._marker_style is not None:
            kwargs["marker"] = self._marker_style
        if self._alpha is not None:
            kwargs["alpha"] = self._alpha
        if self._zorder is not None:
            kwargs["zorder"] = self._zorder
        if self.label is not None:
            kwargs["label"] = self.label

        return kwargs


class ColourPicker:
    def __init__(self, num_colours, cyclic=True, cmap_name=None):
        if cmap_name is None:
            if cyclic:
                cmap_name = "hsv"
            else:
                cmap_name = "cool"
        if cyclic:
            endpoint = False
        else:
            endpoint = True

        cmap = plt.get_cmap(cmap_name)
        cmap_sample_points = np.linspace(0, 1, num_colours, endpoint)
        self._colours = [cmap(i) for i in cmap_sample_points]

    def __call__(self, colour_ind):
        return self._colours[colour_ind]

class SubplotProperties:
    def __init__(self, num_x, num_y):
        self._num_x = num_x
        self._num_y = num_y

class AxisProperties:
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        tight_layout=False,
    ):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.tight_layout = tight_layout

class LegendProperties:
    def __init__(self, width_ratio=0.2):
        self.width_ratio = width_ratio

def save_and_close(plot_name, fig, dir_name=None, file_ext="png"):
    if dir_name is None:
        dir_name = RESULTS_DIR
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    plot_name_clean = "".join(
        c for c in plot_name if c.isalnum() or c in " -_.,"
    )
    file_name = "%s.%s" % (plot_name_clean, file_ext)
    full_path = os.path.join(dir_name, file_name)
    fig.savefig(full_path)
    plt.close(fig)

    return full_path

def plot(
    line_list,
    plot_name,
    dir_name=None,
    axis_properties=None,
    legend_properties=None,
    figsize=None,
):
    if legend_properties is not None:
        if figsize is None:
            figsize = [10, 6]
        gridspec_kw = {"width_ratios": [1, legend_properties.width_ratio]}
        sp = plt.subplots(1, 2, figsize=figsize, gridspec_kw=gridspec_kw)
        fig, (plot_axis, legend_axis) = sp
    else:
        if figsize is None:
            figsize = [8, 6]
        fig, plot_axis = plt.subplots(1, 1, figsize=figsize)

    for line in line_list:
        plot_axis.plot(line.x, line.y, **line.get_kwargs(), figure=fig)

    plot_axis.grid(True)
    plot_axis.set_title(plot_name)

    if axis_properties is not None:
        if axis_properties.tight_layout:
            fig.tight_layout()
        if axis_properties.xlabel is not None:
            plot_axis.set_xlabel(axis_properties.xlabel)
        if axis_properties.ylabel is not None:
            plot_axis.set_ylabel(axis_properties.ylabel)
        if axis_properties.xlim is not None:
            plot_axis.set_xlim(axis_properties.xlim)
        if axis_properties.ylim is not None:
            plot_axis.set_ylim(axis_properties.ylim)

    if legend_properties is not None:
        legend_axis.legend(
            handles=[
                matplotlib.lines.Line2D([], [], **line.get_kwargs())
                for line in line_list if line.label is not None
            ],
            loc="center",
        )
        legend_axis.axis("off")

    save_and_close(plot_name, fig, dir_name)
