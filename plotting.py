import os
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches
import numpy as np
import util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "Results")

class Line:
    def __init__(self, x, y, **kwargs):
        if x is None:
            x = range(len(y))
        self._x = x
        self._y = y
        self._kwargs = kwargs

    def plot(self, axis):
        axis.plot(self._x, self._y, **self._kwargs)

    def has_label(self):
        return ("label" in self._kwargs)

    def get_handle(self):
        if self.has_label:
            if "alpha" in self._kwargs:
                alpha = self._kwargs.pop("alpha")
                handle = self._get_handle_from_kwargs(self._kwargs)
                self._kwargs["alpha"] = alpha
            else:
                handle = self._get_handle_from_kwargs(self._kwargs)

            return handle

    def _get_handle_from_kwargs(self, kwargs):
        return matplotlib.lines.Line2D([], [], **kwargs)

class FillBetween(Line):
    def __init__(self, x, y1, y2, **kwargs):
        self._x = x
        self._y1 = y1
        self._y2 = y2
        self._kwargs = kwargs

    def plot(self, axis):
        axis.fill_between(self._x, self._y1, self._y2, **self._kwargs)

    def get_handle(self):
        if self.has_label:
            return matplotlib.patches.Patch(**self._kwargs)

class HVLine(Line):
    def __init__(self, h=None, v=None, **kwargs):
        self._h = h
        self._v = v
        self._kwargs = kwargs

    def plot(self, axis):
        if self._h is not None:
            axis.axhline(self._h, **self._kwargs)
        if self._v is not None:
            axis.axvline(self._v, **self._kwargs)

class Bar(FillBetween):
    def __init__(self, x, height, **kwargs):
        self._x = x
        self._height = height
        self._kwargs = kwargs

    def plot(self, axis):
        axis.bar(self._x, self._height, **self._kwargs)

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
        log_xscale=False,
        log_yscale=False,
        tight_layout=True,
        rotate_xticklabels=False,
    ):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.log_xscale = log_xscale
        self.log_yscale = log_yscale
        self.tight_layout = tight_layout
        self.rotate_xticklabels = rotate_xticklabels

class LegendProperties:
    def __init__(self, width_ratio=0.2):
        self.width_ratio = width_ratio

def save_and_close(plot_name, fig, dir_name=None, file_ext="png"):
    if dir_name is None:
        dir_name = RESULTS_DIR
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    file_name = "%s.%s" % (util.clean_filename(plot_name), file_ext)
    full_path = os.path.join(dir_name, file_name)

    print("Saving image in \"%s\"" % full_path)
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
        line.plot(plot_axis)

    plot_axis.grid(True, which="both")
    plot_axis.set_title(plot_name)

    if legend_properties is not None:
        legend_axis.legend(
            handles=[
                line.get_handle() for line in line_list if line.has_label()
            ],
            loc="center",
        )
        legend_axis.axis("off")

    if axis_properties is not None:
        if axis_properties.xlabel is not None:
            plot_axis.set_xlabel(axis_properties.xlabel)
        if axis_properties.ylabel is not None:
            plot_axis.set_ylabel(axis_properties.ylabel)
        if axis_properties.xlim is not None:
            plot_axis.set_xlim(axis_properties.xlim)
        if axis_properties.ylim is not None:
            plot_axis.set_ylim(axis_properties.ylim)
        if axis_properties.log_xscale:
            plot_axis.set_xscale("log")
        if axis_properties.log_yscale:
            plot_axis.set_yscale("log")
        if axis_properties.rotate_xticklabels:
            for xtl in plot_axis.get_xticklabels():
                xtl.set(rotation=-45, ha="left")
        if axis_properties.tight_layout:
            fig.tight_layout()

    plot_filename = save_and_close(plot_name, fig, dir_name)
    return plot_filename
