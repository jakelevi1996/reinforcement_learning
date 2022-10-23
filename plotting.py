"""
MIT License

Copyright (c) 2022 JAKE LEVI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches
import numpy as np
import PIL
import util

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
        if self.has_label():
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
        if self.has_label():
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

class HVSpan(FillBetween):
    def __init__(self, xlims=None, ylims=None, **kwargs):
        self._xlims = xlims
        self._ylims = ylims
        self._kwargs = kwargs

    def plot(self, axis):
        if self._xlims is not None:
            axis.axvspan(*self._xlims, **self._kwargs)
        if self._ylims is not None:
            axis.axhspan(*self._ylims, **self._kwargs)

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
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._xlim = xlim
        self._ylim = ylim
        self._log_xscale = log_xscale
        self._log_yscale = log_yscale
        self._tight_layout = tight_layout
        self._rotate_xticklabels = rotate_xticklabels

    def apply(self, axis, figure):
        if self._xlabel is not None:
            axis.set_xlabel(self._xlabel)
        if self._ylabel is not None:
            axis.set_ylabel(self._ylabel)
        if self._xlim is not None:
            axis.set_xlim(self._xlim)
        if self._ylim is not None:
            axis.set_ylim(self._ylim)
        if self._log_xscale:
            axis.set_xscale("log")
        if self._log_yscale:
            axis.set_yscale("log")
        if self._rotate_xticklabels:
            for xtl in axis.get_xticklabels():
                xtl.set(rotation=-45, ha="left")
        if self._tight_layout:
            figure.tight_layout()

class LegendProperties:
    def __init__(self, width_ratio=0.2):
        self.width_ratio = width_ratio

def save_and_close(plot_name, dir_name, fig, verbose, file_ext="png"):
    if dir_name is None:
        dir_name = util.RESULTS_DIR
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    if len(os.path.abspath(dir_name)) + len(plot_name) > 235:
        plot_name_len = max(0, 235 - len(os.path.abspath(dir_name)))
        plot_name = plot_name[:plot_name_len] + "(...)"

    file_name = "%s.%s" % (util.clean_filename(plot_name), file_ext)
    full_path = os.path.join(dir_name, file_name)

    if verbose:
        print("Saving image in \"%s\"" % full_path)

    fig.savefig(full_path)
    plt.close(fig)

    return full_path

def plot(
    *lines,
    plot_name=None,
    dir_name=None,
    axis_properties=None,
    legend_properties=None,
    figsize=None,
    save=True,
    verbose=True,
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

    for line in lines:
        line.plot(plot_axis)

    plot_axis.grid(True, which="both")
    if plot_name is None:
        plot_name = "Output"
    if len(plot_name) > 80:
        plot_name = textwrap.fill(plot_name, width=60, break_long_words=False)
    plot_axis.set_title(plot_name)

    if legend_properties is not None:
        legend_axis.legend(
            handles=[
                line.get_handle() for line in lines if line.has_label()
            ],
            loc="center",
        )
        legend_axis.axis("off")

    if axis_properties is None:
        axis_properties = AxisProperties()

    axis_properties.apply(plot_axis, fig)

    if save:
        plot_filename = save_and_close(plot_name, dir_name, fig, verbose)
        return plot_filename

def make_gif(
    *input_paths,
    output_name=None,
    output_dir=None,
    frame_duration_ms=100,
    optimise=False,
    loop_forever=True,
    n_loops=1,
    verbose=True,
):
    if output_name is None:
        output_name = "Output"

    if output_dir is None:
        output_dir = util.RESULTS_DIR
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if loop_forever:
        n_loops = 0

    first_frame = PIL.Image.open(input_paths[0])
    file_name = "%s.gif" % util.clean_filename(output_name)
    full_path = os.path.join(output_dir, file_name)

    if verbose:
        print("Saving gif in \"%s\"" % full_path)

    first_frame.save(
        full_path,
        format="gif",
        save_all=True,
        append_images=[PIL.Image.open(f) for f in input_paths[1:]],
        duration=frame_duration_ms,
        optimise=optimise,
        loop=n_loops,
    )
    return full_path
