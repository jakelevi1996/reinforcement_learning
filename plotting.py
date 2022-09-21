import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches
import numpy as np
import util

class Line:
    def __init__(self, x, y, **kwargs):
        """
        Initialise a Line object. x and y are iterables containing the
        horizontal and vertical coordinates of the data to be plotted for this
        line, and **kwargs contains keyword-arguments passed to axis.plot, for
        example:

        - `c=r`             - Plot each line segment/marker in red
        - `ls="--"`         - Plot each line segment using a dashed line-style
        - `marker=o`        - Plot each data point using circular markers
        - `alpha=0.8`       - Make these lines/markers 80% opaque (IE 20%
          transparent)
        - `zorder=20`       - Make these lines/markers appear in front of
          anything with `zorder` < 20 and behind anything with `zorder` > 20
        - `label="Line 1"`  - If a legend is being used, then add an entry to
          the legend for this line with the label "Line 1"

        For more information on available keyword arguments, see the
        documentation for matplotlib.pyplot.plot at
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
        """
        if x is None:
            x = range(len(y))
        self._x = x
        self._y = y
        self._kwargs = kwargs

    def plot(self, axis):
        """
        Plot this object on the given axis, using the keyword arguments with
        which this object was initialised
        """
        axis.plot(self._x, self._y, **self._kwargs)

    def has_label(self):
        """
        Return True if this object has been initialised with the `label`
        keyword, and is therefore intended to be included in the legend, if a
        legend is being used. Otherwise return False
        """
        return ("label" in self._kwargs)

    def get_handle(self):
        """
        Get a handle for this Line object, to be used as a legend entry
        """
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
        """
        Initialise a FillBetween object, which can be useful for example for
        plotting standard deviations. x is an iterable containing the
        horizontal coordinates for the lower and upper edges of the filled
        shape, y1 contains the vertical coordinates for the lower edge of the
        filled shape, and y2 contains the vertical coordinates for the upper
        edge of the filled shape. **kwargs contains keyword-arguments used for
        plotting, as described in the docstring for the initialiser for the
        Line class, but note that if a colour is specified for this shape then
        it should be specified using the `color` keyword argument, as `c`
        cannot be used as a shorthand keyword argument
        """
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
        """
        Initialise a HVLine object, which is used to plot a horizontal and/or
        vertical line. h is the vertical coordinate for a horizontal line. If h
        is None, no horizontal line is plotted. v is the horizontal coordinate
        for a vertical line. If v is None, no vertical line is plotted.
        **kwargs contains keyword-arguments used for plotting, as described in
        the docstring for the initialiser for the Line class.
        """
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
        """
        Initialise a Bar object, which is useful for plotting bar charts. x can
        be set to the horizontal coordinate of the bar, or a string to describe
        a categorical variable, and height should be a number which specified
        the height of the bar. **kwargs contains keyword-arguments used for
        plotting, as described in the docstring for the initialiser for the
        Line class, but note that if a colour is specified for this shape then
        it should be specified using the `color` keyword argument, as `c`
        cannot be used as a shorthand keyword argument
        """
        self._x = x
        self._height = height
        self._kwargs = kwargs

    def plot(self, axis):
        axis.bar(self._x, self._height, **self._kwargs)

class ColourPicker:
    def __init__(self, num_colours, cyclic=True, cmap_name=None):
        """
        Initialise a ColourPicker object, which is used to generate a
        series of unique colours to be plotted. num_colours should be equal to
        the number of different colours that will be plotted. Once initialised,
        this object can be called with an integer >=0 and < num_colours, which
        returns a colour which is unique to that integer, and can be passed to
        an object which is to be plotted
        """
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
        dir_name = util.RESULTS_DIR
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
    """
    Plot a series of shapes on a graph, apply formatting, save to disk as an
    image file, and return the full path (directory and filename) of the output
    image file. This function accepts the following arguments:

    - line_list should be a list of instances of the Line class or its
      subclasses (EG FillBetween, HVLine, Bar) which will be plotted
    - plot_name should be the title of the plot (which if greater than 80
      characters is wrapped to line lengths of 60 characters), which is also
      used as the filename for the image which is saved to disk (special
      characters and whitespace are first converted to underscores)
    - dir_name (optional) is the directory in which the output image is saved
      (by default, the image is saved in a subdirectory of the current
      directory called "Results")
    - axis_properties (optional) should be an instance of the AxisProperties
      class, and can be used to specify labels and limits for the horizontal
      and vertical axes, whether each axis should be scaled logarithmically,
      whether the horizontal axis tick labels should be rotated (which can be
      useful EG for bar charts with verbose categorical independent variables
      to prevent the labels from overlapping), etc
    - legend_properties (optional) should be an instance of LegendProperties,
      and can be used to specify the relative width of the legend. If this
      argument is provided, then a legend will be added to the figure
    - figsize (optional) should be a list or tuple of length 2, and specifies
      the width and height for the figure in inches
    """
    # Initialise subplots, with/without a legend depending on if the
    # legend_properties argument is provided
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

    # Plot lines
    for line in line_list:
        line.plot(plot_axis)

    # Set grid and title
    plot_axis.grid(True, which="both")
    if len(plot_name) > 80:
        plot_name = textwrap.fill(plot_name, width=60, break_long_words=False)
    plot_axis.set_title(plot_name)

    # If the legend_properties argument was provided then create a legend
    if legend_properties is not None:
        legend_axis.legend(
            handles=[
                line.get_handle() for line in line_list if line.has_label()
            ],
            loc="center",
        )
        legend_axis.axis("off")

    # Set axis properties
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

    # Save and close the figure and return the full filename of the plot
    plot_filename = save_and_close(plot_name, fig, dir_name)
    return plot_filename
