import os
import matplotlib.pyplot as plt

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

        return kwargs


class ColourPicker:
    def __init__(self, cmap=None):
        self._cmap = cmap

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
    figsize=[8, 6],
):
    fig = plt.figure(figsize=figsize)
    for line in line_list:
        plt.plot(line.x, line.y, **line.get_kwargs(), figure=fig)

    axis = fig.axes[0]
    axis.grid(True)
    axis.set_title(plot_name)

    if axis_properties is not None:
        if axis_properties.tight_layout:
            fig.tight_layout()
        if axis_properties.xlabel is not None:
            axis.set_xlabel(axis_properties.xlabel)
        if axis_properties.ylabel is not None:
            axis.set_ylabel(axis_properties.ylabel)
        if axis_properties.xlim is not None:
            axis.set_xlim(axis_properties.xlim)
        if axis_properties.ylim is not None:
            axis.set_ylim(axis_properties.ylim)

    save_and_close(plot_name, fig, dir_name)
