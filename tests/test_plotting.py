import os
import numpy as np
import pytest
import plotting
import util
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_plotting")

def test_plot_lines():
    """
    Test creating a few lines, including instances of both the Line and HVLine
    classes, with a variety of colours, markers, line styles, and
    transparencies, plotting them on a single graph, and saving that graph to
    disk, with specified axis labels
    """
    line_list = [
        plotting.Line([1, 2, 3], [4, 5, 7], c="b"),
        plotting.Line([1.6, 1.3, 1.8], [3.1, 5.6, 4], marker="o", c="r"),
        plotting.Line([1.4, 2.5], [3.5, 3.9], ls="--", c="g"),
        plotting.HVLine(h=5.3, v=2.2, c="m", zorder=-10, lw=10, alpha=0.4),
    ]
    output_filename = plotting.plot(
        *line_list,
        plot_name="test_plot_lines",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(xlabel="x", ylabel="y"),
    )
    assert os.path.isfile(output_filename)

def test_plot_fill():
    """
    Test creating and plotting filled shapes with the plotting.FillBetween
    class
    """
    output_filename = plotting.plot(
        plotting.FillBetween(
            x=[1, 2, 2.5],
            y1=[1.5, 2, 3],
            y2=[4, 3, 4.5],
            color="b",
            alpha=0.3,
        ),
        plotting.FillBetween(
            x=[1.3, 2.1, 3],
            y1=[4, 2, 3],
            y2=[5.5, 4, 4.5],
            color="r",
            alpha=0.3,
        ),
        plot_name="test_plot_fill",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(xlabel="x", ylabel="y"),
    )
    assert os.path.isfile(output_filename)

def test_legend():
    """
    Test creating a legend, and adding various different types and styles of
    lines and filled shapes to that legend. Also test plotting a line which is
    not initialised with the `label` keyword argument, which should not be
    added to the legend, whereas all lines initialised with the `label` keyword
    argument should be added to the legend
    """
    line_list = [
        plotting.Line([1, 2], [1, 2], marker="o", c="r", label="Red line"),
        plotting.Line([1.2, 1.8], [1.8, 1.2], c="g", label="Green line"),
        plotting.Line([1.3, 1.7], [1.5, 1.6], marker="o", c="y"),
        plotting.HVLine(h=1.7, c="m", ls="--", label="hline"),
        plotting.FillBetween(
            x=[1.3, 1.6],
            y1=[1.2, 1.3],
            y2=[1.1, 1.0],
            fc="b",
            alpha=0.5,
            label="Patch",
        ),
    ]
    axis_properties = plotting.AxisProperties(xlabel="x", ylabel="y")
    output_filename = plotting.plot(
        *line_list,
        plot_name="test_legend",
        dir_name=OUTPUT_DIR,
        axis_properties=axis_properties,
        legend_properties=plotting.LegendProperties(),
    )
    assert os.path.isfile(output_filename)

def test_plot_bar():
    """
    Test creating a bar chart using the plotting.Bar class, and also test
    passing `rotate_xticklabels=True` to `plotting.AxisProperties` (this is
    useful for bar charts with long strings as independent variables which
    would otherwise overlap)
    """
    x1 = "Red" * 10
    x2 = "Green" * 5
    output_filename = plotting.plot(
        plotting.Bar(x1, 3.1, color="r", zorder=10, label="Bar 1"),
        plotting.Bar(x2, 4.3, color="g", zorder=10, label="Bar 2"),
        plot_name="test_plot_bar",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="Category",
            ylabel="Height",
            rotate_xticklabels=True,
        ),
        legend_properties=plotting.LegendProperties(),
    )
    assert os.path.isfile(output_filename)

def test_log_axes():
    """
    Test making plots with:

    - Logarithmic x axis and linear y axis
    - Linear x axis and logarithmic y axis
    - Both logarithmic x axis and logarithmic y axis
    """
    x1 = [1, 2, 3, 4, 5, 6]
    y1 = 1e-3 * np.array([1.2, 6, 120, 600, 1e4, 9e4])
    output_filename = plotting.plot(
        plotting.Line(x1, y1, c="b", marker="o"),
        plot_name="test_log_axes - log y axis",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties("x", "y", log_yscale=True),
    )
    assert os.path.isfile(output_filename)

    x2 = [0.1, 1, 10, 100, 1000]
    y2 = [3.8, 3.2, 1.8, 1.2, -1.2]
    output_filename = plotting.plot(
        plotting.Line(x2, y2, c="b", marker="o"),
        plot_name="test_log_axes - log x axis",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties("x", "y", log_xscale=True),
    )
    assert os.path.isfile(output_filename)

    x3 = [1, 10, 100, 1000]
    noise = np.array([0.4, 1.8, 0.3, 2.2])
    y3 = 1e-4 * np.power(x3, 2.3) * noise
    output_filename = plotting.plot(
        plotting.Line(x3, y3, c="b", marker="o"),
        plot_name="test_log_axes - log both axes",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="x",
            ylabel="y",
            log_xscale=True,
            log_yscale=True,
        ),
    )
    assert os.path.isfile(output_filename)

@pytest.mark.parametrize("num_colours, cyclic", [[5, True], [7, False]])
def test_colour_picker(num_colours, cyclic):
    """
    Test the plotting.ColourPicker class for generating unique colours for
    different plotting elements, with both cyclic and non-cyclic colour maps
    """
    cp = plotting.ColourPicker(num_colours, cyclic)
    x = np.linspace(-1, 7, 100)
    line_list = [
        plotting.Line(
            x=x,
            y=((1 + (i/10)) * np.sin(x + (i / num_colours))),
            c=cp(i),
            label="Line %i" % i,
        )
        for i in range(num_colours)
    ]
    output_filename = plotting.plot(
        *line_list,
        plot_name="test_colour_picker, cyclic=%s" % cyclic,
        dir_name=OUTPUT_DIR,
        legend_properties=plotting.LegendProperties(),
    )
    assert os.path.isfile(output_filename)

def test_title():
    """
    Check that long titles are wrapped onto multiple lines, plot names
    containing invalid characters can still be used to generate valid filenames
    by replacing invalid characters from the plot names, and latex formatting
    within the title (or axis or legend labels) is formatted correctly
    """
    title = (
        "This is a very long title containing /|\\*:<\"$pecial?\">:*/|\\ "
        "characters which wraps multiple lines because it is too long for "
        "one line. It also contains $\\sum_{{i}}{{\\left[\\frac{{latex}}{{"
        "\\alpha_i^\\beta}}\\right]}}$"
    )
    output_filename = plotting.plot(
        plotting.Line(
            x=[1, 2, 3],
            y=[4, 4.5, 6],
            c="b",
            marker="o",
            label="$\\beta ^ \\varepsilon$",
        ),
        plot_name=title,
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="$x_1$",
            ylabel="$x_2$",
        ),
        legend_properties=plotting.LegendProperties()
    )
    assert os.path.isfile(output_filename)
