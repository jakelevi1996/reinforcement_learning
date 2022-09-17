import numpy as np
import pytest
import plotting
import util
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_plotting")

def test_plot_lines():
    line1 = plotting.Line([1, 2, 3], [4, 5, 7], c="b")
    line2 = plotting.Line([1.6, 1.3, 1.8], [3.1, 5.6, 4], marker="o", c="r")
    line3 = plotting.Line([1.4, 2.5], [3.5, 3.9], ls="--", c="g")
    line4 = plotting.HVLine(h=5.3, v=2.2, c="m", zorder=-10, lw=10, alpha=0.4)
    plotting.plot(
        line_list=[line1, line2, line3, line4],
        plot_name="test_plot_lines",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(xlabel="x", ylabel="y"),
    )

def test_plot_fill():
    fill1 = plotting.FillBetween(
        x=[1, 2, 2.5],
        y1=[1.5, 2, 3],
        y2=[4, 3, 4.5],
        color="b",
        alpha=0.3,
    )
    fill2 = plotting.FillBetween(
        x=[1.3, 2.1, 3],
        y1=[4, 2, 3],
        y2=[5.5, 4, 4.5],
        color="r",
        alpha=0.3,
    )
    plotting.plot(
        line_list=[fill1, fill2],
        plot_name="test_plot_fill",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(xlabel="x", ylabel="y"),
    )

def test_legend():
    line1 = plotting.Line([1, 2], [1, 2], marker="o", c="r", label="Red line")
    line2 = plotting.Line([1.2, 1.8], [1.8, 1.2], c="g", label="Green line")
    line3 = plotting.Line([1.3, 1.7], [1.5, 1.6], marker="o", c="y")
    line4 = plotting.HVLine(h=1.7, c="m", ls="--", label="hline")
    fill1 = plotting.FillBetween(
        x=[1.3, 1.6],
        y1=[1.2, 1.3],
        y2=[1.1, 1.0],
        fc="b",
        alpha=0.5,
        label="Patch",
    )
    axis_properties = plotting.AxisProperties(xlabel="x", ylabel="y")
    plotting.plot(
        line_list=[line1, line2, line3, line4, fill1],
        plot_name="test_legend",
        dir_name=OUTPUT_DIR,
        axis_properties=axis_properties,
        legend_properties=plotting.LegendProperties(),
    )

def test_plot_bar():
    x1 = "Red" * 10
    x2 = "Green" * 5
    plotting.plot(
        line_list=[
            plotting.Bar(x1, 3.1, color="r", zorder=10, label="Bar 1"),
            plotting.Bar(x2, 4.3, color="g", zorder=10, label="Bar 2"),
        ],
        plot_name="test_plot_bar",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="Category",
            ylabel="Height",
            rotate_xticklabels=True,
        ),
        legend_properties=plotting.LegendProperties(),
    )

def test_log_axes():
    x1 = [1, 2, 3, 4, 5, 6]
    y1 = 1e-3 * np.array([1.2, 6, 120, 600, 1e4, 9e4])
    plotting.plot(
        line_list=[plotting.Line(x1, y1, c="b", marker="o")],
        plot_name="test_log_axes - log y axis",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties("x", "y", log_yscale=True),
    )

    x2 = [0.1, 1, 10, 100, 1000]
    y2 = [3.8, 3.2, 1.8, 1.2, -1.2]
    plotting.plot(
        line_list=[plotting.Line(x2, y2, c="b", marker="o")],
        plot_name="test_log_axes - log x axis",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties("x", "y", log_xscale=True),
    )

    x3 = [1, 10, 100, 1000]
    noise = np.array([0.4, 1.8, 0.3, 2.2])
    y3 = 1e-4 * np.power(x3, 2.3) * noise
    plotting.plot(
        line_list=[plotting.Line(x3, y3, c="b", marker="o")],
        plot_name="test_log_axes - log both axes",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="x",
            ylabel="y",
            log_xscale=True,
            log_yscale=True,
        ),
    )

@pytest.mark.parametrize("num_colours, cyclic", [[5, True], [7, False]])
def test_colour_picker(num_colours, cyclic):
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
    plotting.plot(
        line_list=line_list,
        plot_name="test_colour_picker, cyclic=%s" % cyclic,
        dir_name=OUTPUT_DIR,
        legend_properties=plotting.LegendProperties(),
    )

def test_title():
    """ Check that long titles are wrapped, invalid characters are removed from
    the title, and latex formatting within the title (or axis or legend labels)
    is formatted correctly """
    title = (
        "This is a very long title containing /|\\*:<\"$pecial?\">:*/|\\ "
        "characters which wraps multiple lines because it is too long for "
        "one line. It also contains $\\sum_{{i}}{{\\left[\\frac{{latex}}{{"
        "\\alpha_i^\\beta}}\\right]}}$"
    )
    line = plotting.Line(
        x=[1, 2, 3],
        y=[4, 4.5, 6],
        c="b",
        marker="o",
        label="$\\beta ^ \\varepsilon$",
    )
    plotting.plot(
        [line],
        plot_name=title,
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="$x_1$",
            ylabel="$x_2$",
        ),
        legend_properties=plotting.LegendProperties()
    )
