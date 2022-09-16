import plotting
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

def test_plot_bar():
    plotting.plot(
        line_list=[
            plotting.Bar("Red", 3.1, color="r", zorder=10),
            plotting.Bar("Green", 4.3, color="g", zorder=10),
        ],
        plot_name="test_plot_bar",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="Category",
            ylabel="Height",
        ),
    )

def test_legend():
    pass

def test_log_axes():
    pass

def test_colour_picker():
    pass

def test_rotate_xtick_labels():
    pass

def test_long_title():
    pass
