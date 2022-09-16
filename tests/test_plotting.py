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
    pass

def test_plot_bar():
    pass

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
