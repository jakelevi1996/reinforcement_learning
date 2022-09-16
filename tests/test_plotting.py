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
    plotting.plot(
        line_list=[
            plotting.Bar("Red", 3.1, color="r", zorder=10, label="Bar 1"),
            plotting.Bar("Green", 4.3, color="g", zorder=10, label="Bar 2"),
        ],
        plot_name="test_plot_bar",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="Category",
            ylabel="Height",
        ),
        legend_properties=plotting.LegendProperties(),
    )

def test_log_axes():
    seed = util.Seeder().get_seed("test_log_axes")
    rng = np.random.default_rng(seed)
    n = 100
    x = np.linspace(-2, 5, n)
    noise = 0.1 * rng.normal(size=n)
    plotting.plot(
        line_list=[plotting.Line(x, np.exp(x + noise), c="b")],
        plot_name="test_log_axes - log y axis",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties("x", "y", log_yscale=True),
    )
    plotting.plot(
        line_list=[plotting.Line(np.exp(x), x + noise, c="b")],
        plot_name="test_log_axes - log x axis",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties("x", "y", log_xscale=True),
    )
    x2 = np.exp(np.linspace(2, 8, n))
    plotting.plot(
        line_list=[plotting.Line(x2, np.power(x2 + 50*noise, -1.5), c="b")],
        plot_name="test_log_axes - log both axes",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="x",
            ylabel="y",
            log_xscale=True,
            log_yscale=True,
        ),
    )

def test_colour_picker():
    pass

def test_rotate_xtick_labels():
    pass

def test_long_title():
    pass
