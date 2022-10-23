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

import numpy as np
import util
import plotting

def get_range(val_lo, val_hi, val_num=10, log_space=False):
    if log_space:
        log_lo, log_hi = np.log([val_lo, val_hi])
        val_range = np.exp(np.linspace(log_lo, log_hi, val_num))
    else:
        val_range = np.linspace(val_lo, val_hi, val_num)

    return val_range

class Parameter:
    def __init__(
        self,
        name,
        default,
        val_range,
        log_x_axis=False,
        plot_axis_properties=None,
    ):
        self.name = name
        self.default = default
        self.val_range = val_range
        self.val_results_dict = None

        if plot_axis_properties is None:
            plot_axis_properties = plotting.AxisProperties(
                xlabel=name,
                ylabel="Result",
                log_xscale=log_x_axis,
            )

        self.plot_axis_properties = plot_axis_properties

    def __repr__(self):
        return (
            "Parameter(name=%r, default=%r, range=%r)"
            % (self.name, self.default, self.val_range)
        )

class Experiment:
    def run(self, **kwargs):
        raise NotImplementedError()

class ParamSweeper:
    def __init__(
        self,
        experiment,
        n_repeats=5,
        n_sigma=1,
        higher_is_better=True,
        print_every=1,
        verbose=True,
        printer=None,
    ):
        self._experiment = experiment
        self._n_repeats = n_repeats
        self._n_sigma = n_sigma
        self._higher_is_better = higher_is_better
        self._print_every = print_every
        self._verbose = verbose
        if printer is None:
            printer = util.Printer()
        self._print = printer

        self._param_list = list()
        self._params_to_results_dict = dict()
        self._context = util.ExceptionContext(
            suppress_exceptions=True,
            printer=printer,
        )

    def add_parameter(self, parameter):
        self._param_list.append(parameter)

    def find_best_parameters(self):
        while True:
            self._has_updated_any_parameters = False
            for parameter in self._param_list:
                self._print("\nSweeping over parameter %r..." % parameter.name)
                self.sweep_parameter(parameter, update_parameters=True)
            if not self._has_updated_any_parameters:
                self._print("\nFinished sweeping through parameters")
                break

        self._print("Best parameters found:")
        for param in self._param_list:
            self._print("> %20r = %s" % (param.name, param.default))

        return {param.name: param.default for param in self._param_list}

    def sweep_parameter(self, parameter, update_parameters=True):
        param_dict = {param.name: param.default for param in self._param_list}
        val_results_dict = dict()

        for val in parameter.val_range:
            param_dict[parameter.name] = val
            param_tuple = tuple(sorted(param_dict.items()))

            if param_tuple not in self._params_to_results_dict:
                results_list = self._run_experiment(param_dict)
                self._params_to_results_dict[param_tuple] = results_list
            else:
                results_list = self._params_to_results_dict[param_tuple]

            val_results_dict[val] = results_list

        if update_parameters:
            best_param_val, score = self._get_best_param_val(val_results_dict)
            if parameter.default != best_param_val:
                self._print(
                    "\nParameter %r default value changing from %r to %r"
                    % (parameter.name, parameter.default, best_param_val)
                )
                self._print(
                    "New optimal objective function value = %r" % score
                )
                parameter.default = best_param_val
                self._has_updated_any_parameters = True

        parameter.val_results_dict = val_results_dict

        return val_results_dict

    def tighten_ranges(self, new_num_vals=15):
        for param in self._param_list:
            if any(not util.is_numeric(v) for v in param.val_range):
                continue
            lo_candidates = [v for v in param.val_range if v < param.default]
            hi_candidates = [v for v in param.val_range if v > param.default]
            if len(lo_candidates) > 0:
                val_lo = max(lo_candidates)
            else:
                val_lo = param.default / 2
            if len(hi_candidates) > 0:
                val_hi = min(hi_candidates)
            else:
                val_hi = param.default * 2
            new_range = get_range(val_lo, val_hi, new_num_vals)
            param.val_range = np.sort(
                np.concatenate([new_range, [param.default]])
            )

    def plot(
        self,
        experiment_name="Experiment",
        output_dir=None,
        **plot_kwargs,
    ):
        filename_list = []
        for param in self._param_list:
            if param.val_results_dict is None:
                continue

            val_results_dict = param.val_results_dict
            all_results_pairs = [
                [val, result]
                for val, result_list in val_results_dict.items()
                for result in result_list
            ]
            all_results_x, all_results_y = zip(*all_results_pairs)

            val_list = [
                val for val in param.val_range
                if len(val_results_dict[val]) > 0
            ]
            results_list_list = [val_results_dict[val] for val in val_list]
            mean = np.array([np.mean(x) for x in results_list_list])
            std  = np.array([np.std( x) for x in results_list_list])

            mean_default = np.mean(val_results_dict[param.default])
            std_default  = np.std( val_results_dict[param.default])

            if self._higher_is_better:
                optimal_h = mean_default - (self._n_sigma * std_default)
            else:
                optimal_h = mean_default + (self._n_sigma * std_default)

            if util.is_numeric(param.default):
                param_default_str = "%.3g" % param.default
            else:
                param_default_str = str(param.default)

            plot_filename = plotting.plot(
                plotting.Line(
                    all_results_x,
                    all_results_y,
                    c="b",
                    ls="",
                    marker="o",
                    alpha=0.3,
                    label="Result",
                    zorder=20,
                ),
                plotting.Line(
                    val_list,
                    mean,
                    c="b",
                    label="Mean results",
                    zorder=30,
                ),
                plotting.FillBetween(
                    val_list,
                    mean + (self._n_sigma * std),
                    mean - (self._n_sigma * std),
                    color="b",
                    label="$\\pm %s \\sigma$" % self._n_sigma,
                    alpha=0.3,
                    zorder=10,
                ),
                plotting.HVLine(
                    v=param.default,
                    h=optimal_h,
                    c="r",
                    ls="--",
                    label="Optimal value = %s" % param_default_str,
                    zorder=40,
                ),
                plot_name=(
                    "Parameter sweep results for %r, varying parameter %r"
                    % (experiment_name, param.name)
                ),
                dir_name=output_dir,
                legend_properties=plotting.LegendProperties(),
                axis_properties=param.plot_axis_properties,
                **plot_kwargs,
            )
            filename_list.append(plot_filename)

        return filename_list

    def _run_experiment(self, experiment_param_dict):
        if self._verbose:
            self._print("Running an experiment with parameters:")
            for name, value in experiment_param_dict.items():
                self._print("| %20r = %r" % (name, value))

        results_list = []
        for i in range(self._n_repeats):
            with self._context:
                score = self._experiment.run(**experiment_param_dict)
                results_list.append(score)
                if self._verbose and ((i % self._print_every) == 0):
                    self._print(
                        "Repeat %i/%i, result is %s"
                        % (i, self._n_repeats, score)
                    )

        return results_list

    def _get_best_param_val(self, val_results_dict):
        non_empty_results_dict = {
            val: results_list
            for val, results_list in val_results_dict.items()
            if len(results_list) > 0
        }
        if self._higher_is_better:
            score_dict = {
                val: (np.mean(results) - self._n_sigma * np.std(results))
                for val, results in non_empty_results_dict.items()
            }
            best_param_val = max(
                non_empty_results_dict.keys(),
                key=lambda val: score_dict[val],
            )
        else:
            score_dict = {
                val: (np.mean(results) + self._n_sigma * np.std(results))
                for val, results in non_empty_results_dict.items()
            }
            best_param_val = min(
                non_empty_results_dict.keys(),
                key=lambda val: score_dict[val],
            )

        return best_param_val, score_dict[best_param_val]
