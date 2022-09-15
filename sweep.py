import numpy as np
import util
import plotting

def dict_to_tuple(d):
    return tuple(sorted(d.items()))

class Parameter:
    def __init__(
        self,
        name,
        default,
        val_range=None,
        val_lo=None,
        val_hi=None,
        val_num=10,
        log_space=False,
        plot_axis_properties=None,
    ):
        self.name = name
        self.default = default
        self.val_results_dict = None

        if (val_range is None) and ((val_lo is None) or (val_hi is None)):
            raise ValueError(
                "Must either specify val_range or specify val_lo and val_hi"
            )
        if val_range is None:
            if log_space:
                val_range = np.exp(
                    np.linspace(np.log(val_lo), np.log(val_hi), val_num)
                )
            else:
                val_range = np.linspace(val_lo, val_hi, val_num)

        self.val_range = val_range

        if plot_axis_properties is None:
            plot_axis_properties = plotting.AxisProperties(
                xlabel=name,
                ylabel="Result",
            )
            plot_axis_properties.log_xscale = log_space

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
        output_file=None,
        print_every=1,
    ):
        self._experiment = experiment
        self._n_repeats = n_repeats
        self._n_sigma = n_sigma
        self._higher_is_better = higher_is_better
        self._file = output_file
        self._print_every = print_every

        self._param_list = list()
        self._params_to_results_dict = dict()
        self._context = util.ExceptionContext(
            suppress_exceptions=True,
            output_file=output_file,
        )

    def add_parameter(self, parameter):
        self._param_list.append(parameter)

    def find_best_parameters(self):
        while True:
            self._has_updated_any_parameters = False
            for parameter in self._param_list:
                self._print(
                    "\nSweeping over parameter %r..."
                    % parameter.name
                )
                self.sweep_parameter(parameter, update_parameters=True)
            if not self._has_updated_any_parameters:
                self._print("\nFinished sweeping through parameters")
                break

        self._print("Best parameters found:")
        for param in self._param_list:
            self._print("> %20r = %s" % (param.name, param.default))

    def sweep_parameter(self, parameter, update_parameters=True):
        param_dict = {param.name: param.default for param in self._param_list}
        val_results_dict = dict()

        for val in parameter.val_range:
            param_dict[parameter.name] = val
            param_tuple = dict_to_tuple(param_dict)

            if param_tuple not in self._params_to_results_dict:
                results_list = self._run_experiment(param_dict)
                self._params_to_results_dict[param_tuple] = results_list
            else:
                results_list = self._params_to_results_dict[param_tuple]

            val_results_dict[val] = results_list

        if update_parameters:
            best_param_val = self._get_best_param_val(val_results_dict)
            if parameter.default != best_param_val:
                self._print(
                    "\nParameter %r default value changing from %s to %s"
                    % (parameter.name, parameter.default, best_param_val),
                )
                parameter.default = best_param_val
                self._has_updated_any_parameters = True

        parameter.val_results_dict = val_results_dict

        return val_results_dict

    def plot(self, output_dir=None):
        filename_list = []
        for param in self._param_list:
            if param.val_results_dict is None:
                continue
            val_results_dict = param.val_results_dict
            val_list = [
                val for val in param.val_range
                if len(val_results_dict[val]) > 0
            ]
            results_list_list = [val_results_dict[val] for val in val_list]
            all_results_pairs = [
                [val, result]
                for val, result_list in val_results_dict.items()
                for result in result_list
            ]
            all_results_x, all_results_y = zip(*all_results_pairs)
            mean = np.array([np.mean(x) for x in results_list_list])
            std  = np.array([np.std( x) for x in results_list_list])
            mean_default = np.mean(val_results_dict[param.default])
            std_default  = np.std( val_results_dict[param.default])
            if self._higher_is_better:
                optimal_h = mean_default - (self._n_sigma * std_default)
            else:
                optimal_h = mean_default + (self._n_sigma * std_default)
            all_results_line = plotting.Line(
                all_results_x,
                all_results_y,
                c="b",
                ls="",
                marker="o",
                alpha=0.3,
                label="Result",
                zorder=20,
            )
            mean_line = plotting.Line(
                val_list,
                mean,
                c="b",
                label="Mean results",
                zorder=30,
            )
            std_line = plotting.FillBetween(
                val_list,
                mean + (self._n_sigma * std),
                mean - (self._n_sigma * std),
                color="b",
                label="$\\pm %s \\sigma$" % self._n_sigma,
                alpha=0.3,
                zorder=10,
            )
            default_line = plotting.HVLine(
                v=param.default,
                h=optimal_h,
                c="r",
                ls="--",
                label="Optimal value = %s" % param.default,
                zorder=40,
            )
            plot_filename = plotting.plot(
                [all_results_line, mean_line, std_line, default_line],
                "Parameter sweep results, varying parameter %s" % param.name,
                output_dir,
                legend_properties=plotting.LegendProperties(),
                axis_properties=param.plot_axis_properties,
            )
            filename_list.append(plot_filename)

        return filename_list

    def _run_experiment(self, experiment_param_dict):
        self._print("Running an experiment with the following parameters:")
        for name, value in experiment_param_dict.items():
            self._print("| %20r = %r" % (name, value))

        results_list = []
        for i in range(self._n_repeats):
            with self._context:
                score = self._experiment.run(**experiment_param_dict)
                results_list.append(score)
                if (i % self._print_every) == 0:
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

        return best_param_val

    def _print(self, s):
        print(s)
        if self._file is not None:
            print(s, file=self._file)
