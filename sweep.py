import numpy as np
import util
import plotting

def dict_to_tuple(d):
    return tuple(sorted(d.items()))

class Parameter:
    def __init__(self, name, default, val_range):
        self.name = name
        self.default = default
        self.val_range = val_range
        self.val_results_dict = None

    def __repr__(self):
        return (
            "Parameter(name=%r, default=%r, range=%r)"
            % (self.name, self.default, self.val_range)
        )

class ParamSweeper:
    def __init__(
        self,
        func,
        n_repeats=5,
        n_sigma=1,
        higher_is_better=True,
        output_file=None,
    ):
        self._func = func
        self._n_repeats = n_repeats
        self._n_sigma = n_sigma
        self._higher_is_better = higher_is_better
        self._file = output_file

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
                self._print("Sweeping over parameter %r..." % parameter.name)
                self.sweep_parameter(parameter, update_parameters=True)
            if not self._has_updated_any_parameters:
                self._print("Finished sweeping through parameters")
                break

        self._print("Best parameters found:")
        for param in self._param_list:
            self._print("> %10r = %s" % (param.name, param.default))

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

            if len(results_list) > 0:
                val_results_dict[val] = results_list

        if update_parameters:
            best_param_val = self._get_best_param_val(val_results_dict)
            if parameter.default != best_param_val:
                self._print(
                    "Parameter %r default value changing from %s to %s"
                    % (parameter.name, parameter.default, best_param_val),
                )
                parameter.default = best_param_val
                self._has_updated_any_parameters = True

        parameter.val_results_dict = val_results_dict

        return val_results_dict

    def plot(self, output_dir=None):
        for param in self._param_list:
            if param.val_results_dict is None:
                continue
            val_results_dict = param.val_results_dict
            val_list = param.val_range
            results_list_list = [val_results_dict[val] for val in val_list]
            mean = np.array([np.mean(x) for x in results_list_list])
            std  = np.array([np.std( x) for x in results_list_list])
            mean_default = np.mean(val_results_dict[param.default])
            std_default  = np.std( val_results_dict[param.default])
            if self._higher_is_better:
                optimal_h = mean_default - (self._n_sigma * std_default)
            else:
                optimal_h = mean_default + (self._n_sigma * std_default)
            plot_name = (
                "Parameter sweep results, varying parameter %s"
                % param.name
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
            plotting.plot(
                [mean_line, std_line, default_line],
                plot_name,
                output_dir,
                legend_properties=plotting.LegendProperties(),
                axis_properties=plotting.AxisProperties(
                    xlabel=param.name,
                    ylabel="Result",
                )
            )

    def _run_experiment(self, experiment_param_dict):
        self._print("Running an experiment with the following parameters:")
        for name, value in experiment_param_dict.items():
            self._print("| %10r = %r" % (name, value))

        results_list = []
        for i in range(self._n_repeats):
            with self._context:
                score = self._func(**experiment_param_dict)
                results_list.append(score)
                self._print(
                    "Repeat %i/%i, result is %s"
                    % (i + 1, self._n_repeats, score)
                )

        return results_list

    def _get_best_param_val(self, val_results_dict):
        if self._higher_is_better:
            score_dict = {
                val: (np.mean(results) - self._n_sigma * np.std(results))
                for val, results in val_results_dict.items()
            }
            best_param_val = max(
                val_results_dict.keys(),
                key=lambda val: score_dict[val],
            )
        else:
            score_dict = {
                val: (np.mean(results) + self._n_sigma * np.std(results))
                for val, results in val_results_dict.items()
            }
            best_param_val = min(
                val_results_dict.keys(),
                key=lambda val: score_dict[val],
            )

        return best_param_val

    def _print(self, s):
        print(s)
        if self._file is not None:
            print(s, file=self._file)
