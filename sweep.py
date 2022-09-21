import numpy as np
import util
import plotting

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
        """
        Initialise a Parameter object, which represents a parameter that will
        be added to instance of the ParamSweeper class. Multiple values for
        this parameter can be swept over to find the approximately optimal
        value for this parameter in a given experiment, and the results can be
        plotted on a graph. This object can be initialised with the following
        arguments:

        - name: string representing the name of this parameter (EG
          "step_size"). This name should match one of the argument names of the
          run method of an Experiment subclass for the given experiment (see
          below), and an instance of this Experiment subclass should be passed
          to the ParamSweeper class during initialisation
        - default: the initial value that this parameter will take in
          experiments while other parameters are being swept over (EG 0.7)
        - val_range (optional): iterable containing all the values that this
          parameter will take in experiments while this parameter is being
          swept over. If this argument is not provided, the range of values can
          be specified using the keyword arguments val_lo, val_hi, val_num, and
          log_space (see below)
        - val_lo, val_hi, val_num, log_space (optional): if val_range is not
          provided, val_num number of values are generated between val_lo and
          val_hi, spaced linearly if log_space is False, or logarithmically if
          log_space is True, and this parameter will take these values in
          experiments when it is being swept over (by default, val_num=10 and
          log_space=False)
        - plot_axis_properties (optional): specify axis properties for the plot
          that is made of the results of the given experiment for the different
          values that this parameter takes
        """
        self.name = name
        self.default = default
        self.val_results_dict = None

        if val_range is None:
            if (val_lo is None) or (val_hi is None):
                raise ValueError(
                    "Must either specify val_range or specify val_lo and "
                    "val_hi"
                )
            if log_space:
                log_lo, log_hi = np.log([val_lo, val_hi])
                val_range = np.exp(np.linspace(log_lo, log_hi, val_num))
            else:
                val_range = np.linspace(val_lo, val_hi, val_num)

        self.val_range = val_range

        if plot_axis_properties is None:
            plot_axis_properties = plotting.AxisProperties(
                xlabel=name,
                ylabel="Result",
                log_xscale=log_space,
            )

        self.plot_axis_properties = plot_axis_properties

    def __repr__(self):
        return (
            "Parameter(name=%r, default=%r, range=%r)"
            % (self.name, self.default, self.val_range)
        )

class Experiment:
    """
    This class represents an abstract experiment to be run by an instance of
    the ParamSweeper class. A particular experiment should inherit from this
    class and override the `run` method below, and an instance of this subclass
    should be passed as an argument when initialising a ParamSweeper instance.
    The run method will be run for a specified number of repeats for various
    different combinations of parameter values. The argument names of the run
    method should match the names of the parameters that are added to the
    ParamSweeper instance.
    """
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
        printer=None,
    ):
        """
        Initialise an instance of the ParamSweeper class, which can be used to
        sweep over multiple different values of multiple different parameters,
        performing multiple repeats of a given experiment with a variety of
        different combinations of parameter values, to find the approximately
        optimal value for each parameter in the given experiment. This class
        can also be used to plot the results of these experiments on a graph.
        This object can be initialised with the following arguments:

        - experiment: an instance of a subclass of the Experiment class (see
          above) which has overriden the `run` method. The `run` method of this
          experiment object will be called with multiple different combinations
          of parameter values
        - n_repeats:  (optional) the number of repeats to perform of each
          experiment with a given set of parameter values (default is 5)
        - n_sigma:  (optional) the number of standard deviations above/below
          the mean (depending on if higher_is_better is True or False) which is
          used to decide which value of a parameter is optimal (which prevents
          a parameter value being considered optimal if it has an excessively
          high variance, implying that it is not reliably optimal), and also
          how many standard deviations above and below the mean are displayed
          in any output plots (default is 1)
        - higher_is_better:  (optional) when considering the optimal value for
          each parameter, if higher_is_better is True then a higher result is
          considered better, otherwise a lower result is considered better
          (default is True)
        - print_every:  (optional) integer number of repeats of the given
          experiment that should take place between printing a progress message
          to the console and/or a text file (default is 1)
        - printer: (optional) instance of util.Printer, which can be used to
          specify whether progress messages should be printed to the console
          and/or a given text file, or not at all (default is to print progress
          messages to the console)
        """
        self._experiment = experiment
        self._n_repeats = n_repeats
        self._n_sigma = n_sigma
        self._higher_is_better = higher_is_better
        self._print_every = print_every
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
        """
        Add a parameter to this ParamSweeper instance. parameter should be an
        instance of Parameter
        """
        self._param_list.append(parameter)

    def find_best_parameters(self):
        """
        Find the approximately optimal value for each parameter by iteratively
        sweeping over the given range of values for each parameter, changing
        the default value for each parameter every time a more optimal value is
        found, until all parameter values stop changing. Note that each
        parameter can have its range of values swept over multiple times before
        this stopping condition is reached.
        """
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
        """
        Sweep over the range of values for the given parameter, performing the
        given number of repeats of the given experiment for each different
        value, while all other parameters are kept at their default values. The
        list of results for each combination of parameter values is saved for
        later in a dictionary, and if the list of results for a given
        combination of parameter values has already been stored in the
        dictionary, then that list of results is retrieved from the dictionary
        instead of performing the experiment again with that same combination
        of parameter values.

        If update_parameters is True, then after the given experiment has been
        performed for the given number of repeats for each different value of
        the given parameter, the default value for the given parameter is set
        to that which maximises a linear combination of the mean and standard
        deviation of the results list. The use of both the mean and the
        standard deviation of the results list prevents a parameter value being
        considered optimal if it has an excessively high variance, implying
        that it is not reliably optimal. The relative weighting of mean and
        standard deviation used to decide the optimal value of this parameter
        depends on the value of n_sigma with which this ParamSweeper instance
        was initialised.
        """
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

    def plot(self, experiment_name="Experiment", output_dir=None):
        """
        For each parameter that has been added to this ParamSweeper instance,
        plot the results of the given experiment for each value of that
        parameter (while the values of all other parameters are set to their
        default values), and save the plot in an image file in the directory
        specified by the `output_dir` argument (if `output_dir` is not
        provided, the image is saved in `util.RESULTS_DIR`). `experiment_name`
        should be a string describing the name of the experiment, which is used
        in the plot titles and filenames of the output images. This method
        returns a list of the filenames of all the plots that it creates.
        """
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
                label="Optimal value = %s" % param_default_str,
                zorder=40,
            )
            plot_filename = plotting.plot(
                [all_results_line, mean_line, std_line, default_line],
                plot_name=(
                    "Parameter sweep results for %r, varying parameter %r"
                    % (experiment_name, param.name)
                ),
                dir_name=output_dir,
                legend_properties=plotting.LegendProperties(),
                axis_properties=param.plot_axis_properties,
            )
            filename_list.append(plot_filename)

        return filename_list

    def _run_experiment(self, experiment_param_dict):
        """
        Given a dictionary mapping parameter names to values for those
        parameters, run the given experiment with the given parameters for the
        given number of repeats, and return a list containing the results for
        each successful repeat of the experiment. If an exception is raised
        while running the experiment, the exception is suppressed, details of
        the exception are printed to the console and/or a text file or neither
        (depending on the value of `printer` with which this instance of
        ParamSweeper was initialised), and no result is added to the output
        list for that repeat of the experiment.
        """
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
        """
        Given a dictionary mapping each value of a particular parameter to a
        list of results (one result for each repeat of the given experiment,
        with all other parameters set to their current default values), return
        the optimal value of the parameter, which is considered to be that
        which maximises a particular linear combination of the mean and
        standard deviation of the corresponding results list. The use of both
        the mean and the standard deviation of the results list prevents a
        parameter value being considered optimal if it has an excessively high
        variance, implying that it is not reliably optimal.
        """
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
