import os
import pytest
import numpy as np
import sweep
import util
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_sweep")

@pytest.mark.parametrize("higher_is_better", [True, False])
def test_sweep(higher_is_better):
    """
    Test the sweep.ParamSweeper class, including the add_parameter,
    find_best_parameters, and plot methods, initialised with higher_is_better
    as both True and False (each in a different test run, facilitated by
    pytest.mark.parametrize), and check that that the optimal parameters are
    found by the find_best_parameters method in both cases
    """
    if higher_is_better:
        output_dir = os.path.join(OUTPUT_DIR, "higher_is_better")
    else:
        output_dir = os.path.join(OUTPUT_DIR, "lower_is_better")

    printer = util.Printer("Console_output.txt", output_dir)
    target = [2, 5, 7]
    rng = util.Seeder().get_rng("test_sweep", higher_is_better)

    class SimpleExperiment(sweep.Experiment):
        def run(self, x, y, z):
            noise = rng.normal()
            if higher_is_better:
                return - sq_distance([x, y, z], target) + noise
            else:
                return sq_distance([x, y, z], target) + noise

    sweeper = sweep.ParamSweeper(
        experiment=SimpleExperiment(),
        n_repeats=100,
        n_sigma=2.5,
        higher_is_better=higher_is_better,
        print_every=50,
        printer=printer,
    )
    sweeper.add_parameter(sweep.Parameter("x", 0, list(range(11))))
    sweeper.add_parameter(sweep.Parameter("y", 0, list(range(11))))
    sweeper.add_parameter(sweep.Parameter("z", 0, list(range(11))))
    optimal_param_dict = sweeper.find_best_parameters()
    sweeper.plot("test_sweep", output_dir)

    printer(
        "%i experiments performed in total"
        % len(sweeper._params_to_results_dict)
    )

    optimal_params = [optimal_param_dict[key] for key in ["x", "y", "z"]]
    assert optimal_params == target

def test_sweep_errors():
    """
    Test sweeping over the parameters of an experiment in which some
    combinations of parameters cause an exception to be raised, that the
    exceptions are suppressed, that the results of the parameter sweeps can
    still be plotted (even though some combinations of parameters that were
    tested have no results to be plotted), and that the results of valid and
    invalid experiments pass sanity checks
    """
    output_dir = os.path.join(OUTPUT_DIR, "test_sweep_errors")
    printer = util.Printer("Console_output.txt", output_dir)
    rng = util.Seeder().get_rng("test_sweep_errors")
    target = [2, 5, 7]
    num_repeats = 20

    def is_valid(x, y, z):
        return (((x + y + z) % 2) != 0)

    class ErrorExperiment(sweep.Experiment):
        def run(self, x, y, z):
            if not is_valid(x, y, z):
                raise ValueError("Arguments are invalid")

            noise = rng.normal()
            return - sq_distance([x, y, z], target) + noise

    sweeper = sweep.ParamSweeper(
        experiment=ErrorExperiment(),
        n_repeats=num_repeats,
        n_sigma=2.5,
        higher_is_better=True,
        print_every=10,
        printer=printer,
    )
    sweeper.add_parameter(sweep.Parameter("x", 0, list(range(11))))
    sweeper.add_parameter(sweep.Parameter("y", 0, list(range(11))))
    sweeper.add_parameter(sweep.Parameter("z", 0, list(range(11))))
    sweeper.find_best_parameters()
    sweeper.plot("test_sweep_errors", output_dir)

    # Perform sanity checks on the results of valid and invalid experiments
    num_experiments = len(sweeper._params_to_results_dict)
    valid_experiments = {
        param_tuple: results_list
        for param_tuple, results_list
        in sweeper._params_to_results_dict.items()
        if len(results_list) > 0
    }
    invalid_experiments = {
        param_tuple: results_list
        for param_tuple, results_list
        in sweeper._params_to_results_dict.items()
        if len(results_list) == 0
    }
    num_valid   = len(valid_experiments)
    num_invalid = len(invalid_experiments)
    assert (num_valid   > 0) and (num_valid   < num_experiments)
    assert (num_invalid > 0) and (num_invalid < num_experiments)
    assert (num_valid + num_invalid) == num_experiments
    for param_tuple, results_list in valid_experiments.items():
        x, y, z = [pair[1] for pair in param_tuple]
        assert is_valid(x, y, z)
        assert len(results_list) == num_repeats

    for param_tuple, results_list in invalid_experiments.items():
        x, y, z = [pair[1] for pair in param_tuple]
        assert not is_valid(x, y, z)
        assert len(results_list) == 0

    printer(
        "%i experiments performed in total, of which %i were valid, and %i "
        "were invalid"
        % (num_experiments, num_valid, num_invalid)
    )

def test_sweep_categorical_and_log_range_parameters():
    """
    Test sweeping over a parameter which takes categorical (non-numerical)
    values, that the optimal value of this parameter is found without error,
    and that the optimal value of the categorical parameter is considered to be
    that which most reliably produces high results (not simply the value with
    the highest mean results). Also test initialising a Parameter using the
    val_lo, val_hi, val_num, and log_space arguments
    """
    output_dir = os.path.join(
        OUTPUT_DIR,
        "test_sweep_categorical_and_log_range_parameters",
    )
    printer = util.Printer("Console_output.txt", output_dir)
    rng = util.Seeder().get_rng(output_dir)
    categories = ["apple", "orange", "pear"]

    class SemiCategorical(sweep.Experiment):
        def run(self, x, y, category):
            if category == "apple":
                return - sq_distance([x, y], [3, 4]) + rng.normal(0, 2)
            if category == "orange":
                return - sq_distance([x, y], [3, 4]) + rng.normal(13, 1)
            if category == "pear":
                return - sq_distance([x, y], [3, 4]) + rng.normal(14, 3)
            else:
                raise ValueError("Invalid category")

    sweeper = sweep.ParamSweeper(
        experiment=SemiCategorical(),
        n_repeats=100,
        n_sigma=2.5,
        higher_is_better=True,
        print_every=50,
        printer=printer,
    )
    sweeper.add_parameter(sweep.Parameter("x", 0, list(range(11))))
    sweeper.add_parameter(sweep.Parameter("y", 0.1, None, 0.1, 10, 20, True))
    sweeper.add_parameter(sweep.Parameter("category", "apple", categories))
    optimal_param_dict = sweeper.find_best_parameters()
    sweeper.plot("test_sweep_categorical_parameter", output_dir)

    printer(
        "%i experiments performed in total"
        % len(sweeper._params_to_results_dict)
    )

    assert optimal_param_dict["category"] == "orange"

def test_multiple_sweeps():
    """
    Test finding the optimal parameters for an experiment which is contrived to
    require each parameter to change default values multiple times, by
    repeatedly changing the experiment such that the parmeter values are
    attracted towards a target which repeatedly changes location. Test also
    that the ParamSweeper instance tests combinations of parameter values which
    are expected and not ones which are unexpected
    """
    output_dir = os.path.join(OUTPUT_DIR, "test_multiple_sweeps")
    printer = util.Printer("Console_output.txt", output_dir)

    class MultiSweep(sweep.Experiment):
        def __init__(self, target_list, printer):
            self._target_iter = iter(target_list)
            self._target = next(self._target_iter)
            self._baseline = 0
            self._printer = printer

        def run(self, x, y, z):
            if [x, y, z] == self._target:
                self._printer("\n*** Target %s reached" % self._target)
                self._target = next(self._target_iter, self._target)
                self._baseline += sq_distance([x, y, z], self._target)
                self._printer("*** New target is %s" % self._target)

            reward = self._baseline - sq_distance([x, y, z], self._target)
            return reward

    target_list = [
        [0 , 0 , 0 ],
        [0 , 0 , 10],
        [0 , 10, 10],
        [10, 10, 10],
        [10, 10, 0 ],
        [10, 5 , 0 ],
        [5 , 5 , 0 ],
        [5 , 5 , 5 ],
        [5 , 10, 5 ],
    ]
    experiment = MultiSweep(target_list, printer)

    sweeper = sweep.ParamSweeper(experiment, 1, printer=printer)
    sweeper.add_parameter(sweep.Parameter("x", 0, list(range(11))))
    sweeper.add_parameter(sweep.Parameter("y", 0, list(range(11))))
    sweeper.add_parameter(sweep.Parameter("z", 0, list(range(11))))
    optimal_param_dict = sweeper.find_best_parameters()
    sweeper.plot("test_multiple_sweeps", output_dir)

    printer(
        "%i experiments performed in total"
        % len(sweeper._params_to_results_dict)
    )

    # Check that the optimal parameters returned by find_best_parameters are
    # equal to the final target
    optimal_params = [optimal_param_dict[key] for key in ["x", "y", "z"]]
    assert optimal_params == target_list[-1]

    # Check that every point in the list of targets has been tested in an
    # experiment
    for target in target_list:
        target_tuple = (("x", target[0]), ("y", target[1]), ("z", target[2]))
        assert target_tuple in sweeper._params_to_results_dict
        printer(
            "target %s found in dictionary with results %s"
            % (target, sweeper._params_to_results_dict[target_tuple])
        )

    # Check that points on the main diagonal are tested if they're in the
    # target list, and not tested if they're not in the target list
    for i in range(11):
        point_tuple = (("x", i), ("y", i), ("z", i))
        if [i, i, i] in target_list:
            assert point_tuple in sweeper._params_to_results_dict
            printer(
                "Diagonal point %s found in dictionary with results %s"
                % ([i, i, i], sweeper._params_to_results_dict[point_tuple])
            )
        else:
            assert point_tuple not in sweeper._params_to_results_dict
            printer("Diagonal point %s not found in dictionary" % [i, i, i])

def sq_distance(v1, v2):
    return np.sum(np.square(np.array(v1) - np.array(v2)))
