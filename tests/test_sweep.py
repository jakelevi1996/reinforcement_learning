import os
import pytest
import numpy as np
import sweep
import util
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_sweep")

@pytest.mark.parametrize("higher_is_better", [True, False])
def test_sweep(higher_is_better):
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
        assert len(results_list) > 0

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

    optimal_params = [optimal_param_dict[key] for key in ["x", "y", "z"]]
    assert optimal_params == target_list[-1]

    for target in target_list:
        target_tuple = (("x", target[0]), ("y", target[1]), ("z", target[2]))
        assert target_tuple in sweeper._params_to_results_dict

    for i in range(11):
        point_tuple = (("x", i), ("y", i), ("z", i))
        if i in [0, 5, 10]:
            assert point_tuple in sweeper._params_to_results_dict
        else:
            assert point_tuple not in sweeper._params_to_results_dict

def sq_distance(v1, v2):
    return np.sum(np.square(np.array(v1) - np.array(v2)))
