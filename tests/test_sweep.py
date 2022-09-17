import numpy as np
import sweep
import util
import tests.util

def test_sweep():
    pass

def test_sweep_errors():
    pass

def test_sweep_categorical_parameter():
    pass

def test_multiple_sweeps():
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
    output_dir = tests.util.get_output_dir(
        "test_sweep",
        "test_multiple_sweeps",
    )
    printer = util.Printer("Console output.txt", output_dir)
    experiment = MultiSweep(target_list, printer)
    sweeper = sweep.ParamSweeper(experiment, 1, printer=printer)
    sweeper.add_parameter(sweep.Parameter("x", 0, list(range(11))))
    sweeper.add_parameter(sweep.Parameter("y", 0, list(range(11))))
    sweeper.add_parameter(sweep.Parameter("z", 0, list(range(11))))
    optimal_param_dict = sweeper.find_best_parameters()
    printer(
        "%i experiments performed in total"
        % len(sweeper._params_to_results_dict)
    )
    sweeper.plot("test_multiple_sweeps", output_dir)

    optimal_params = [optimal_param_dict[key] for key in ["x", "y", "z"]]
    assert optimal_params == target_list[-1]

    for target in target_list:
        target_dict = {"x": target[0], "y": target[1], "z": target[2]}
        target_tuple = sweep.dict_to_tuple(target_dict)
        assert target_tuple in sweeper._params_to_results_dict

    for i in range(11):
        point_tuple = sweep.dict_to_tuple({"x": i, "y": i, "z": i})
        if i in [0, 5, 10]:
            assert point_tuple in sweeper._params_to_results_dict
        else:
            assert point_tuple not in sweeper._params_to_results_dict

def sq_distance(v1, v2):
    return np.sum(np.square(np.array(v1) - np.array(v2)))
