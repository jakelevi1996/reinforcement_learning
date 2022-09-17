import os
import pytest
import numpy as np
import util
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_util")

def test_result():
    output_filename = os.path.join(OUTPUT_DIR, "result_data.pkl")
    data = [1, 2, 3]
    result = util.Result(output_filename, data)
    with result.get_results_saving_context(suppress_exceptions=True):
        data[2] *= 4
        raise ValueError()

    loaded_data = result.load()
    assert loaded_data == [1, 2, 12]

def test_exception_context():
    printer = util.Printer(
        output_filename="test_exception_context.txt",
        output_dir=OUTPUT_DIR,
    )
    printer("About to enter ExceptionContext...")
    with util.ExceptionContext(suppress_exceptions=True, printer=printer):
        printer("In context, about to raise ValueError...")
        raise ValueError("Error message")

    printer("ExceptionContext has exited, now back in test_exception_context")

    with pytest.raises(ValueError):
        with util.ExceptionContext(suppress_exceptions=False):
            raise ValueError

def test_printer():
    printer = util.Printer(
        output_filename="test_printer.txt",
        output_dir=OUTPUT_DIR,
        print_to_console=False,
    )
    printer.print("Testing print method")
    printer("Testing __call__ method")
    printer("Testing close method")
    printer.close()

    with pytest.raises(ValueError):
        printer("Checking close method worked")

def test_seeder():
    seeder = util.Seeder()
    seed_list = [
        seeder.get_seed(3, "string", seeder),
        seeder.get_seed(3, "string", seeder),
        seeder.get_seed(3, "string", seeder),
        seeder.get_seed(3, "string", seeder),
        seeder.get_seed(123),
        seeder.get_seed(321),
    ]
    num_seeds = len(seed_list)
    num_unique_seeds = len(set(seed_list))
    assert num_unique_seeds == num_seeds

    printer = util.Printer("test_seeder.txt", OUTPUT_DIR)
    printer("seed_list = %s" % seed_list)

def test_is_numeric():
    assert util.is_numeric(3)
    assert util.is_numeric(3.3)
    assert util.is_numeric(np.double(4))
    assert util.is_numeric(np.long(4))
    assert util.is_numeric(np.uint(4.5))
    assert util.is_numeric(np.linspace(0, 1)[20])
    assert util.is_numeric(np.linspace(0, 1, dtype=float)[20])
    assert util.is_numeric(np.linspace(0, 100, dtype=int)[20])
    assert util.is_numeric(np.linspace(0, 100, dtype=np.uint)[20])
    assert not util.is_numeric(np.linspace(0, 1))
    assert not util.is_numeric("frog")
    assert not util.is_numeric(util)
    assert not util.is_numeric(util.is_numeric)
