import os
import pytest
import util
import tests.util

RESULTS_DIR = tests.util.get_output_dir("test_util")

def test_result():
    output_filename = os.path.join(RESULTS_DIR, "result_data.pkl")
    data = [1, 2, 3]
    result = util.Result(output_filename, data)
    with result.get_results_saving_context(suppress_exceptions=True):
        data[2] *= 4
        raise ValueError()

    loaded_data = result.load()
    assert loaded_data == [1, 2, 12]

def test_exception_context():
    pass

def test_printer():
    printer = util.Printer(
        output_filename="test_printer.txt",
        output_dir=RESULTS_DIR,
        print_to_console=False,
    )
    printer.print("Testing print method")
    printer("Testing __call__ method")
    printer("Testing close method")
    printer.close()

    with pytest.raises(ValueError):
        printer("Checking close method worked")

def test_seeder():
    pass
