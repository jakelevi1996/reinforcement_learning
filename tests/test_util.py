import pytest
import util
import tests.util

RESULTS_DIR = tests.util.get_output_dir("test_util")

def test_result():
    pass

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