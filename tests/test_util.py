import os
import pytest
import util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "Results", "test_util")

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
