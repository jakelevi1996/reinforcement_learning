import os
import pytest
import numpy as np
import util
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_util")

def test_result():
    """
    Test the Result class, including the get_context method (which indirectly
    calls the save method when the returned context exits) and the load method.
    Also test that changes made to the data during the returned context (IE
    after the Result instance is initialised with the data) are reflected in
    the data that is loaded from file (by a different Result instance) after an
    exception is raised inside the returned context
    """
    output_filename = os.path.join(OUTPUT_DIR, "result_data.pkl")
    data = [1, 2, 3]
    result = util.Result(output_filename, data)
    with result.get_context(save=True, suppress_exceptions=True):
        data[2] *= 4
        raise ValueError()

    loaded_data = util.Result(output_filename).load()
    assert loaded_data == [1, 2, 12]

def test_exception_context():
    """
    Test the ExceptionContext class, including that expressions are suppressed
    when it is initialised with `suppress_exceptions=True` and used as a
    context manager, but not when it is initialised with
    `suppress_exceptions=False` and used as a context manager
    """
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
        with util.ExceptionContext(
            suppress_exceptions=False,
            printer=printer,
        ):
            raise ValueError

def test_printer():
    """
    Test the Printer class, including the print method, the close method, and
    and calling an instance of the Printer class directly. Also test that the
    file that is supposed to be created by the Printer class actually exists
    """
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

    assert os.path.isfile(os.path.join(OUTPUT_DIR, "test_printer.txt"))

def test_seeder():
    """
    Test the Seeder class for generating random seeds and random number
    generators, including the get_seed and get_rng methods
    """
    # Test that two freshly initialised instances of Seeder return the same
    # seed if called with the same arguments but are not invariant to the order
    # of arguments or strings
    assert util.Seeder().get_seed("123") == util.Seeder().get_seed("123")
    assert util.Seeder().get_seed("123") != util.Seeder().get_seed("321")
    assert util.Seeder().get_seed(1, 2, 3) != util.Seeder().get_seed(3, 2, 1)

    # Test that a single instance of Seeder does not return the same seed
    # twice, even when get_seed is called multiple times with the same
    # arguments
    seeder = util.Seeder()
    assert seeder.get_seed("123") != seeder.get_seed("123")

    # Test that get_seed can accept multiple arguments of various different
    # types, and that the seeds returned from a single instance of the Seeder
    # class are always unique
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

    # Test that two freshly initialised instances of Seeder return
    # equivalent random number generators if called with the same arguments
    x1 = util.Seeder().get_rng("test_seeder").normal(size=10)
    x2 = util.Seeder().get_rng("test_seeder").normal(size=10)
    assert x1.size == 10
    assert x2.size == 10
    assert np.all(x1 == x2)

    # Test that two freshly initialised instances of Seeder return different
    # random number generators if get_rng is called with different arguments
    x3 = util.Seeder().get_rng("test_seeder", 2).normal(size=10)
    assert x3.size == 10
    assert not np.all(x1 == x3)
    assert not np.all(x2 == x3)

    # Test that the same instance of Seeder returns different random number
    # generators when called multiple times with the same arguments
    seeder2 = util.Seeder()
    x4 = seeder2.get_rng("test_seeder").normal(size=10)
    x5 = seeder2.get_rng("test_seeder").normal(size=10)
    assert x4.size == 10
    assert x5.size == 10
    assert not np.all(x4 == x5)

def test_is_numeric():
    """
    Test the is_numeric function, and that it returns:

    - True for integers, floats, and various different types of numpy scalars
    - False for instances of non-numeric types (such as strings, modules, and
      functions), numpy arrays, and instances of the Python built-in complex
      type
    """
    assert util.is_numeric(3)
    assert util.is_numeric(3.3)
    assert util.is_numeric(np.double(4))
    assert util.is_numeric(np.long(4))
    assert util.is_numeric(np.uint(4.5))
    assert util.is_numeric(np.linspace(0, 1)[20])
    assert util.is_numeric(np.linspace(0, 1, dtype=float)[20])
    assert util.is_numeric(np.linspace(0, 100, dtype=int)[20])
    assert util.is_numeric(np.linspace(0, 100, dtype=np.uint)[20])
    assert not util.is_numeric(complex(3.3, 4.2))
    assert not util.is_numeric(np.linspace(0, 1))
    assert not util.is_numeric("frog")
    assert not util.is_numeric(util)
    assert not util.is_numeric(util.is_numeric)
