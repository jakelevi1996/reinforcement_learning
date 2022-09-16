import os
import pickle
import traceback
import datetime
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "Results")

class Result:
    def __init__(self, filename, data=None):
        self._data = data
        self._filename = filename

    def get_data(self):
        return self._data

    def get_results_saving_context(self, suppress_exceptions=False):
        return ResultSavingContext(self, suppress_exceptions)

    def save(self):
        print("\nSaving results data to \"%s\"..." % self._filename)
        if not os.path.isdir(os.path.dirname(self._filename)):
            os.makedirs(os.path.dirname(self._filename))
        with open(self._filename, "wb") as f:
            pickle.dump(self._data, f)

    def load(self):
        print("Loading results data from \"%s\"..." % self._filename)
        with open(self._filename, "rb") as f:
            self._data = pickle.load(f)
        return self._data

class ResultSavingContext:
    def __init__(self, result, suppress_exceptions):
        self._result = result
        self._suppress_exceptions = suppress_exceptions

    def __enter__(self):
        return self._result

    def __exit__(self, *args):
        self._result.save()
        if self._suppress_exceptions:
            return True

class BlankContext:
    def __enter__(self):
        return

    def __exit__(self, *args):
        return

class ExceptionContext:
    def __init__(self, suppress_exceptions=True, printer=None):
        self._suppress_exceptions = suppress_exceptions
        if printer is None:
            printer = Printer()
        self._print = printer

    def __enter__(self):
        return

    def __exit__(self, *args):
        if args[0] is not None:
            self._print("%s: An exception occured:" % datetime.datetime.now())
            self._print("".join(traceback.format_exception(*args)))
            if self._suppress_exceptions:
                self._print("Suppressing exception and continuing...")
                return True

class Printer:
    def __init__(
        self,
        output_filename=None,
        output_dir=None,
        print_to_console=True,
    ):
        if output_filename is not None:
            if output_dir is None:
                output_dir = RESULTS_DIR

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            output_path = os.path.join(output_dir, output_filename)
            self._file = open(output_path, "w")
        else:
            self._file = None

        self._print_to_console = print_to_console

    def __call__(self, s):
        self.print(s)

    def print(self, s):
        if self._print_to_console:
            print(s)
        if self._file is not None:
            print(s, file=self._file)

    def close(self):
        if self._file is not None:
            self._file.close()

class Seeder:
    def __init__(self):
        self._used_seeds = set()

    def get_seed(self, *args):
        seed = sum(ord(c) for c in str(args))
        while seed in self._used_seeds:
            seed += 1

        self._used_seeds.add(seed)
        return seed

def time_func(func, *args, **kwargs):
    t_start = time.perf_counter()
    func(*args, **kwargs)
    t_total = time.perf_counter() - t_start

    print("\nFinished %r function in %.1fs" % (func.__name__, t_total))

def clean_filename(filename_str, allowed_non_alnum_chars="-_.,"):
    filename_str_clean = "".join(
        c if (c.isalnum() or c in allowed_non_alnum_chars) else "_"
        for c in str(filename_str)
    )
    return filename_str_clean
