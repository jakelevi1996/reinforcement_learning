import os
import pickle
import traceback
import datetime

class Result:
    def __init__(self, filename, data=None):
        self._data = data
        self._filename = filename

    def get_data(self):
        return self._data

    def get_results_saving_context(self):
        return ResultSavingContext(self)

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
    def __init__(self, result):
        self._result = result

    def __enter__(self):
        return self._result

    def __exit__(self, *args):
        self._result.save()

class BlankContext:
    def __enter__(self):
        return

    def __exit__(self, *args):
        return

class ExceptionContext:
    def __init__(self, suppress_exceptions=True, output_file=None):
        self._suppress_exceptions = suppress_exceptions
        self._file = output_file

    def __enter__(self):
        return

    def __exit__(self, *args):
        if args[0] is not None:
            self._print("%s: An exception occured:" % datetime.datetime.now())
            traceback.print_exception(*args)
            if self._file is not None:
                traceback.print_exception(*args, file=self._file)
            if self._suppress_exceptions:
                self._print("Suppressing exception and continuing...")
                return True

    def _print(self, s):
        print(s)
        if self._file is not None:
            print(s, file=self._file)

def clean_filename(filename_str, allowed_non_alnum_chars="-_.,"):
    filename_str_clean = "".join(
        c if (c.isalnum() or c in allowed_non_alnum_chars) else "_"
        for c in str(filename_str)
    )
    return filename_str_clean
