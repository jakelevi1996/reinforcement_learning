import os
import pickle

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

    def __exit__(self, exc_type, exc_value, traceback):
        self._result.save()
