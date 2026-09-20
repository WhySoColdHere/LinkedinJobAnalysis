import pandas as pd
from scipy.sparse import spmatrix


class Model:
    def __init__(self, is_compare, **kwargs):
        self.x_train: spmatrix = kwargs['x_train']
        self.x_test: spmatrix = kwargs['x_test']
        self.y_train: pd.Series = kwargs['y_train']
        self.y_test: pd.Series = kwargs['y_test']
        self.model = None

        self.accuracy = 0
        self.conf_matrix = []
        self._report = None
        self.is_compare = is_compare

    def pretty_results(self):
        print(f'Accuracy: {self.accuracy * 100:.2f}%')
        print(f'Confusion Matrix:\n{self.conf_matrix}')

        print('\nReport:')
        del self._report['accuracy']
        for main_key in self._report:
            print(main_key)
            for key, value in self._report[main_key].items():
                print(f"\t{key}: {value:.2f}")
            print()

    def get_report(self):
        return self._report

    def run(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__} instance"
