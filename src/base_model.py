import pandas as pd
from scipy.sparse import spmatrix
from data_cleaning import DataCleaner


class Model:

    def __init__(self, **kwargs):
        self.x_train: spmatrix = kwargs['x_train']
        self.x_test: spmatrix = kwargs['x_test']
        self.y_train: pd.Series = kwargs['y_train']
        self.y_test: pd.Series = kwargs['y_test']
        self.model = None

        self.accuracy = 0
        self.conf_matrix = []
        self.report = None

    def run(self):
        pass

    def pretty_results(self):
        print(f'Accuracy: {self.accuracy * 100:.2f}%')
        print(f'Confusion Matrix:\n{self.conf_matrix}')

        print('\nReport:')
        del self.report['accuracy']
        for main_key in self.report:
            print(main_key)
            for key, value in self.report[main_key].items():
                print(f"\t{key}: {value:.2f}")
            print()

    @staticmethod
    def compare_models(dc: DataCleaner, *args):
        data = dc.get_clean_data()
        for model in args:
            cur_model = model(**data)
            cur_model.run()
            print(f"\n\n{cur_model}:\n{pd.DataFrame(cur_model.report)}")

    def __repr__(self):
        return f"{self.__class__.__name__} instance"
