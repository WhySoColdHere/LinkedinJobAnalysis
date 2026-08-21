from pandas import Series
from scipy.sparse import spmatrix


class Model:
    def __init__(self, **kwargs):
        self.x_train: spmatrix = kwargs['x_train']
        self.x_test: spmatrix = kwargs['x_test']
        self.y_train: Series = kwargs['y_train']
        self.y_test: Series = kwargs['y_test']
        self.model = None

    def run(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__} instance"
