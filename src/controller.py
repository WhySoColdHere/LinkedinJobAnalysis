from random_forest_model import RandomForestClassifierModel
from logistic_regression_model import LogisticRegressionModel
from pt_neural_network import NN
from data_cleaning import DataCleaner
from exceptions import UnavailableModelError
from base_model import Model
import pandas as pd


class Controller:
    models = {
        1: RandomForestClassifierModel,
        2: LogisticRegressionModel,
        3: NN
    }

    def __init__(self):
        print(
            "Which model you would like to use?\n1 - Random Forest Classifier\n2 - Logistic Regression\n3 - Neural Network"
        )
        print("4 - if you want to compare models")

        try:
            available_models_count = len(Controller.models)

            # self.answer = int(input())
            self.answer = 4

            if self.answer > available_models_count and self.answer != 4:
                raise UnavailableModelError(
                    f"You are trying to use an unavailable model ({self.answer}). Last available model's number is {available_models_count}")

            self.dc = DataCleaner()

            if self.answer == 4:
                self.compare_models()
            else:
                self.run_model()

        except ValueError:
            print("Please enter an integer")
            Controller()

    def run_model(self):
        model = Controller.models[self.answer](**self.dc.get_clean_data())
        model.run()
        model.pretty_results()

    def compare_models(self):
        Model.compare_models(self.dc, *self.models.values())


controller = Controller()
