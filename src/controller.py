from random_forest_model import RandomForestClassifierModel
from logistic_regression_model import LogisticRegressionModel
from data_cleaning import DataCleaner
from exceptions import UnavailableModelError


class Controller:
    models = {
        1: RandomForestClassifierModel,
        2: LogisticRegressionModel,
    }

    def __init__(self):
        print("Which model you would like to use?\n1 - Random Forest Classifier\n2 - Logistic Regression")
        try:
            available_models_count = len(Controller.models)

            # self.answer = int(input())
            self.answer = 2

            if self.answer > available_models_count:
                raise UnavailableModelError(
                    f"You are trying to use an unavailable model ({self.answer}). Last available model's number is {available_models_count}")

            self.dc = DataCleaner()
            self.run_model()

        except ValueError:
            print("Please enter an integer")
            Controller()

    def run_model(self):
        return Controller.models[self.answer](**self.dc.get_clean_data()).run()


controller = Controller()
