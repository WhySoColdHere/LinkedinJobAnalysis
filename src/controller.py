from random_forest_model import RFC
from logistic_regression_model import LogReg
from pt_neural_network import NN
from data_cleaning import DataCleaner
from exceptions import UnavailableModelError
from comparator import Comparator


class Controller:
    models = {
        1: RFC,
        2: LogReg,
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

        except ValueError as e:
            print(e)
            print("Please enter an integer")
            Controller()

    def run_model(self):
        model = Controller.models[self.answer](**self.dc.get_clean_data())
        model.run()
        model.pretty_results()

    def compare_models(self):
        cleaned_data = self.dc.get_clean_data()
        labels = []
        reports = []
        for i in range(1, len(Controller.models) + 1):
            model = Controller.models[i](is_compare=True, **cleaned_data)
            model.run()
            labels.append(model.__class__.__name__)
            reports.append(model.get_report())

        comparator = Comparator(labels, reports)
        comparator.compare()

controller = Controller()


