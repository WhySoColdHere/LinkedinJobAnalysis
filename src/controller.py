from random_forest_model import Model
from data_cleaning import DataCleaner
from exceptions import UnavailableModelError


class Controller:
    def __init__(self, available_models_count):
        print("Which model you would like to use?\n1 - Random Forest Classifier")
        try:
            self.answer = int(input())
            self.answer = 1

            if self.answer > available_models_count:
                raise UnavailableModelError(
                    f"You are trying to use an unavailable model ({self.answer}). Last available model's number is {available_models_count}")

            self.dc = DataCleaner()

            match self.answer:
                case 1:
                    self._RFC_model()
        except ValueError:
            print("Please enter an integer")
            Controller(available_models_count)

    def _RFC_model(self):
        return Model(**self.dc.get_clean_data()).predict()


controller = Controller(1)
