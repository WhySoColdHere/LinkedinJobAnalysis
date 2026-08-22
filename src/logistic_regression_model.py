from base_model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class LogisticRegressionModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LogisticRegression()
        self.vectorizer = kwargs['vectorizer']

    def run(self):
        self.model.fit(self.x_train, self.y_train)
        predictions = self.model.predict(self.x_test)

        self.accuracy = accuracy_score(self.y_test, predictions)
        self.conf_matrix = confusion_matrix(self.y_test, predictions)
        self.report = classification_report(self.y_test, predictions, output_dict=True)

