from base_model import Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class LogReg(Model):
    def __init__(self, is_compare=False, **kwargs):
        super().__init__(is_compare, **kwargs)
        self.model = LogisticRegression()
        self.vectorizer = kwargs['vectorizer']

    def run(self):
        self.model.fit(self.x_train, self.y_train)
        predictions = self.model.predict(self.x_test)

        self.accuracy = accuracy_score(self.y_test, predictions)
        self.conf_matrix = confusion_matrix(self.y_test, predictions)
        self._report = classification_report(self.y_test, predictions, output_dict=True)



