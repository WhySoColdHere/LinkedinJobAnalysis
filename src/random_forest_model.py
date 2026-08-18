from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class Model:
    def __init__(self, **kwargs):
        self.x_train = kwargs['x_train']
        self.x_test = kwargs['x_test']
        self.y_train = kwargs['y_train']
        self.y_test = kwargs['y_test']

    def predict(self):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.x_train, self.y_train)
        predictions = model.predict(self.x_test)

        accuracy = accuracy_score(self.y_test, predictions)
        conf_matrix = confusion_matrix(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Confusion Matrix: \n{conf_matrix}")
        print(f"Report: {report}")

