from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from base_model import Model


class RandomForestClassifierModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    def run(self):
        self.model.fit(self.x_train, self.y_train)
        predictions = self.model.predict(self.x_test)

        accuracy = accuracy_score(self.y_test, predictions)
        conf_matrix = confusion_matrix(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Confusion Matrix: \n{conf_matrix}")

        return report
