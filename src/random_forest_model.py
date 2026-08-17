from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Model:
    def __init__(self, clear_data):
        self.y = clear_data['level_Mid senior']
        self.x = clear_data[['time_until_processing', 'type_Onsite', 'type_Remote']]

    def predict(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        print(self.y.value_counts(normalize=True))
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy
