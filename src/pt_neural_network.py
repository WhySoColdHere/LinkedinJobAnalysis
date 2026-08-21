import torch
from base_model import Model
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class JobDataset(Dataset):
    def __init__(self, x, y):
        self.y = (y == 'Mid senior').astype(float)

        self.x = torch.tensor(x.toarray(), dtype=torch.float32)
        self.y = torch.tensor(self.y.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __str__(self):
        return f"X:\n{self.x}\n\nY:\n{self.y}"


class NN(Model, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        nn.Module.__init__(self)

        self.train_dataset = JobDataset(self.x_train, self.y_train)
        self.test_dataset = JobDataset(self.x_test, self.y_test)
        self.batch_size = 32
        self.neurons_count = 128
        self.epochs = 10
        learning_rate = 0.001

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.layer1 = nn.Linear(in_features=self.x_train.shape[1], out_features=self.neurons_count)
        self.layer2 = nn.Linear(in_features=self.neurons_count, out_features=1)
        self.relu = nn.ReLU()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.report = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x

    def _train(self):
        print(f'{"-" * 20} Training {"-" * 20}')
        for epoch in range(self.epochs):
            total_loss = 0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                output = self(x_batch)
                y_batch = y_batch.unsqueeze(1)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / len(self.train_loader)

            print(f"Epoch {epoch + 1}/{self.epochs}"
                  f" --> Loss: {average_loss:.4f}")

    def _test(self):
        print(f'{"-" * 20} Test {"-" * 20}')

        all_predictions = []
        all_targets = []

        self.eval()

        with torch.no_grad():
            for x_batch, y_batch in self.test_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = self(x_batch)
                probability = torch.sigmoid(output)
                prediction = (probability >= 0.5).float()
                y_batch = y_batch.unsqueeze(1)

                all_predictions.extend(prediction.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())

        print(f"Test accuracy: {accuracy_score(all_targets, all_predictions) * 100:.2f}%")
        print(f"Confusion matrix:\n{confusion_matrix(all_targets, all_predictions)}")

        print("Report:")
        report = classification_report(
            all_targets,
            all_predictions,
            target_names=["Associate", "Mid Senior"],
            output_dict=True
        )

        self.report = report

    def run(self):
        self._train()
        print()
        self._test()

        return self.report
