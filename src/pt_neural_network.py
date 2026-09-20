import torch
from base_model import Model
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, precision_score


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
        return f"X shape: {self.x.shape}\nY shape: {self.y.shape}"


class NN(Model, nn.Module):
    def __init__(self, is_compare=False, **kwargs):
        super().__init__(is_compare, **kwargs)
        nn.Module.__init__(self)

        self.is_compare = is_compare
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

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)

        return x

    def _train(self):
        if not self.is_compare:
            print(f'{"-" * 20} Training {"-" * 20}')

        self.train()

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

            if not self.is_compare:
                print(f"Epoch {epoch + 1}/{self.epochs}"
                      f" --> Loss: {average_loss:.4f}")

    def _test(self):
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

        self.accuracy = accuracy_score(all_targets, all_predictions)
        self.conf_matrix = confusion_matrix(all_targets, all_predictions)
        self._report = classification_report(
            all_targets,
            all_predictions,
            target_names=["Associate", "Mid senior"],
            output_dict=True
        )
        print(f"ROC-AUC score:\n{roc_auc_score(all_predictions, all_targets)}\n")
        print(f"Precision-recall curve\n{precision_recall_curve(all_predictions, all_targets)}")
    def run(self):
        self._train()
        print()
        self._test()


