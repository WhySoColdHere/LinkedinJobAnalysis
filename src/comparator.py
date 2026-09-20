import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class Comparator:
    def __init__(self, labels, reports):
        self.labels = labels
        self.reports = reports

    def compare(self):
        models = self.labels
        long_dicts = {
            'model': [],
            'target_class': [],
            'metric': [],
            'value': [],
        }

        accuracy_dicts = {
            'model': [],
            'accuracy': []
        }

        average_dicts = {
            'model': [],
            'metric': [],
            'value': [],
        }

        def add_data(container, **kwargs):
            for k_key, k_value in kwargs.items():
                container[k_key].append(k_value)

        for i in range(len(self.reports)):
            for key, val in self.reports[i].items():
                if isinstance(val, dict):
                    for metric_key, metric_val in val.items():
                        if key in ['Associate', 'Mid senior']:
                            add_data(long_dicts, model=models[i], target_class=key, metric=metric_key, value=metric_val)
                        else:
                            add_data(average_dicts, model=models[i], metric=metric_key, value=metric_val)
                else:
                    add_data(accuracy_dicts, model=models[i], accuracy=val)

        df_long = pd.DataFrame(long_dicts)
        df_accuracy = pd.DataFrame(accuracy_dicts)
        df_average = pd.DataFrame(average_dicts)

        get_class = lambda df, class_name: df[
            (df['target_class'] == class_name) & (df['metric'].isin(['precision', 'recall', 'f1-score']))
            ]

        associate = get_class(df_long, 'Associate')
        mid_senior = get_class(df_long, 'Mid senior')

        get_metrics = lambda inner_df, metric: inner_df[inner_df['metric'] == metric]

        precision_associate = get_metrics(associate, 'precision')
        recall_associate = get_metrics(associate, 'recall')

        precision_mid_senior = get_metrics(mid_senior, 'precision')
        recall_mid_senior = get_metrics(mid_senior, 'recall')

        f1_targets = df_long[
            (df_long['target_class'].isin(['Associate', 'Mid senior'])) & (df_long['metric'] == 'f1-score')
            ]

        fig1, ax1 = plt.subplots(num="Accuracy", figsize=(5, 6))
        ax1.scatter(df_accuracy['model'], df_accuracy['accuracy'])
        ax1.set_ylim(df_accuracy['accuracy'].min() - 0.002, df_accuracy['accuracy'].max() + 0.002)
        ax1.set_title("Models accuracy")

        fig2, ax2 = plt.subplots(1, 2, num='Precision-recall', figsize=(12, 6))
        ax2[0].bar(precision_associate['model'], precision_associate['value'])
        ax2[0].set_title("Precision")
        ax2[1].bar(recall_associate['model'], recall_associate['value'])
        ax2[1].set_title("Recall")
        fig2.suptitle('Associate metrics')
        for ax in ax2:
            ax.set_ylim(0, 1)

        fig3, ax3 = plt.subplots(1, 2, num='Precision-recall-mid-senior', figsize=(12, 6))
        ax3[0].bar(precision_mid_senior['model'], precision_mid_senior['value'])
        ax3[0].set_title("Precision")
        ax3[1].bar(recall_mid_senior['model'], recall_mid_senior['value'])
        ax3[1].set_title("Recall")
        fig3.suptitle('Mid senior metrics')
        for ax in ax3:
            ax.set_ylim(0, 1)

        fig4, ax4 = plt.subplots(1, 2, num="F1-score", figsize=(8, 4))
        ax4[0].bar(x=f1_targets['model'].unique(),
                   height=f1_targets[f1_targets['target_class'] == "Associate"]['value'])
        ax4[0].set_title("Associate F1-score")
        ax4[1].bar(x=f1_targets['model'].unique(),
                   height=f1_targets[f1_targets['target_class'] == "Mid senior"]['value'])
        ax4[1].set_title("Mid senior F1-score")
        ax4[0].set_ylim(0, 1)

        plt.show()
