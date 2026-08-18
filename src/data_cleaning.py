import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class DataCleaner:
    def __init__(self):
        pd.set_option("display.max_columns", None)
        self.__dirty_data = pd.read_csv('../data/job_postings.csv')
        self.vectorize = TfidfVectorizer(token_pattern=r'(?u)\b\w*[A-Za-z]\w*\b')

    def get_clean_data(self):
        self.__dirty_data.drop_duplicates(inplace=True)

        y = self.__dirty_data['job_level']
        x = self.__dirty_data['job_title'].str.replace(r'\b(senior|lead|associate)\b', '', regex=True, case=False)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

        x_train = self.vectorize.fit_transform(x_train)
        x_test = self.vectorize.transform(x_test)

        return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
