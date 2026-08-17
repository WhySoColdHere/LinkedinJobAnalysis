import pandas as pd


class DataCleaner:
    def __init__(self):
        pd.set_option("display.max_columns", None)
        self.__dirty_data = pd.read_csv('../data/job_postings.csv')

    def get_clear_data(self):
        self.__dirty_data = self.__dirty_data.drop([
            "got_summary", "got_ner", "is_being_worked",
            "search_city", "search_country", "last_status",
            "search_position"], axis=1)

        self.__dirty_data.drop_duplicates(inplace=True)

        self.__dirty_data['first_seen'] = pd.to_datetime(self.__dirty_data['first_seen'], utc=True)
        self.__dirty_data['last_processed_time'] = pd.to_datetime(self.__dirty_data['last_processed_time'],
                                                                  utc=True).dt.floor('s')
        self.__dirty_data['time_until_processing'] = (
                    self.__dirty_data['last_processed_time'] - self.__dirty_data['first_seen']).dt.days

        self.__dirty_data = pd.get_dummies(self.__dirty_data, columns=['job_level'], prefix='level')
        self.__dirty_data = pd.get_dummies(self.__dirty_data, columns=['job_type'], prefix='type', drop_first=True)

        self.__dirty_data[['city', 'state', 'country']] = self.__dirty_data['job_location'].str.replace(' ',
                                                                                                        '').str.split(
            ',',
            expand=True).fillna(
            'Unknown')
        clear_data = self.__dirty_data.drop('job_location', axis=1)

        return clear_data

