import pandas as pd
import numpy as np
import os.path
from datetime import datetime


class CsvLite:
    def __init__(self, filepath_to_read, filepath_to_save):
        self.filepath_to_read = filepath_to_read
        self.filepath_to_save = filepath_to_save

    def change_datatype(self, df):

        old_df_size = df.memory_usage(index = False, deep=True).sum() / 1024**2

        df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
        df[df.select_dtypes(np.int64).columns] = df.select_dtypes(np.int64).astype(np.int32)
        df['city_name'] = df['city_name'].astype('category')
        df['region'] = df['region'].astype('category')
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        new_df_size = df.memory_usage(index = False, deep=True).sum() / 1024**2

        return (df, old_df_size, new_df_size)


    def save_df(self, df):
        if not os.path.isfile(self.filepath_to_save):
            df.to_csv(self.filepath_to_save, mode='w', header=True, index=False)
        else:
            df.to_csv(self.filepath_to_save, mode='a', header=False, index=False)


    def transform(self, chunksize):
        old_size = 0
        new_size = 0
        i = 0

        with pd.read_csv(self.filepath_to_read, chunksize=chunksize) as reader:
                for chunk in reader:
                    df, old_df_size, new_df_size = self.change_datatype(chunk)
                    old_size += old_df_size
                    new_size += new_df_size
                    i += chunksize
                    self.save_df(df)
                    print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: {i} rows processed')
        
        print(f'Old size: {round(old_size/1024, 2)} gb\nNew size: {round(new_size/1024, 2)} gb')


csv_lite = CsvLite('air_weather_data.csv', 'air_weather_data_lite.csv')
csv_lite.transform(1000000)

