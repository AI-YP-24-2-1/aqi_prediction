import pandas as pd
from sklearn.model_selection import train_test_split
import os.path
from datetime import datetime


def get_data(chunksize):

    i = 0
    with pd.read_csv('csv_data/air_weather_data.csv', chunksize=chunksize) as file:
        for df in file:

            X_aqi = df.drop(['european_aqi'], axis=1)
            Y_aqi = df['european_aqi']

            train_x_aqi, test_x_aqi, train_y_aqi, test_y_aqi = train_test_split(X_aqi, Y_aqi, test_size=0.25, random_state=42)

            save_df(train_x_aqi, 'csv_data/train_x_aqi.csv')
            save_df(test_x_aqi, 'csv_data/test_x_aqi.csv')
            save_df(train_y_aqi, 'csv_data/train_y_aqi.csv')
            save_df(test_y_aqi, 'csv_data/test_y_aqi.csv')

            X_pollen = df.drop(['alder_pollen','birch_pollen','grass_pollen','mugwort_pollen','olive_pollen','ragweed_pollen'], axis=1)
            Y_pollen = df[['alder_pollen','birch_pollen','grass_pollen','mugwort_pollen','olive_pollen','ragweed_pollen']]
            train_x_pollen, test_x_pollen, train_y_pollen, test_y_pollen = train_test_split(X_pollen, Y_pollen, test_size=0.25, random_state=42)

            save_df(train_x_pollen, 'csv_data/train_x_pollen.csv')
            save_df(test_x_pollen, 'csv_data/test_x_pollen.csv')
            save_df(train_y_pollen, 'csv_data/train_y_pollen.csv')
            save_df(test_y_pollen, 'csv_data/test_y_pollen.csv')

            i += len(df)
            print(f'{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}: {i} rows processed')
    
    
    return df

def save_df(df, filename):
    if not os.path.isfile(filename):
        df.to_csv(filename, mode='w', header=True, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

get_data(1000000)
