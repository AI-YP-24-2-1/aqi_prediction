import pandas as pd
from sklearn.linear_model import SGDRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import time
from sklearn.metrics import r2_score, mean_squared_error as MSE, root_mean_squared_error as RMSE
import os
import gc
from sklearn.model_selection import train_test_split
import logging


class BaseModel:
    def __init__(self, model_name: str, chunksize: int):
        self.model_name = model_name
        self.files_path = f'{'/'.join(os.getcwd().split('/')[:3])}/year_project'
        self.chunksize = chunksize

        self.setup_logging()
        self.setup_folders()

        self.models = {
                'regressor': {
                    'european_aqi': SGDRegressor()
                },
                'scaler': {
                    'aqi_scaler': StandardScaler()
                },
                'path': {
                    'dataset': f'{self.files_path}/csv_data/air_weather_data_transformed_without_pollen.csv',
                    'train_x': f'{self.files_path}/{self.model_name}/train_x_aqi.csv',
                    'train_y': f'{self.files_path}/{self.model_name}/train_y_aqi.csv',
                    'test_x': f'{self.files_path}/{self.model_name}/test_x_aqi.csv',
                    'test_y': f'{self.files_path}/{self.model_name}/test_y_aqi.csv',
                    'pickle': f'app/models/{self.model_name}.pickle'
                }
        }

    
    def setup_logging(self) -> None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    

    def setup_folders(self) -> None:
        if not os.getcwd().endswith('.github'):
            os.chdir('../')
        
        if not os.path.exists(f'{self.files_path}/{self.model_name}'):
            os.makedirs(f'{self.files_path}/{self.model_name}')


    def log(self, level: str, message: str, *args) -> None:
        if level == 'info':
            logging.info(message.format(*args))
        elif level == 'error':
            logging.error(message.format(*args))
        elif level == 'warning':
            logging.warning(message.format(*args))
    

    def split_data(self) -> None:
        i = 0
        self.log('info', 'Reading dataset')
        with pd.read_csv(self.models['path']['dataset'], chunksize=self.chunksize) as file:
            self.log('info', 'Dataset read')

            for df in file:

                self.log('info', 'Splitting dataset at X and Y')
                X_aqi = df.drop(['european_aqi'], axis=1)
                Y_aqi = df['european_aqi']
                self.log('info', 'Dataset split at X and Y')

                self.log('info', 'Splitting dataset at train and test')
                train_x_aqi, test_x_aqi, train_y_aqi, test_y_aqi = train_test_split(X_aqi, Y_aqi, test_size=0.25, random_state=42)
                self.log('info', 'Dataset split')

                self.log('info', 'Saving train_x dataset')
                self.save_df(train_x_aqi, self.models['path']['train_x'])
                self.log('info', 'train_x dataset saved')

                self.log('info', 'Saving test_x dataset')
                self.save_df(test_x_aqi, self.models['path']['test_x'])
                self.log('info', 'test_x dataset saved')

                self.log('info', 'Saving train_y dataset')
                self.save_df(train_y_aqi, self.models['path']['train_y'])
                self.log('info', 'train_y dataset saved')

                self.log('info', 'Saving test_y dataset')
                self.save_df(test_y_aqi, self.models['path']['test_y'])
                self.log('info', 'test_y dataset saved')

                i += len(df)
                self.log('info', '{} rows processed', i)
    

    def save_df(self, df: pd.DataFrame, filename: str) -> None:
        if not os.path.isfile(filename):
            df.to_csv(filename, mode='w', header=True, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)
    

    def train_model(self) -> None:
        rows = 1

        while True:
            self.log('info', 'Reading train data')
            train_x = pd.read_csv(self.models['path']['train_x'], skiprows=range(1, rows), nrows=self.chunksize, header=0).select_dtypes(['float', 'int'])
            train_y = pd.read_csv(self.models['path']['train_y'], skiprows=range(1, rows), nrows=self.chunksize)
            self.log('info', 'Data read')

            self.log('info', 'Filling np.nan')
            for column in train_x.columns:
                median_value = train_x[column].median() if not train_x[column].isna().all() else 0
                train_x[column] = train_x[column].fillna(median_value)
            self.log('info', 'np.nan filled')

            self.log('info', 'Scaling dataset')
            train_x = pd.DataFrame(self.models['scaler']['aqi_scaler'].fit_transform(train_x), columns=train_x.columns)
            self.log('info', 'Dataset scaled')

            self.log('info', 'training model')
            model = self.models['regressor']['european_aqi']
            train_y_model = np.array(train_y['european_aqi'].fillna(0)).ravel()
            model.partial_fit(train_x, train_y_model)
            self.log('info', 'Model trained')

            rows += train_y.shape[0]
            self.log('info', '{} rows processed', rows-1)

            if train_y.shape[0] != self.chunksize:
                break

            del train_x
            del train_y
            
            gc.collect()
            time.sleep(5)

        
        self.log('info', 'Saving model')
        joblib.dump(self.models['regressor']['european_aqi'], self.models['path']['pickle'])
        self.log('info', 'Model saved')


    def model_quality(self) -> None:
        self.log('info', 'Reading test data')
        test_x = pd.read_csv(self.models['path']['test_x']).select_dtypes(['int', 'float'])
        test_y = pd.read_csv(self.models['path']['test_y']).select_dtypes(['int', 'float'])
        self.log('info', 'Data read')

        self.log('info', 'Filling np.nan')
        for column in test_x.columns:
            median_value = test_x[column].median() if not test_x[column].isna().all() else 0
            test_x[column] = test_x[column].fillna(median_value)

        for column in test_y.columns:
            test_y[column] = test_y[column].fillna(test_y[column].median())

        self.log('info', 'np.nan filled')

        self.log('info', 'Scaling dataset')
        test_x = pd.DataFrame(self.models['scaler']['aqi_scaler'].transform(test_x), columns=test_x.columns)
        self.log('info', 'Dataset scaled')

        self.log('info', 'Loading model')
        model = joblib.load(self.models['path']['pickle'])
        self.log('info', 'Model loaded')

        self.log('info', 'Predicting AQI')
        pred = model.predict(test_x)
        self.log('info', 'AQI predicted')

        self.log('info', 'Calculating metrics')
        r2 = r2_score(test_y['european_aqi'], pred)
        mse = MSE(test_y['european_aqi'], pred)
        rmse = RMSE(test_y['european_aqi'], pred)
        self.log('info', 'Metrics calculated')
        self.log('info', '\n---------------RESULTS:---------------\nModel: AQI\nr2: {}\nMSE: {}\nRMSE: {}', r2, mse, rmse)


#base_model = BaseModel(model_name='aqi_model', chunksize=20000000)
#base_model.split_data()
#base_model.train_model()
#base_model.model_quality()
