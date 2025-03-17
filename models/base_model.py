import time
import os
import gc
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import root_mean_squared_error as RMSE


class BaseModel:
    def __init__(self, chunksize: int):
        self.file_path = f'{'/'.join(os.getcwd().split('/')[:3])}/year_project'
        self.chunksize = chunksize

        self.test_x = pd.DataFrame()
        self.test_y = pd.DataFrame()

        self.setup_logging()
        self.setup_folders()

        self.scaler = StandardScaler()

        self.models = {
            'european_aqi': {
                'model': SGDRegressor(alpha=0.01, l1_ratio=1, max_iter=1000, tol=0.1, eta0=0.001),
                'path': 'app/models/european_model.pickle'
            }
        }

        self.path = {
            'dataset': f'{self.file_path}/csv_data/air_weather_data_lite.csv',
            'train_x': f'{self.file_path}/aqi_model/train_x_aqi.csv',
            'train_y': f'{self.file_path}/aqi_model/train_y_aqi.csv',
            'test_x': f'{self.file_path}/aqi_model/test_x_aqi.csv',
            'test_y': f'{self.file_path}/aqi_model/test_y_aqi.csv'
        }

    def setup_logging(self) -> None:
        '''
        logging settings
        '''
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s'
                            )

    def setup_folders(self) -> None:
        '''
        folder settings
        '''
        if not os.getcwd().endswith('aqi_prediction'):
            os.chdir('../')
        if not os.path.exists(f'{self.file_path}/aqi_model'):
            os.makedirs(f'{self.file_path}/aqi_model')

    def log(self, level: str, message: str, *args) -> None:
        '''
        Logging messages
        '''

        if level == 'info':
            logging.info(message.format(*args))
        elif level == 'error':
            logging.error(message.format(*args))
        elif level == 'warning':
            logging.warning(message.format(*args))

    def split_data(self) -> None:
        '''
        Splitting dataset at train and test
        '''

        i = 0
        self.log('info', 'Reading dataset')
        with pd.read_csv(self.path['dataset'],
                         chunksize=self.chunksize) as file:
            self.log('info', 'Dataset read')

            for df in file:
                self.log('info', 'Splitting dataset at X and Y')
                X_aqi = df.drop(['european_aqi'], axis=1)
                Y_aqi = df['european_aqi']
                self.log('info', 'Dataset split at X and Y')

                self.log('info', 'Splitting dataset at train and test')
                train_x, test_x, train_y, test_y = train_test_split(
                    X_aqi, Y_aqi, test_size=0.25, random_state=42
                )
                self.log('info', 'Dataset split')

                self.log('info', 'Saving train_x dataset')
                self.save_df(train_x, self.path['train_x'])
                self.log('info', 'train_x dataset saved')

                self.log('info', 'Saving test_x dataset')
                self.save_df(test_x, self.path['test_x'])
                self.log('info', 'test_x dataset saved')

                self.log('info', 'Saving train_y dataset')
                self.save_df(train_y, self.path['train_y'])
                self.log('info', 'train_y dataset saved')

                self.log('info', 'Saving test_y dataset')
                self.save_df(test_y, self.path['test_y'])
                self.log('info', 'test_y dataset saved')

                i += len(df)
                self.log('info', '{} rows processed', i)

    def save_df(self, df: pd.DataFrame, filename: str) -> None:
        '''
        Saving dataset
        '''

        if not os.path.isfile(filename):
            df.to_csv(filename, mode='w', header=True, index=False)
        else:
            df.to_csv(filename, mode='a', header=False, index=False)

    def grid_search(self, models_list, param_grid, cv) -> None:
        '''
        Using gridsearch to find best params
        '''

        self.log('info', 'Reading train data')
        train_x = pd.read_csv(self.path['train_x'],
                              nrows=self.chunksize, header=0).select_dtypes(
                                  ['float', 'int']
                                  )
        train_y = pd.read_csv(self.path['train_y'], nrows=self.chunksize)
        self.log('info', 'Data read')

        self.log('info', 'Filling np.nan')
        for column in train_x.columns:
            train_x[column] = train_x.groupby('city_id')[column].transform(
                lambda x: x.fillna(x.mean()) if x.notna().any()
                else x.fillna(0)
                )
        self.log('info', 'np.nan filled')

        self.log('info', 'Scaling dataset')
        train_x = pd.DataFrame(self.scaler.fit_transform(train_x),
                               columns=train_x.columns
                               )
        self.log('info', 'Dataset scaled')

        for model_name in models_list:
            self.log('info', 'Searching params with gridsearch for {} model',
                     model_name
                     )
            model = self.models[model_name]['model']
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                       scoring='neg_root_mean_squared_error',
                                       cv=cv, n_jobs=-1, verbose=2
                                       )
            train_y_model = np.array(train_y['european_aqi'].fillna(0)).ravel()
            grid_search.fit(train_x, train_y_model)
            self.log('info', 'params found for {} model', model_name)
            self.log('info', 'Best params: {}', grid_search.best_params_)

        self.log('info', '{} rows processed', self.chunksize)

        self.models['grid_search'] = {
            'model': grid_search.best_estimator_,
            'path': 'app/models/grid_search.pickle'
        }
        joblib.dump(self.models['grid_search']['model'],
                    self.models['grid_search']['path']
                    )
        self.model_quality('grid_search')

    def train_model(self) -> None:
        '''
        Training model
        '''

        rows = 1

        while True:
            self.log('info', 'Reading train data')
            train_x = pd.read_csv(self.path['train_x'],
                                  skiprows=range(1, rows),
                                  nrows=self.chunksize,
                                  header=0).select_dtypes(['float', 'int'])

            train_y = pd.read_csv(self.path['train_y'],
                                  skiprows=range(1, rows), nrows=self.chunksize
                                  )
            self.log('info', 'Data read')

            self.log('info', 'Filling np.nan')
            for column in train_x.columns:
                train_x[column] = train_x.groupby('city_id')[column].transform(
                    lambda x: x.fillna(x.mean()) if x.notna().any()
                    else x.fillna(0)
                )
            self.log('info', 'np.nan filled')

            self.log('info', 'Scaling dataset')
            train_x = pd.DataFrame(self.scaler.fit_transform(train_x),
                                   columns=train_x.columns
                                   )
            self.log('info', 'Dataset scaled')

            for model_name in list(self.models.keys()):
                self.log('info', 'training {} model', model_name)
                model = self.models[model_name]['model']
                train_y_model = np.array(train_y['european_aqi'].fillna(
                    train_y['european_aqi'].median())
                ).ravel()
                model.partial_fit(train_x, train_y_model)
                self.log('info', 'Model {} trained', model_name)

            rows += train_y.shape[0]
            self.log('info', '{} rows processed', rows-1)

            if train_y.shape[0] != self.chunksize:
                break

            del train_x
            del train_y
            gc.collect()
            time.sleep(5)
            break

        for model_name in list(self.models.keys()):
            self.log('info', 'Saving {} model', model_name)
            joblib.dump(self.models[model_name]['model'],
                        self.models[model_name]['path']
                        )
            self.log('info', 'Model {} saved', model_name)

        for model_name in list(self.models.keys()):
            self.model_quality(model_name)

    def model_quality(self, model_name: str) -> None:
        '''
        Measuring model quality
        '''

        if self.test_x.empty and self.test_y.empty:
            self.log('info', 'Reading test data')
            self.test_x = pd.read_csv(self.path['test_x'], nrows=self.chunksize).select_dtypes(
                ['int', 'float']
            )
            self.test_y = pd.read_csv(self.path['test_y'], nrows=self.chunksize).select_dtypes(
                ['int', 'float']
            )
            self.log('info', 'Data read')

            self.log('info', 'Filling np.nan')
            for column in self.test_x.columns:
                #self.test_x[column] = self.test_x.groupby('city_id')
                #[column].transform(
                #    lambda x: x.fillna(x.mean()) if x.notna().any()
                #    else x.fillna(0)
                #    )
                self.test_x[column] = self.test_x[column].fillna(
                    self.test_x[column].median()
                    )

            for column in self.test_y.columns:
                self.test_y[column] = self.test_y[column].fillna(
                    self.test_y[column].median()
                    )

            self.log('info', 'np.nan filled')

            self.log('info', 'Scaling dataset')
            self.test_x = pd.DataFrame(self.scaler.transform(self.test_x),
                                       columns=self.test_x.columns
                                       )
            self.log('info', 'Dataset scaled')

        self.log('info', 'Loading {} model', model_name)
        model = joblib.load(self.models[model_name]['path'])
        self.log('info', 'Model {} loaded', model_name)

        self.log('info', 'Predicting AQI')
        pred = model.predict(self.test_x)
        self.log('info', 'AQI predicted')

        self.log('info', 'Calculating metrics')
        r2 = r2_score(self.test_y['european_aqi'], pred)
        mse = MSE(self.test_y['european_aqi'], pred)
        rmse = RMSE(self.test_y['european_aqi'], pred)
        coef = [round(coef, 4) for coef in model.coef_]
        self.log('info', 'Metrics calculated')
        self.log('info', '\n---------------RESULTS:---------------\nModel: {}\
                 \nr2: {}\nMSE: {}\nRMSE: {}\ncoef: {}\n',
                 model_name, r2, mse, rmse,
                 dict(zip(self.test_x.columns, coef))
                 )


base_model = BaseModel(chunksize=100000)
param_grid = {
'alpha': [0, 0.01, 1],
'l1_ratio': [0, 0.5, 1],
'max_iter': [1000],
'tol': [0.01, 0.1],
'eta0': [0.001, 0.1]
}
base_model.grid_search(['european_aqi'], param_grid, cv=3)
# base_model.split_data()
#base_model.train_model()
