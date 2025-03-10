from http import HTTPStatus
import os
from io import StringIO
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Union
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi import responses, BackgroundTasks
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import uvicorn

from api.models import ResultResponse, ListModelsResponse, CompareModelsResponse, FitResponse
from constants import MODELS_SOURCE_DIR, RESULTS_DIR


app = FastAPI()
models = {}

def delete_file(file_path: str):
    '''
    Remove file
    '''
    os.remove(file_path)

def get_loss_list(model, X, y):
    # Обучаем SGDRegressor
    # Для отслеживания функции потерь на каждой итерации воспользуемся методом verbose
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    model.fit(X, y)

    sys.stdout = old_stdout
    loss_history = mystdout.getvalue()

    loss_list = []
    for line in loss_history.split('\n'):
        if(len(line.split("loss: ")) == 1):
            continue
        loss_list.append(float(line.split("loss: ")[-1]))

    return loss_list

def setup_logging() -> None:
    '''
    logging settings
    '''

    log_dir = 'logs'
    log_file_path = os.path.join(log_dir, 'log_file.log')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.isfile(log_file_path):
        open(log_file_path, 'a').close()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[RotatingFileHandler(log_file_path,
                                                      maxBytes=10*1024*1024,
                                                      backupCount=10),
                                  logging.StreamHandler()]
                                  )

def log(level: str, message: str, *args) -> None:
    '''
    Logging messages
    '''

    if level == 'info':
        logging.info(message.format(*args))
    elif level == 'error':
        logging.error(message.format(*args))
    elif level == 'warning':
        logging.warning(message.format(*args))

setup_logging()


@app.post("/fit", status_code=HTTPStatus.CREATED)
async def fit(file: UploadFile = File(...), model_name: str = Form(...),
              alpha: Union[float, None] = Form(...),
              l1_ratio: Union[float, None] = Form(...),
              max_iter: Union[float, None] = Form(...),
              tol: Union[float, None] = Form(...),
              eta0: Union[float, None] = Form(...)
              ):
    '''
    Training model
    '''

    alpha = alpha if alpha != -1 else 0.0001
    l1_ratio = l1_ratio if l1_ratio != -1 else 0.15
    max_iter = max_iter if max_iter != -1 else 1000
    tol = tol if tol != -1 else 0.001
    eta0 = eta0 if eta0 != -1 else 0.01
    log('info', 'alpha = {}, l1_ratio = {}, max_iter = {}, tol = {},'
        ' eta0 = {}', alpha, l1_ratio, max_iter, tol, eta0)

    log('info', 'Reading file')
    contents = await file.read()
    log('info', 'File read')
    files = [model[:-7] for model in os.listdir(MODELS_SOURCE_DIR)]

    if model_name in models:
        log('error', 'Model {} already exists', model_name)
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN,
                            detail="Модель с таким названием уже существует"
                            )

    if model_name in files:
        log('error', 'Model {} already exists but not loaded', model_name)
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN,
                            detail='Модель с таким названием уже существует, '
                            'но еще не загружена'
                            )
    
    if not 0 <= l1_ratio <= 1:
        log('error', 'l1_ratio={}, not in range[0;1]', l1_ratio)
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN,
                            detail='l1_ratio должен быть в диапазоне [0;1]'
                            )

    if alpha < 0:
        log('error', 'alpha={}, less than 0', alpha)
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN,
                            detail='alpha должен быть больше 0'
                            )
    
    if not 1 <= max_iter <= 100000:
        log('error', 'max_iter={}, less than 0', alpha)
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN,
                            detail='max_iter должен быть в диапазоне [1;100000]'
                            )
    
    if not 0 <= max_iter <= 100000:
        log('error', 'tol={}, less than 0', alpha)
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN,
                            detail='tol должен быть в диапазоне [0;100000]'
                            )
    
    if not 0 <= eta0 <= 100000:
        log('error', 'eta0={}, less than 0', alpha)
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN,
                            detail='eta0 должен быть в диапазоне [0;100000]'
                            )

    models[model_name] = {'scaler': StandardScaler(),
                          'regressor': SGDRegressor(alpha=alpha, l1_ratio=l1_ratio, verbose=1, max_iter=max_iter, tol=tol, eta0=eta0)
                          }
    
    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    log('info', 'Splitting at x and y')
    X_aqi = df.drop(['european_aqi'], axis=1).select_dtypes(['float', 'int'])
    Y_aqi = df['european_aqi']
    log('info', 'Dataset split at x and y')

    log('info', 'Splitting dataset and train and test')
    train_x, test_x, train_y, test_y = train_test_split(X_aqi, Y_aqi,
                                                        test_size=0.25,
                                                        random_state=42
                                                        )
    log('info', 'Dataset split at train and test')

    log('info', 'Filling np.nan for train data')
    for column in train_x.columns:
        if not train_x[column].isna().all():
            median_value = train_x[column].median()
        else:
            median_value = 0

        train_x[column] = train_x[column].fillna(median_value)
    log('info', 'np.nan filled for train data')

    log('info', 'Filling np.nan for test data')
    for column in test_x.columns:
        if not test_x[column].isna().all():
            median_value = test_x[column].median()
        else:
            median_value = 0

        test_x[column] = test_x[column].fillna(median_value)
    log('info', 'np.nan filled for test data')

    log('info', 'Scaling train_x')
    train_x = pd.DataFrame(models[model_name]['scaler'].fit_transform(train_x),
                           columns=train_x.columns
                           )
    log('info', 'Train_x scaled')

    log('info', 'Scaling test_x')
    test_x = pd.DataFrame(models[model_name]['scaler'].transform(test_x),
                           columns=test_x.columns
                           )
    log('info', 'Test_x scaled')

    train_y = np.array(train_y.fillna(0)).ravel()
    test_y = np.array(test_y.fillna(0)).ravel()

    log('info', 'Training model')
    model = models[model_name]['regressor']
    loss_list = get_loss_list(model, train_x, train_y)
    log('info', 'Model trained')

    log('info', 'Predicting AQI')
    pred = model.predict(test_x)
    log('info', 'AQI predicted')

    log('info', 'Calculating metrics')
    r2 = round(r2_score(test_y, pred),4)
    mse = round(MSE(test_y, pred), 4)
    rmse = round(RMSE(test_y, pred),4)
    coef = [round(coef, 4) for coef in model.coef_]

    log('info', 'Model: {}\nr2: {}\nMSE: {}\nRMSE: {}\ncoef: {}\n',
                 model_name, r2, mse, rmse,
                 dict(zip(test_x.columns, coef))
                 )
    log('info', 'Metrics calculated for model {}', model_name)

    log('info', 'Saving model')
    joblib.dump(models[model_name]['regressor'], f'{MODELS_SOURCE_DIR}/{model_name}.pickle')
    log('info', 'Model saved')

    coef_dict = {}
    coef_dict['intercept'] = round(float(model.intercept_), 4)
    for i in range(len(test_x.columns)):
        coef_dict[test_x.columns[i]] = coef[i]

    message_text = f"Модель {model_name} обучена"
    data = {
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'max_iter': max_iter,
        'tol': tol,
        'eta0': eta0,
        'r2': r2,
        'MSE': mse, 
        'RMSE': rmse,
        'coef': coef_dict,
        'loss_list': loss_list
    }

    models[model_name]['data'] = data

    return FitResponse(message=message_text, data=data)


@app.get("/load_main", response_model=ResultResponse)
async def load_main():
    '''
    Loading main model
    '''

    model_name = 'aqi_model'

    if model_name not in models:
        try:
            log('info', 'Loading main model')
            model_pickle = joblib.load(f'{MODELS_SOURCE_DIR}/{model_name}.pickle')
            models[model_name] = {'scaler': StandardScaler(),
                                  'regressor': model_pickle
                                  }
            log('info', 'Main model loaded')
        except Exception:
            log('error', 'Model not found')
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND,
                                detail="Модель не найдена"
                                )

    return ResultResponse(message=f"Model {model_name} loaded", success=True)


@app.post("/load", response_model=ResultResponse)
async def load(model_name: str = Form(...)):
    '''
    Loading selected model
    '''

    if model_name not in models:
        try:
            log('info', 'Loading {} model', model_name)
            model_pickle = joblib.load(f'{MODELS_SOURCE_DIR}/{model_name}.pickle')
            models[model_name] = {'scaler': StandardScaler(),
                                  'regressor': model_pickle
                                  }
            log('info', 'Model {} loaded', model_name)
        except Exception:
            log('info', 'Model {} not found', model_name)
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND,
                                detail="Модель не найдена"
                                )

    return ResultResponse(message=f"Model {model_name} loaded", success=True)


@app.post("/predict", status_code=HTTPStatus.CREATED)
async def predict(background_tasks: BackgroundTasks,
                  model_name: str = Form(...),
                  file: UploadFile = File(...)):
    '''
    Predicting AQI
    '''

    log('info', 'Reading file')
    contents = await file.read()
    log('info', 'File read')

    log('info', 'Selecting numeric columns')
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    x = pd.DataFrame(df).select_dtypes(['float', 'int'])
    log('info', 'Numeric columns selected')

    log('info', 'Filling np.nan')
    for column in x.columns:
        median_value = x[column].median() if not x[column].isna().all() else 0
        x[column] = x[column].fillna(median_value)
    log('info', 'np.nan filled')

    log('info', 'Scaling data')
    x = pd.DataFrame(models[model_name]['scaler'].fit_transform(x),
                     columns=x.columns
                     )
    log('info', 'Data scaled')

    if model_name not in models:
        log('error', 'Model {} not loaded', model_name)
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND,
                            detail="Модель не загружена"
                            )

    if x.empty:
        log('error', 'Numeric columns not in x')
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND,
                            detail="Отсутствует X"
                            )

    try:
        log('info', 'Predicting AQI with {} model', model_name)
        model = models[model_name]['regressor']
        pred = model.predict(x)
        log('info', 'AQI predicted')

        df = pd.DataFrame(df)
        df['european_aqi_prediction'] = pred.tolist()

        log('info', 'Saving prediction')
        df.to_csv(f'{RESULTS_DIR}/{model_name}_prediction.csv', index=False)
        log('info', 'Prediction saved')

        background_tasks.add_task(delete_file,
                                  f'{RESULTS_DIR}/{model_name}_prediction.csv'
                                  )

        return responses.FileResponse(f'{RESULTS_DIR}/{model_name}_prediction.csv',
                                      media_type='text/csv',
                                      filename=f'{model_name}_prediction.csv'
                                      )

    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                            detail=f"Возникла ошибка: {str(e)}"
                            )


@app.get("/list_models", response_model=ListModelsResponse)
async def list_models():
    '''
    Showing loaded models
    '''

    log('info', 'Showing loaded models')

    return ListModelsResponse(models=[model for model in models])


@app.get("/list_models_for_comparison", response_model=ListModelsResponse)
async def list_models():
    '''
    Showing models for comparison
    '''

    log('info', 'Showing models for comparison')

    models_list = []

    for model_name in models:
        try:
            data = models[model_name]['data']
            models_list.append(model_name)
        except Exception:
            continue

    return ListModelsResponse(models=models_list)


@app.get("/list_models_not_loaded", response_model=ListModelsResponse)
async def list_models_not_loaded():
    '''
    Showing not loaded models
    '''

    models_list = []
    
    log('info', 'Showing not loaded models')
    for model in os.listdir(MODELS_SOURCE_DIR):
        if (model.replace('.pickle', '') not in models and
                model[-6:] == 'pickle'):

            model = model.replace('.pickle', '')
            models_list.append(model)

    return ListModelsResponse(models=models_list)


@app.delete("/remove_all", response_model=ResultResponse)
async def remove_all():
    '''
    Removing all models
    '''

    log('info', 'Removing all models')
    models_list = []
    for model in os.listdir(MODELS_SOURCE_DIR):
        if model[-6:] == 'pickle':
            model = model.replace('.pickle', '')
            models_list.append(model)

    for model_name in models_list:
        try:
            if model_name != 'aqi_model':
                os.remove(f'{MODELS_SOURCE_DIR}/{model_name}.pickle')
            del models[model_name]
        except Exception:
            continue

    return ResultResponse(message="Models were removed", success=True)


@app.post("/compare_models", response_model=CompareModelsResponse)
async def compare_models(model_name: str = Form(...)):
    '''
    Showing data for selected models
    '''
    log('info', 'Showing data for selected models')
    model_data = models[model_name]['data']
    return CompareModelsResponse(models_data=model_data)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
