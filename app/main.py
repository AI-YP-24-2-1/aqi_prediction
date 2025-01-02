from http import HTTPStatus
import os
from io import StringIO
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Union, Any
from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi import responses, BackgroundTasks
from pydantic import BaseModel, RootModel
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
# import uvicorn


app = FastAPI()
models = {}


class ApiResponse(BaseModel):
    message: str
    success: bool


class ModelListResponseBase(BaseModel):
    models: List[Any]


class ModelListResponse(RootModel[List[ModelListResponseBase]]):
    pass


class ApiResponseBase(BaseModel):
    message: str
    data: Union[Dict, None] = None


class ApiResponseForecast(BaseModel):
    message: str
    data: Dict[str, Any]


def delete_file(file_path: str):
    '''
    Remove file
    '''
    os.remove(file_path)

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


@app.post("/fit", response_model=ApiResponse, status_code=HTTPStatus.CREATED)
async def fit(file: UploadFile = File(...), model_name: str = Form(...)):
    '''
    Training model
    '''

    log('info', 'Reading file')
    contents = await file.read()
    log('info', 'File read')
    files = [model[:-7] for model in os.listdir('models/')]

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
    
    models[model_name] = {'scaler': StandardScaler(),
                          'regressor': SGDRegressor()
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

    log('info', 'Filling np.nan')
    for column in train_x.columns:
        if not train_x[column].isna().all():
            median_value = train_x[column].median()
        else:
            median_value = 0

        train_x[column] = train_x[column].fillna(median_value)
    log('info', 'np.nan filled')

    log('info', 'Scaling train_x')
    train_x = pd.DataFrame(models[model_name]['scaler'].fit_transform(train_x),
                           columns=train_x.columns
                           )
    log('info', 'Train_x scaled')

    train_y = np.array(train_y.fillna(0)).ravel()

    log('info', 'Training model')
    model = models[model_name]['regressor']
    model.partial_fit(train_x, train_y)
    log('info', 'Model trained')

    log('info', 'Saving model')
    joblib.dump(models[model_name]['regressor'], f'models/{model_name}.pickle')
    log('info', 'Model saved')

    return ApiResponse(message="Model trained successfully", success=True)


@app.get("/load_main", response_model=ApiResponse)
async def load_main():
    '''
    Loading main model
    '''

    model_name = 'aqi_model'

    if model_name not in models:
        try:
            log('info', 'Loading main model')
            model_pickle = joblib.load(f'models/{model_name}.pickle')
            models[model_name] = {'scaler': StandardScaler(),
                                  'regressor': model_pickle
                                  }
            log('info', 'Main model loaded')
        except Exception:
            log('error', 'Model not found')
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND,
                                detail="Модель не найдена"
                                )

    return ApiResponse(message=f"Model {model_name} loaded", success=True)


@app.post("/load", response_model=ApiResponse)
async def load(model_name: str = Form(...)):
    '''
    Loading selected model
    '''

    if model_name not in models:
        try:
            log('info', 'Loading {} model', model_name)
            model_pickle = joblib.load(f'models/{model_name}.pickle')
            models[model_name] = {'scaler': StandardScaler(),
                                  'regressor': model_pickle
                                  }
            log('info', 'Model {} loaded', model_name)
        except Exception:
            log('info', 'Model {} not found', model_name)
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND,
                                detail="Модель не найдена"
                                )

    return ApiResponse(message=f"Model {model_name} loaded", success=True)


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
        df.to_csv(f'models/{model_name}_prediction.csv', index=False)
        log('info', 'Prediction saved')

        background_tasks.add_task(delete_file,
                                  f'models/{model_name}_prediction.csv'
                                  )

        return responses.FileResponse(f'models/{model_name}_prediction.csv',
                                      media_type='text/csv',
                                      filename=f'{model_name}_prediction.csv'
                                      )

    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                            detail=f"Возникла ошибка: {str(e)}"
                            )


@app.get("/list_models", response_model=ModelListResponseBase)
async def list_models():
    '''
    Showing loaded models
    '''

    log('info', 'Showing loaded models')

    return ModelListResponseBase(models=[model for model in models])


@app.get("/list_models_not_loaded", response_model=ModelListResponseBase)
async def list_models_not_loaded():
    '''
    Showing not loaded models
    '''

    models_list = []
    
    log('info', 'Showing not loaded models')
    for model in os.listdir('models'):
        if (model.replace('.pickle', '') not in models and
                model[-6:] == 'pickle'):

            model = model.replace('.pickle', '')
            models_list.append(model)

    return ModelListResponseBase(models=models_list)


@app.delete("/remove_all", response_model=ApiResponse)
async def remove_all():
    '''
    Removing all models
    '''

    log('info', 'Removing all models')
    models_list = []
    for model in os.listdir('models'):
        if model[-6:] == 'pickle':
            model = model.replace('.pickle', '')
            models_list.append(model)

    for model_name in models_list:
        try:
            if model_name != 'aqi_model':
                os.remove(f'models/{model_name}.pickle')
            del models[model_name]
        except Exception:
            continue

    return ApiResponse(message="Models were removed", success=True)

# if __name__ == "__main__":
# uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
