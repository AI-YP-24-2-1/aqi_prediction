from fastapi import FastAPI, UploadFile, Form, File, HTTPException, responses, BackgroundTasks
from pydantic import BaseModel
from http import HTTPStatus
import uvicorn
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import numpy as np
from io import StringIO
from typing import Dict, List, Union, Any
from pydantic import BaseModel, RootModel


app = FastAPI()

models = {}


class ApiResponse(BaseModel):
    message: str
    success: bool

class ModelListResponseBase(BaseModel):
    models: List[str]

class ModelListResponse(RootModel[List[ModelListResponseBase]]):
    pass

class ApiResponseBase(BaseModel):
    message: str
    data: Union[Dict, None] = None

class ApiResponseForecast(BaseModel):
    message: str
    data: Dict[str, Any]


def delete_file(file_path: str):
    os.remove(file_path)


@app.post("/fit", response_model=ApiResponse, status_code=HTTPStatus.CREATED)
async def fit(file: UploadFile = File(...), model_name: str = Form(...)):
    contents = await file.read()
    files = [model[:-7] for model in os.listdir('models/')]

    if model_name in models:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="Модель с таким названием уже существует")
    
    if model_name in files:
        raise HTTPException(status_code=HTTPStatus.FORBIDDEN, detail="Модель с таким названием уже существует, но еще не загружена")
    
    models[model_name] = {'scaler': StandardScaler(), 'regressor': SGDRegressor()}

    df = pd.read_csv(StringIO(contents.decode('utf-8')))

    X_aqi = df.drop(['european_aqi'], axis=1).select_dtypes(['float', 'int'])
    Y_aqi = df['european_aqi']

    train_x, test_x, train_y, test_y = train_test_split(X_aqi, Y_aqi, test_size=0.25, random_state=42)

    for column in train_x.columns:
        median_value = train_x[column].median() if not train_x[column].isna().all() else 0
        train_x[column] = train_x[column].fillna(median_value)
    
    train_x = pd.DataFrame(models[model_name]['scaler'].fit_transform(train_x), columns=train_x.columns)
    train_y = np.array(train_y.fillna(0)).ravel()

    model = models[model_name]['regressor']
    model.partial_fit(train_x, train_y)
    joblib.dump(models[model_name]['regressor'], f'models/{model_name}.pickle')

    return ApiResponse(message="Model trained successfully", success=True)


@app.get("/load_main", response_model=ApiResponse)
async def load_main():
    model_name = 'aqi_model'
    
    if model_name not in models.keys():
        try:
            model_pickle = joblib.load(f'models/{model_name}.pickle')
            models[model_name] = {'scaler': StandardScaler(), 'regressor': model_pickle}
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Модель не найдена")
    
    return ApiResponse(message=f"Model {model_name} loaded", success=True)


@app.post("/load", response_model=ApiResponse)
async def load(model_name: str = Form(...)):
    
    if model_name not in models.keys():
        try:
            model_pickle = joblib.load(f'models/{model_name}.pickle')
            models[model_name] = {'scaler': StandardScaler(), 'regressor': model_pickle}
        except Exception as e:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Модель не найдена")
    
    return ApiResponse(message=f"Model {model_name} loaded", success=True)


@app.post("/predict", status_code=HTTPStatus.CREATED)
async def predict(background_tasks: BackgroundTasks, model_name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()

    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    x = pd.DataFrame(df).select_dtypes(['float', 'int'])

    for column in x.columns:
        median_value = x[column].median() if not x[column].isna().all() else 0
        x[column] = x[column].fillna(median_value)
    
    x = pd.DataFrame(models[model_name]['scaler'].fit_transform(x), columns=x.columns)

    if model_name not in models.keys():
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Модель не загружена")
    
    if x.empty:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Отсутствует X")
    
    try:
        model = models[model_name]['regressor']
        pred = model.predict(x)

        df = pd.DataFrame(df)
        df['european_aqi_prediction'] = pred.tolist()
        df.to_csv(f'models/{model_name}_prediction.csv', index=False) 

        background_tasks.add_task(delete_file, f'models/{model_name}_prediction.csv')

        return responses.FileResponse(f'models/{model_name}_prediction.csv', media_type='text/csv', filename=f'{model_name}_prediction.csv')
    
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=f"Возникла ошибка: {str(e)}")


@app.get("/list_models", response_model=ModelListResponseBase)
async def list_models():
    return ModelListResponseBase(models = [model for model in models.keys()])


@app.get("/list_models_not_loaded", response_model=ModelListResponseBase)
async def list_models():
    return ModelListResponseBase(models = [model.replace('.pickle', '') for model in os.listdir('models') if model.replace('.pickle', '') not in models.keys() and model[-6:] == 'pickle'])


@app.delete("/remove_all", response_model=ApiResponse)
async def remove_all():
    global models

    models_list = [model.replace('.pickle', '') for model in os.listdir('models') if model[-6:] == 'pickle']
    for model_name in models_list:
        try:
            if model_name != 'aqi_model':
                os.remove(f'models/{model_name}.pickle')
            del models[model_name]
        except Exception as e:
            continue
    
    return ApiResponse(message=f"Models were removed", success=True)


#if __name__ == "__main__":
#    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
