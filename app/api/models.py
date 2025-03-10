from pydantic import BaseModel, RootModel
from typing import Dict, List, Union, Any

class ResultResponse(BaseModel):
    message: str
    success: bool


class ListModelsResponse(BaseModel):
    models: List[Any]


class CompareModelsResponse(BaseModel):
    models_data: Dict


class ModelcomparisionResponse(RootModel[List[ListModelsResponse]]):
    pass


class FitResponse(BaseModel):
    message: str
    data: Union[Dict, None] = None


class ApiForecastResponse(BaseModel):
    message: str
    data: Dict[str, Any]


class ApiTrainedResponse(BaseModel):
    message: str
    data: Dict[str, Any]