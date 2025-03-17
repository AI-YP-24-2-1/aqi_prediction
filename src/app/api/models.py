from pydantic import BaseModel
from typing import Dict, List, Union, Any


class ApiResponse(BaseModel):
    message: str
    success: bool


class ModelListResponse(BaseModel):
    models: List[Any]


class CompareModelsResponse(BaseModel):
    models_data: Dict


class ApiDataResponse(BaseModel):
    message: str
    data: Union[Dict, None] = None
