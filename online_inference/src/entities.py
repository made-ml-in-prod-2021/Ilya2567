from typing import List, Union

import numpy as np
from pydantic import BaseModel, conlist, validator

# Поскольку модель, предобучена перечень и последовательность входных данных фиксированы
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
            'oldpeak', 'slope', 'ca', 'thal']


class DiagnosisResponse(BaseModel):
    id: str
    diagnosis: int


class DiagnosisRequest(BaseModel):
    data: List[conlist(Union[float, str], min_items=1, max_items=100)]
    features: List[str]

    @validator('features')
    def validate_model_features(cls, features):
        if not features == FEATURES:
            raise ValueError(f'Invalid features or order! Valid features are: {FEATURES}')
        return features

    @validator('data')
    def validate_number_data_columns_and_features(cls, data):
        if np.array(data).shape[1] != len(FEATURES):
            raise ValueError(f'Invalid columns number for data! Valid numbers are: {len(FEATURES)}')
        return data
