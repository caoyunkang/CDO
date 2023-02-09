from models.CDO import *
from loguru import logger

def get_model_from_args(**kwargs)->CDOModel:
    model = CDOModel(**kwargs)
    return model