import json
from typing import Any

import numpy as np

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger

from schemas import Health
from config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = Health(
        name=settings.PROJECT_NAME, api_version="1.0.0", model_version="model_version"
    )

    return health.dict()

