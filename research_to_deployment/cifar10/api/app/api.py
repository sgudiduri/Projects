import numpy as np
from fastapi import APIRouter, File, UploadFile
from schemas import Health
from config import settings
from validation import allowed_file
import cv2

from cifar_10_model.config.config import Config
from cifar_10_model.predict import Predict
from cifar_10_model.processing.data_management import DataService
from cifar_10_model.processing.preprocessors import Preprocessor


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

@api_router.post("/imageclassifier", response_model=dict, status_code=200)
async def classifier(image: UploadFile = File(...)) -> dict:
    """
    Root post
    """    
    if image and allowed_file(image.filename):
        try:
            contents = await image.read()
            nparr = np.fromstring(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            c = Config()
            dm = DataService(c.IMAGE_SIZE, c.BATCH_SIZE,c.TRAINED_MODEL_DIR, c.MODEL_PATH)
            p = Preprocessor(c.IMAGE_SIZE)
            pred = Predict(dm,p)
            results = pred.get_image_results(img)
            return results
        except:
            return {"filename": f"{image.filename} not allowed"}
    else:    
        return {"filename": f"{image.filename} not allowed"}