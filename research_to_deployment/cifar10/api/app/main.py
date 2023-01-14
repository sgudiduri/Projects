from typing import Any

from fastapi import APIRouter, FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from loguru import logger
import os
#os.system("pip install opencv-python")

import numpy as np
import cv2

from cifar_10_model.config.config import Config
from cifar_10_model.predict import Predict
from cifar_10_model.processing.data_management import DataService
from cifar_10_model.processing.preprocessors import Preprocessor

favicon_path = 'favicon.ico'
api_router = APIRouter()

app = FastAPI(
    title="CIFAR 10 Imgage Classification API", openapi_url=f"/api/v1/openapi.json"
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in set(['png', 'jpg', 'jpeg'])


root_router = APIRouter()


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)

@app.get("/")
async def docs_redirect():
    return RedirectResponse(url='/docs')

@root_router.get("/health", response_model=dict, status_code=200)
async def health() -> dict:
    """
    Root Get
    """
    health = dict(
        name="CIFAR 10 Imgage Classification API", api_version="1.0.0", model_version="7.0.0"
    )

    return health

@root_router.post("/imageclassifier", response_model=dict, status_code=200)
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
            img_class = dict(
                classification = list(results.keys())[0],
                accuracy = f"{list(results.values())[0]}%",
                message = "successfully Identified"
            )
        except:
            img_class = dict(
                classification = "",
                accuracy = "",
                message = "oops try again"
            )
            return img_class.dict()
    else:    
        img_class = dict(
            classification = "",
            accuracy = "",
            message = "File type not acceptable"
        )
    
    return img_class
       


app.include_router(root_router)

# #Set all CORS enabled origins
# if __name__ == "__main__":
#     # Use this for debugging purposes only
#     logger.warning("Running in development mode. Do not run like this in production.")
#     import uvicorn

#     uvicorn.run(app, host="localhost", port=8001, log_level="debug")
