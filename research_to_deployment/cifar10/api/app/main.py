from typing import Any

from fastapi import APIRouter, FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
import os
#os.system("pip install opencv-python")

import numpy as np
import cv2

from cifar_10_model.config.config import Config
from cifar_10_model.predict import Predict
from cifar_10_model.processing.data_management import DataService
from cifar_10_model.processing.preprocessors import Preprocessor



api_router = APIRouter()

app = FastAPI(
    title="CIFAR 10 Imgage Classification API", openapi_url=f"/api/v1/openapi.json"
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in set(['png', 'jpg', 'jpeg'])


root_router = APIRouter()


@root_router.get("/")
def index(request: Request) -> Any:
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;background-color: beige;'>"
        "<h1 style='text-align: center; margin-top: 10%;'>Welcome to the Image Classification API</h1>"
        "<div style='text-align: center; font-size: x-large;'>Check the docs: <a href='/docs'>here</a></div>"
        "<div id='hl-aria-live-message-container' aria-live='polite' class='visually-hidden'></div>"
        "<div id='hl-aria-live-alert-container' role='alert' aria-live='assertive' class='visually-hidden'></div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


@root_router.get("/health", response_model=dict, status_code=200)
def health() -> dict:
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
