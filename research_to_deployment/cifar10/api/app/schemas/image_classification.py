from pydantic import BaseModel


class ImageClassification(BaseModel):
    classification: str
    accuracy: str
    message: str