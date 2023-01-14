FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip uninstall tensorflow
RUN pip install tensorflow-cpu

COPY ./app /app/app

