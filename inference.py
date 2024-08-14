import io
import os

import cv2
import hydra
import logger
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from omegaconf import DictConfig
from PIL import Image
from ultralytics import YOLO

Logger = logger.logging.getLogger()

import wget

app = FastAPI()

gender_age_model: pl.LightningModule
face_detector: YOLO

# Применение преобразований к изображению
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Эндпоинт для обработки изображения
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = np.array(image)

        # Применение модели
        result = gender_age_model.predict(image)

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Эндпоинт для обработки видео
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        video = cv2.VideoCapture(io.BytesIO(contents))

        # Чтение кадров и предсказание
        results = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            result = gender_age_model.predict(frame)
            results.append(result)

        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@hydra.main(version_base=None, config_path="config", config_name="inference")
def inference(cfg: DictConfig):

    if not os.path.exists(cfg.aggefa_path):
        Logger.warning(
            f"Aggefa checkpopint file {cfg.aggefa_path} is not finded. Downloading...."
        )
        wget.download(cfg.aggefa_download_url, out="checkpoints/")
        print()
    else:
        Logger.info("Downloaded Aggefa checkpoint has finded and will be used")

    if not os.path.exists(cfg.yolo_path):
        Logger.warning(
            f"Yolo checkpopint file {cfg.yolo_path} is not finded. Downloading...."
        )
        wget.download(cfg.yolo_download_url, out="checkpoints/")
        print()
    else:
        Logger.info("Downloaded Yolo checkpoint has finded and will be used")

    global gender_age_model
    gender_age_model = hydra.utils.instantiate(cfg.module, _recursive_=False)
    gender_age_model.load_state_dict(torch.load(cfg.aggefa_path)["state_dict"])

    global face_detector
    face_detector = YOLO(cfg.yolo_path)


if __name__ == "__main__":
    inference()
