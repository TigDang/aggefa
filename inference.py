import asyncio
import io
import os

import aiofiles
import cv2
import hydra
import logger
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
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
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Эндпоинт для обработки изображения
@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Детекция лиц
        boxes = face_detector(image)
        results = []
        for box in boxes:
            # Кропим по детекции
            bbox = box.boxes.xyxy[0]
            bbox = (bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item())
            face = np.array(image.crop(bbox))
            print("2")
            # Применение модели
            age, gender = gender_age_model(transform(face)[None, :])
            age = int(age.item())
            gender = "Male" if gender.item() > 0.5 else "Female"

            results.append({"boxes xyxy": bbox, "age": age, "gender": gender})

        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Эндпоинт для обработки видео
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    try:
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
            try:
                contents = await file.read()
                await temp.write(contents)
            except Exception:
                return {"message": "There was an error uploading the file"}
            finally:
                await file.close()

        res = await run_in_threadpool(
            process_video, temp.name
        )  # Pass temp.name to VideoCapture()

        # Чтение кадров и предсказание
        results = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            # Детекция лиц
            boxes = face_detector(frame)
            results_of_frame = []
            for box in boxes:
                # Кропим по детекции
                bbox = box.boxes.xyxy[0]
                bbox = (bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item())
                face = np.array(frame.crop(bbox))
                print("2")
                # Применение модели
                age, gender = gender_age_model(transform(face)[None, :])
                age = int(age.item())
                gender = "Male" if gender.item() > 0.5 else "Female"

                results_of_frame.append(
                    {"boxes xyxy": bbox, "age": age, "gender": gender}
                )
            results.append(results_of_frame)

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

    # res = face_detector('https://upload.wikimedia.org/wikipedia/commons/6/68/Joe_Biden_presidential_portrait.jpg')

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    inference()
