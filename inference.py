import asyncio
import io
import os
import tempfile

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
        if len(boxes[0].boxes.cls) == 0:  # Если лиц нет....
            results = [{"boxes xyxy": "no faces :("}]
        else:  # Если лица есть
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
    # Проверяем, что файл видео
    if not file.content_type.startswith("video"):
        raise HTTPException(status_code=400, detail="Invalid file format")

    try:
        # Временное сохранение видеофайла
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(await file.read())
            temp_video_path = (
                temp_video.name
            )  # Получаем путь к временно сохраненному файлу

        # Открываем видео с помощью OpenCV
        cap = cv2.VideoCapture(temp_video_path)

        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        # Читаем и обрабатываем видео кадр за кадром
        results = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Пример обработки: преобразуем кадр в оттенки серого
            # Детекция лиц
            frame = Image.fromarray(frame)
            boxes = face_detector(frame)
            if len(boxes[0].boxes.cls) == 0:
                frame_results = [{"boxes xyxy": "no faces :("}]
            else:
                frame_results = []
                for box in boxes:
                    # Кропим по детекции
                    bbox = box.boxes.xyxy[0]
                    bbox = (
                        bbox[0].item(),
                        bbox[1].item(),
                        bbox[2].item(),
                        bbox[3].item(),
                    )
                    face = np.array(frame.crop(bbox))
                    # Применение модели
                    age, gender = gender_age_model(transform(face)[None, :])
                    age = int(age.item())
                    gender = "Male" if gender.item() > 0.5 else "Female"

                    frame_results.append(
                        {"boxes xyxy": bbox, "age": age, "gender": gender}
                    )

            # Здесь можно добавить любую другую обработку кадра, например, детекцию объектов
            results.append(frame_results)

        # Освобождаем ресурсы
        cap.release()
        os.remove(temp_video_path)  # Удаляем временный файл

        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@hydra.main(version_base=None, config_path="config", config_name="inference")
def inference(cfg: DictConfig):
    # Загружаем веса моделей если их нет
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

    #  Определяем модели глобально, загружаем их веса
    global gender_age_model
    gender_age_model = hydra.utils.instantiate(cfg.module, _recursive_=False)
    gender_age_model.load_state_dict(torch.load(cfg.aggefa_path)["state_dict"])

    global face_detector
    face_detector = YOLO(cfg.yolo_path)

    # res = face_detector(torch.rand(1, 3, 512, 512))

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    inference()
