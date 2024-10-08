# aggefa
age, gender &amp; face detector

# Install

### Dependencies:

- Python `3.10.12`
- CUDA `12.6`

### Python environment:

```
git clone https://github.com/TigDang/aggefa
```

```
cd aggefa
```

```
pip install -r requirements.txt
```

### Data:

In this work I use UTKFace dataset from [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new). <i>UTKFace</i> folder is used.

# Training
For training simply run:
```
python train.py
```

Feel free to edit hydra config inside <i>config</i> folder.

Tensorboard server run:
```
tensorboard --logdir tb_logs/
```

# Test
```
python test.py ckpt_path=...
```

# Inference

For that case you must get YOLO weights from HF:

```
wget https://huggingface.co/jaredthejelly/yolov8s-face-detection/resolve/main/YOLOv8-face-detection.pt -P checkpoints/
```

Thanks <b>jaredthejelly</b> for that model weights!

And aggefa weights:

```
wget https://huggingface.co/tigdang/aggefa/resolve/main/aggefa.ckpt -P checkpoints/
```

For json-inference service:

```
python inference.py
```
And then take a look at the `http://127.0.0.1:8000/docs`

# Docker

- Docker version `27.1.1`
- Docker-compose version `1.29.2`


For build container for inference:
```
docker-compose up --build inference
```
