# aggefa
age, gender &amp; face detector

# Install

### Dependencies:

- Python 3.10.12
- CUDA 12.6

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
