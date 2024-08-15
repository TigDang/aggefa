# Используем базовый образ Ubuntu 20.04
FROM ubuntu:20.04

# Устанавливаем переменные окружения для установки временной зоны
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Устанавливаем зависимости для Python
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libffi-dev \
    liblzma-dev \
    libgdbm-dev \
    libnss3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libdb5.3-dev

# Добавляем репозиторий с Python 3.10 и обновляем пакеты
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Устанавливаем Python 3.10
RUN apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip

# Открываем порт для сервера
EXPOSE 8000

# Устанавливаем Python-зависимости
COPY ../requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Копируем ваши исходные коды и скрипты в контейнер
COPY .. /app
WORKDIR /app


# Запуск ssh
CMD ["python", "inference.py"]
