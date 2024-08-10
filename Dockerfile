# Используем официальный образ pytorchlightning в качестве базового
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.12-torch2.4-cuda12.1.0

# Устанавливаем основные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Настраиваем SSH
RUN mkdir /var/run/sshd
RUN echo 'root:password' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Открываем порт для SSH
EXPOSE 22

# Устанавливаем Python-зависимости
COPY ../requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt
# Копируем ваши исходные коды и скрипты в контейнер
COPY .. /app
WORKDIR /app


# Запуск ssh
CMD ["/usr/sbin/sshd", "-D"]
