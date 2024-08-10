# Имя Docker-образа
IMAGE_NAME = aggefa_image

# Имя контейнера
CONTAINER_NAME = aggefa_container

# Путь к Dockerfile
DOCKERFILE_PATH = .

# Путь к приложению
APP_PATH = /app

# Сборка Docker-образа
build:
	docker build -t $(IMAGE_NAME) $(DOCKERFILE_PATH)

# Запуск контейнера
run:
	docker run --gpus all -it --rm --name $(CONTAINER_NAME) -v $(PWD):$(APP_PATH) $(IMAGE_NAME)

start-ssh:
	docker run -d --rm --name $(CONTAINER_NAME)-ssh -p 2222:22 -v $(PWD):$(APP_PATH) $(IMAGE_NAME)

# Остановка контейнера
stop:
	docker stop $(CONTAINER_NAME)

# Удаление контейнера
clean:
	docker rm -f $(CONTAINER_NAME)

# Удаление Docker-образа
remove-image:
	docker rmi $(IMAGE_NAME)
