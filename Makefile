create:
	sudo docker network create -d bridge FaceGeneration

build:
	sudo docker-compose --env-file .env up -d --build --remove-orphans

stop:
	sudo docker-compose down

start:
	sudo docker-compose --env-file .env up 