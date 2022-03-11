all: build run logs

clean:
	docker rm training && docker volume rm training_volume

build:
	docker build -t training_image .

run:
	docker run -v training_volume:/code/app --gpus all --shm-size 64G --privileged -d --name training --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm training_image

logs:
	docker logs training -f 
