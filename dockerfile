FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install python3 python3-pip -y

RUN echo $(python3 --version)

RUN apt-get update

ENV TZ=Europe
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install python3-opencv -y

#RUN apt-get install ffmpeg libsm6 libxext6 python3-opencv \
#    build-essential cmake pkg-config -y

WORKDIR /code

ADD requirements.txt .

RUN pip3 install -r requirements.txt

COPY app/ /code/app/

WORKDIR /code/app

RUN echo $(ls /code/app)

RUN chmod 755 train.sh

RUN apt-get update 
RUN apt-get install -y nvidia-container-toolkit


CMD ["./train.sh"]


