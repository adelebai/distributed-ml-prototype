# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY kubernetes_requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "learner.py"]