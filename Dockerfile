FROM python:3.10-slim
LABEL authors="maziar"

COPY src/ src/
WORKDIR src/

RUN pip3 install -r requirements.txt

EXPOSE 8080
