FROM anibali/pytorch:1.10.2-nocuda-ubuntu20.04

WORKDIR /app

COPY . /app

RUN pip install -r  requirements.txt && python test.py