FROM python:3.9.13


WORKDIR /RP


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


COPY datasets /datasets/
COPY FilterWrapper /FilterWrapper/


CMD [ "python", "/FilterWrapper/main.py"]