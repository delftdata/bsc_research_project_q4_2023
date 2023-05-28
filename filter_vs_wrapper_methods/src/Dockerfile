FROM python:3.9

WORKDIR /filter_vs_wrapper_methods

COPY requirements.txt .

RUN python3.9 -m pip install --upgrade pip setuptools wheel

RUN python3.9 -m pip install -r requirements.txt
    
COPY ./data ./data
COPY ./evaluation ./evaluation
COPY ./methods ./methods
COPY ./ML_models ./ML_models
COPY ./processing ./processing
COPY main.py .