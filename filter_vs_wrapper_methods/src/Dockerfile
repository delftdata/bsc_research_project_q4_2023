FROM python:3.9

WORKDIR /filter_vs_wrapper_methods

COPY requirements.txt .

RUN python3.9 -m pip install --upgrade pip setuptools wheel

RUN python3.9 -m pip install -r requirements.txt

COPY ./evaluator ./evaluator
COPY ./feature_selection_methods ./feature_selection_methods
COPY ./processing ./processing
COPY ./reader ./reader
COPY ./writer ./writer
COPY arguments_main.json .
COPY main.py .