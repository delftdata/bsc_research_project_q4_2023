FROM pytorch/pytorch

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

RUN apt-get update && apt-get install nano -y 

COPY . ./scikit-feature