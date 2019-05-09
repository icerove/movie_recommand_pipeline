FROM python:3.6

RUN pip install pipenv

COPY ./Pipfile ./Pipfile
COPY ./Pipfile.lock ./Pipfile.lock

RUN pipenv sync

RUN curl -O http://files.grouplens.org/datasets/movielens/ml-100k.zip
RUN mkdir -p /root/.surprise_data/ml-100k/
WORKDIR /root/.surprise_data/ml-100k/
RUN unzip /ml-100k.zip
WORKDIR /

COPY . .