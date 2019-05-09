FROM python:3.6

RUN pip install pipenv

COPY ./Pipfile ./Pipfile
COPY ./Pipfile.lock ./Pipfile.lock

RUN pipenv sync

COPY . .