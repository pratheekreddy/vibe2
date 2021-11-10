FROM python:3.8-slim-buster

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev
RUN pip3 install flask
RUN pip3 install Flask-Session 


COPY requirements.txt ./

RUN pip3 install -r requirements.txt

WORKDIR /app

COPY . .
# ENV HOME=/app
WORKDIR /app

 EXPOSE 5000
# COPY . /app
# EXPOSE 8080
 ENTRYPOINT [ "python3" ]

CMD ["app.py"]


