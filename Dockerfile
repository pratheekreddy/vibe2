FROM python:3.8-slim-buster

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev build-essential
RUN pip3 install flask
RUN pip3 install Flask-Session 
RUN pip3 install matplotlib    
RUN pip3 install pandas   
RUN pip3 install scipy     
RUN pip3 install flask_cors   

# We copy just the requirements.txt first to leverage Docker cache

WORKDIR /app

COPY . .
# ENV HOME=/app
WORKDIR /app

# RUN apt-get python -m pip install -U pip
# RUN apt-get python -m pip install -U setuptools
# RUN pip install -r requirements.txt
EXPOSE 5000
# COPY . /app
# EXPOSE 8080
# ENTRYPOINT [ "python" ]

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]