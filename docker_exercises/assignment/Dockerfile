FROM python:3.8-slim-buster

#Create non-root user
RUN useradd -ms /bin/bash python-api
USER pyhton-api

#Copy the source files into the container
WORKDIR /repo
COPY . /repo

#Install pip requirments
RUN python -m pip install -r requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt 

#Define the command to be run
#when the container is started
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["ETL-project.py"]