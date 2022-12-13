FROM python:3.9.12

#Create non-root user
RUN useradd -ms /bin/bash python-api
USER pyhton-api

#Copy the source files into the container
WORKDIR /repo
COPY . /repo

#Install pip requirements
RUN python -m pip install -r requirements.txt

#Define the command to be run when the container is started
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["ETL.py"]

