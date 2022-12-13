FROM python:3.9.12

#Create non-root user
RUN useradd -ms /bin/bash python-api
USER pyhton-api

# Set the working directory to /repo
WORKDIR /repo

# Copy the current directory contents into the container at /app
COPY . /repo

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get install -y ant && \
    apt-get clean;

#Define the command to be run when the container is started
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["ETL.py"]