FROM python:3.8-slim-buster

ENV PYSPARK MAJOR PYTHON VERSION=3

#Create non-root user
RUN useradd -ms /bin/bash python-api
USER pyhton-api

# Set the working directory to /repo
WORKDIR /repo

# Copy the current directory contents into the container at /app
COPY . /repo

# Install any needed packages specified in requirements.txt
RUN python -m pip install -r requirements.txt
RUN pip install --upgrade pip && \

#Define the command to be run when the container is started
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["assigment.py"]