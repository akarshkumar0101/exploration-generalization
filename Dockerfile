FROM --platform=linux/amd64 python:3.10

# set a directory for the app
WORKDIR /app

# needed for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# copy requirements.txt
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy all the files to the container
COPY . exploration-generalization

# define the port number the container should expose
# EXPOSE 5000

# run the command
CMD echo pwd; ls; which python pip jupyter; lscpu
