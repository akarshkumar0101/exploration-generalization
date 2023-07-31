FROM --platform=linux/amd64 python:3.10

# needed for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install torch==2.0.1 numpy==1.24.3 opencv-python==4.8.0.74
RUN pip install jupyterlab==4.0.2 tqdm==4.65.0 ipywidgets==8.0.7
RUN pip install gym==0.23.1 "gymnasium[atari, accept-rom-license]==0.28.1" procgen==0.10.7 envpool==0.8.2
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . /app/.

CMD echo pwd; ls; which python pip jupyter; lscpu
