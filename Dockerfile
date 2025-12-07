# recommended CUDA base image
FROM nvcr.io/nvidia/tensorflow:24.06-tf2-py3 AS final

# set working directory in container
WORKDIR /usr/src/app

# Install utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential wget curl git && \
    rm -rf /var/lib/apt/lists/* # clean up cache 

# upgrade pip 
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt --upgrade --upgrade-strategy only-if-needed 

# copy src code
COPY . .

ENTRYPOINT [ "python3", "-u", "./main.py" ]
CMD []
