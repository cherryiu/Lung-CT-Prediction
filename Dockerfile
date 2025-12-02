# Use the official Python 3.12 slim image as the base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the Python dependencies
# The --no-cache-dir flag reduces the size of the final image.
RUN pip install --no-cache-dir -r requirements.txt

# Copy script & utilities into the container
COPY . .

# Command to run when the container starts
# The environment is fully set up, so we just run the Python script
ENTRYPOINT [ "python", "./main.py" ]
CMD []