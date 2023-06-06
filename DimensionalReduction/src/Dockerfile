FROM pytorch/pytorch

# Copy the requirements file first
COPY requirements.txt /requirements.txt

# Install requirements only if they have changed
RUN set -e; \
    if [ -f /requirements.txt ]; then \
        pip3 install -r /requirements.txt; \
    fi

# Install nano text editor
RUN apt-get update && apt-get install nano -y

# Copy the application code
COPY . /src

# Set the working directory
WORKDIR /src

# ...