FROM pytorch/pytorch

# Copy the requirements file first
COPY requirements.txt /requirements.txt

# Install requirements only if they have changed
RUN pip install -r /requirements.txt

    # Install MATLAB Runtime
#COPY MCR_R2021a_glnxa64_installer.zip /tmp/

#RUN unzip /tmp/MCR_R2021a_glnxa64_installer.zip -d /tmp/mcr && \
#    /tmp/mcr/install -mode silent -agreeToLicense yes && \
#    rm -rf /tmp/mcr /tmp/MCR_R2021a_glnxa64_installer.zip

# Install nano text editor
RUN apt-get update && apt-get install nano -y

# Copy the application code
COPY . /src

# Set MATLAB environment variables
#ENV LD_LIBRARY_PATH=/usr/local/MATLAB/MATLAB_Runtime/v910/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v910/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v910/sys/os/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v910/extern/bin/glnxa64
#ENV XAPPLRESDIR=/usr/local/MATLAB/MATLAB_Runtime/v910/X11/app-defaults

# Set the working directory
WORKDIR /src

# ...