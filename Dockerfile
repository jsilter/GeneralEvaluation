# Use an official Python runtime as a parent image
# FROM python:3.8-slim
FROM mitjclinic/sybil:latest

# Install git
RUN apt-get update && apt-get install -y git

# Set the working directory in the container
ENV GE_HOME=/general_eval_app
RUN mkdir ${GE_HOME}
WORKDIR ${GE_HOME}

# Clone the repository
# RUN git clone https://github.com/reginabarzilaygroup/GeneralEvaluation.git .
COPY . .

# Create and activate a virtual environment
RUN python3 -m venv ge_venv
RUN /bin/bash -c "source ${GE_HOME}/ge_venv/bin/activate"

# Install additional dependencies from setup.cfg
RUN ${GE_HOME}/ge_venv/bin/pip install --no-cache-dir .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME=GeneralEval

# Run the application
ENTRYPOINT ["./scripts/start_ark_general_eval.sh", "sybil"]
