#!/bin/bash

#check if the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
  echo "Virtual environment is not activated. Please activate it and run the script again."
  exit 1
fi

#install dependencies in the virtual environment
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies..."
  pip install -r requirements.txt || { echo "Dependency installation failed. Check your requirements file."; exit 1; }
else
  echo "requirements.txt not found. Skipping dependency installation."
fi

# setting flask environment variables
export FLASK_APP=app.py 
export FLASK_ENV=development 

#run the Flask application
echo "Starting Flask application..."
python3 -m flask run --host=172.25.50.22 --port=5000 || { echo "Failed to start Flask application."; exit 1; }