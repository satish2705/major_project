#!/bin/bash

# filepath: c:\Users\USER\OneDrive\Documents\GitHub\New folder\major_project\drug-main\build_files.sh

# Install dependencies
pip install -r requirements.txt

# Collect static files
python manage.py collectstatic --noinput

# Apply database migrations
python manage.py migrate