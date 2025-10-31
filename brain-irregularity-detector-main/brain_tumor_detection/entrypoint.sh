#!/bin/bash
# Apply Django database migrations (if any)
# python manage.py migrate --noinput

# Start the Gunicorn server
exec gunicorn brain_tumor_detection.wsgi --bind 0.0.0.0:$PORT
