#!/bin/sh

# Collect static files
echo "Collect static files"
poetry run python manage.py collectstatic --noinput

# Apply database migrations
echo "Apply database migrations"
poetry run python manage.py migrate

# Create or refresh materialized views for map hexagon aggregation
echo "Refreshing materialized views"
poetry run python manage.py refresh_materialized_views

# Start server
echo "Starting server"
poetry run gunicorn -b 0.0.0.0:8000 --timeout 300 --workers 4 djangoproject.wsgi