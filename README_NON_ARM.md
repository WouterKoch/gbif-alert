# Running on non-ARM machines

The pre-built Docker images (`niconoe/gbif-alert:1.7.8`) are built for ARM (Apple Silicon). On x86-64 machines, Docker will emulate ARM to run them, which is extremely slow and causes worker timeouts.

The `docker-compose.override.yml` fixes this by building the app image locally from source, producing a native x86-64 image.

## Setup

1. Clone the app source into this directory:

   ```
   git clone https://github.com/WouterKoch/gbif-alert.git gbif-alert
   ```

2. Edit `local_settings_docker.py` (in this directory, not in the `gbif-alert` subdirectory) with your GBIF credentials and site configuration.

3. Build and start:

   ```
   docker compose up --build
   ```

The override file kicks in automatically and builds native images. The site will be available at `http://localhost:1337`.
