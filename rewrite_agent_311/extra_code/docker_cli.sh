#!/usr/bin/env bash
set -eux

# Check if Docker CLI is already installed
if ! command -v docker &> /dev/null; then
    echo "Docker CLI not found. Installing..."
    
    # Add Docker's official GPG key and repository
    apt-get update && apt-get install -y \
        ca-certificates \
        curl \
        gnupg

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
      "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker CLI
    apt-get update && apt-get install -y \
        docker-ce-cli \
        && rm -rf /var/lib/apt/lists/*
else
    echo "Docker CLI already installed, skipping installation."
fi

# Set up Docker socket permissions
if [ -e /var/run/docker.sock ]; then
    DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
    # Create docker group if it doesn't exist
    groupadd -g $DOCKER_GID docker 2>/dev/null || true
    # Add current user to docker group
    usermod -aG docker $(whoami) 2>/dev/null || true
fi

# Original package installation
apt-get update && apt-get install -y \
    firefox-esr \
    libgtk-3-0 \
    libdbus-glib-1-2 \
    python3 \
    python3-pip \
    xvfb \
    x11vnc \
    fluxbox \
    && rm -rf /var/lib/apt/lists/*

# 2. Start Xvfb on display :0 with 1024x768x16
Xvfb :0 -screen 0 1024x768x16 &
sleep 1

# 3. Start a lightweight window manager
fluxbox &
sleep 1

# 4. Launch x11vnc (NO PASSWORD) listening on 0.0.0.0:5900
x11vnc -display :0 -forever -nopw -listen 0.0.0.0 &
sleep 2

# 5. Export DISPLAY so Firefox, Selenium, etc. can use the virtual X server
export DISPLAY=:0

# Verify Docker access
docker version || echo "Warning: Docker command failed. Make sure the Docker socket is mounted."

# 6. (Optional) Launch your Python agent that uses Selenium
#    If you don't want to run agent.py yet, you can comment this out.
# python3 /app/agent.py