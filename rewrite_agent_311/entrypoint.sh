#!/usr/bin/env bash
set -eux

# 1. Install Firefox and VNC packages on-the-fly (since we cannot modify Dockerfile)
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

# 6. (Optional) Launch your Python agent that uses Selenium
#    If you don't want to run agent.py yet, you can comment this out.
# python3 /app/agent.py
