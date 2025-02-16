#!/usr/bin/env bash
set -eux

export DISPLAY=:0

# Install dependencies
apt-get update && apt-get install -y \
    firefox-esr \
    libasound2 \
    libdbus-glib-1-2 \
    libfontconfig \
    fonts-liberation \
    libglu1-mesa \
    libgconf-2-4 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-libav \
    libgtk-3-0 \
    libnss3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libxrender1 \
    libxt6 \
    libgl1-mesa-dri \
    x11vnc \
    xvfb \
    fluxbox \
    dbus \
    xdg-utils \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Start dbus
mkdir -p /var/run/dbus
dbus-daemon --system --fork

# Create minimal /tmp/.X11-unix directory
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix

mkdir -p /tmp
chmod 777 /tmp  # Ensure write permissions
touch /tmp/geckodriver.log
chmod 666 /tmp/geckodriver.log 

# Start Xvfb with smaller memory footprint
Xvfb :0 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for Xvfb
until xdpyinfo -display $DISPLAY >/dev/null 2>&1; do
    echo "Waiting for X server..."
    sleep 0.5
done

# Start fluxbox
fluxbox &
FLUXBOX_PID=$!

# Start VNC server
x11vnc -display :0 -forever -nopw -listen 0.0.0.0 -rfbport 5900 &

VNC_PID=$!

echo "X11 environment is ready"

# Create a Firefox profile directory (without headless mode)
firefox --createprofile "selenium"

# Set some Firefox preferences for stability
echo 'user_pref("browser.tabs.remote.autostart", false);' >> ~/.mozilla/firefox/*.selenium/prefs.js
echo 'user_pref("browser.tabs.remote.autostart.2", false);' >> ~/.mozilla/firefox/*.selenium/prefs.js

# Keep the X11 environment running while the container is alive
trap "kill $XVFB_PID $FLUXBOX_PID $VNC_PID" EXIT
wait $XVFB_PID $FLUXBOX_PID $VNC_PID

# 6. (Optional) Launch your Python agent that uses Selenium
#    If you don't want to run agent.py yet, you can comment this out.
# python3 /app/agent.py
