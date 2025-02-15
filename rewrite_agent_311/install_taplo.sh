#!/bin/bash
set -e

# Define variables
TAPLO_VERSION="0.8.1"
ARCH=$(uname -m)

# Map architecture names
case ${ARCH} in
    x86_64)
        ARCH="x86_64"
        ;;
    aarch64)
        ARCH="aarch64"
        ;;
    *)
        echo "Unsupported architecture: ${ARCH}"
        exit 1
        ;;
esac

# Download taplo
wget -q "https://github.com/tamasfe/taplo/releases/download/0.8.1/taplo-full-linux-${ARCH}.gz"

# Extract the executable
gunzip "taplo-full-linux-${ARCH}.gz"

# Make it executable
chmod +x "taplo-full-linux-${ARCH}"

# Move to the correct name
mv "taplo-full-linux-${ARCH}" taplo