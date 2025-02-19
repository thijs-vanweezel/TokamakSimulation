#!/usr/bin/env bash
set -euo pipefail

# This script installs oh-my-posh, then updates ~/.bashrc to enable the theme.

# Make sure we can install packages.
# If you're already root inside the container, 'sudo' might not be needed:
sudo apt-get update
sudo apt-get install -y --no-install-recommends unzip curl coreutils

# Download and install oh-my-posh (binary goes to /usr/local/bin/oh-my-posh by default)
curl -s https://ohmyposh.dev/install.sh | bash

# Append line to .bashrc to load your custom theme
# Adjust the path if your JSON is somewhere else
cat << 'EOF' >> /home/ubuntu/.bashrc
# Initialize oh-my-posh with the thecyberden theme
eval "$(oh-my-posh init bash --config /home/ubuntu/thecyberden.omp.json)"
EOF

echo "oh-my-posh installation and configuration complete."
source /home/ubuntu/.bashrc