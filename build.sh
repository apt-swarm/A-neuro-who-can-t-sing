#!/bin/bash
# OFFICAL APT-SWARM BUILD SCRIPT
# MAINTAINER: Nagol12344 
# THIS SCRIPT IS MADE FOR GITHUB ACTIONS, IT DOESNT CARE ABOUT YOUR SYSTEM AND WILL JUST INSTAL PACKAGES 
# YOU HAVE BEEN WARNED

# Move into the build directory
cd build

# Install the required packages
sudo apt install -y python3-stdeb dh-python build-essential fakeroot dh-virtualenv virtualenv python3-full

# Make sure that can be ran
chmod +x debian/rules

# Build package!!
(deactivate ; dpkg-buildpackage -us -uc -b )

# move files that need to be uploaded to the built directory
mkdir -p ../built
mv ../*.deb ../built/

export version "version=$(cat setup.py | grep version | awk -F"'" '{print $2}')"