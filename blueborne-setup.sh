#!/bin/bash
sudo apt update
sudo apt-get install bluetooth libbluetooth-dev 
sudo apt install -y --allow-unauthenticated build-essential virtualenv python2-dev libffi-dev manpages-dev libbluetooth-dev
virtualenv --python=/usr/bin/python2.7 ./venv_hunter
source venv_hunter/bin/activate
pip install pwn==1.0
pip install scapy==2.4.5
pip install PyBluez==0.21
pip install pathlib2==2.3.6
