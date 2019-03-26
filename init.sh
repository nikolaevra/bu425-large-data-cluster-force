#!/usr/bin/env bash
sudo apt-get update
sudo apt install python3-pip
pip install --user virtualenv

sudo apt-get -y install ipython ipython-notebook
pip install --user jupyter

virtualenv env
source env/bin/activate
pip3 install --upgrade pip 
pip3 install requirements.txt

