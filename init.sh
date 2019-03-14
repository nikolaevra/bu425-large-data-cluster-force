#!/usr/bin/env bash

sudo apt install python-pip
pip install --user virtualenv

sudo apt-get -y install ipython ipython-notebook
pip install --user jupyter

virtualenv env
source env/bin/activate
pip install --upgrade pip 
pip install requirements.txt

