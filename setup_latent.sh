#!/bin/bash

conda create -n udvd python=3.11
conda activate udvd
pip install -r requirements.txt

cd dataset
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip
unzip -o DAVIS-2017-trainval-480p.zip
unzip -o DAVIS-2017-test-dev-480p.zip
unzip -o DAVIS-2017-test-challenge-480p.zip
cd ..
