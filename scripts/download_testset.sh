#!/bin/bash

cd data
pip install gdown

# Download our testset
# gdown https://drive.google.com/uc\?id\=1XmEQss80EyMr8e_sRRHA0u9hfrL2QZwJ
unzip testset.zip
rm testset.zip

cd ..