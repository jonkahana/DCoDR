#!/bin/bash


cd raw_data
wget https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-data.zip
wget https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
unzip ut-zap50k-data.zip
unzip ut-zap50k-images.zip
mkdir pix2pix
cd pix2pix
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz
tar -xvzf edges2shoes.tar.gz