#!/usr/bin/env bash

#echo Cloning from Remote Github Repository...
#git clone https://github.com/heartcored98/COMET-pytorch comet

# setup.sh should start at comet root directory
mkdir dataset
cd dataset

echo Downloading Dataset from S3 Bucket...
wget -N https://s3.amazonaws.com/comet-dataset/data_xs.tar.gz -O data_xs.tar.gz
wget -N https://s3.amazonaws.com/comet-dataset/data_s.tar.gz -O data_s.tar.gz
wget -N https://s3.amazonaws.com/comet-dataset/data_ms.tar.gz -O data_ms.tar.gz
wget -N https://s3.amazonaws.com/comet-dataset/data_m.tar.gz -O data_m.tar.gz
echo Downloading Complete!

echo Extracting Dataset...
tar -zxvf data_xs.tar.gz
tar -zxvf data_s.tar.gz
tar -zxvf data_ms.tar.gz
tar -zxvf data_m.tar.gz
echo Extracting Complete!

# Install dependency Package
cd ../
pip install --upgrade pip
pip install -r requirements.txt
conda install pytorch-nightly -c pytorch
pip install tensorflow-gpu
cd ../
git clone https://github.com/lanpa/tensorboardX && cd tensorboardX && python setup.py install

cd ../comet