#!/bin/bash

# Fernando Carrió Argos
# 
#

CNN=DnCNN
scale=4

rm ./InputImages/*.jpg
rm ./OutputImages/*.jpg
cp -rf images/4/4_bicubic/*.jpg ./InputImages/
cp ./src/fps_tf_main_DnCNN.cc ./tf_main.cc
make clean
make
mv ./DnCNN ./fps_DnCNN

echo " "
echo "CNN 6"
./fps_DnCNN 6

