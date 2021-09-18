#!/bin/bash

# Fernando Carrió Argos
# 
#

CNN=SRCNN
scale=4

rm ./InputImages/*.jpg
rm ./OutputImages/*.jpg
cp -rf images/4/4_bicubic/*.jpg ./InputImages/
cp ./src/fps_tf_main_SRCNN.cc ./tf_main.cc
make clean
make
mv ./SRCNN ./fps_SRCNN

echo " "
echo "CNN 6"
./fps_SRCNN 6

