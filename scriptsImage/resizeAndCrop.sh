#!/bin/bash
if [ -z "$1" ]
then
      echo "Usage: resizeAndCrop.sh width height images"
      exit
fi
if [ -z "$2" ]
then
      echo "Usage: resizeAndCrop.sh width height images"
      exit
fi
if [ -z "$3" ]
then
      echo "Usage: resizeAndCrop.sh width height images"
      exit
fi
width=$1
height=$2
images=$3

newdir=`dirname $images`"/resize_${width}_${height}"
#echo $newdir
mkdir $newdir
mogrify -resize "${width}x${height}^" -gravity center -crop ${width}x${height}+0+0 +repage -path $newdir $images
