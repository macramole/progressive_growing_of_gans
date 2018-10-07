#!/bin/sh
ffmpeg -i generateResult/%5d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p -b:v 3000k out.mp4
