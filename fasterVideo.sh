ffmpeg -i out.mp4 -filter:v "setpts=0.5*PTS" result.mp4
