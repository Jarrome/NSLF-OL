ffmpeg -framerate 10 -i %d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
