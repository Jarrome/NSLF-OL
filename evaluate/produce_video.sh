for model in HG # AFFM one_HG
do
	for data_type in replica
	do 
		for data in room0 room1 room2 office0 office1 office2 office3 office4
		do 

folder=outdir/${model}/${data_type}/${data}
ffmpeg -framerate 24 -i ${folder}/image/%d.png ${folder}/image.mp4
done
done
done
