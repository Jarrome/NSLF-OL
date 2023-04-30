report=outdir/niceslam_surface_plus_HG1SH_replica_report.txt
echo "metrics" >> ${report}
for model in niceslam_surface_plus_HG1SH
do
	for rep_scene in  office4 #office0 office1 office2 office3 office4 room0 room1 room2
	do

		python evaluate/gen_out_im.py configs/replica/${model}/replica_${rep_scene}.yaml

		#folder=./outdir/${model}/replica/${rep_scene}
		#ffmpeg -framerate 24 -i ${folder}/image/%d.png ${folder}/image.mp4

		# evaluate
		echo ${model} >> ${report}
		echo ${rep_scene} >> ${report}
		python evaluate/pointnerf_src/evaluate.py -g ~/data/replica/ReplicaScenes/${rep_scene}/results/rgb/ -i ~/mana_exp/outdir/${model}/replica/${rep_scene}/image/ -l {0..1999} -is '%d.png' -gs 'frame%06d.jpg' >> ${report}
	done
	
done
