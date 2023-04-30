report=outdir/icl_report.txt
#echo "metrics" > ${report}


for model in HG DSLF AFFM #HG1SH HG DSLF AFFM
do
	#python surface_train_nosurface.py configs/lrkt/lrkt0n_${model}.yaml
	#python surface_train.py configs/lrkt/lrkt0n_${model}.yaml

	python evaluate/gen_out_im.py configs/lrkt/lrkt0n_${model}.yaml
	folder=./outdir/${model}/icl/lrkt0n
	ffmpeg -framerate 24 -i ${folder}/image/%d.png ${folder}/image.mp4
	
	# evaluate
	echo ${model} >> ${report}
	python evaluate/pointnerf_src/evaluate.py -g ~/data/icl/lrkt0n/rgb/ -i ~/mana_exp/outdir/${model}/icl/lrkt0n/image/ -l {0..1508} -is '%s.png' -gs '%s.png' >> ${report}
done
